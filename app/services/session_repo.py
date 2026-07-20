"""
session_repo.py — Persistence repository.

FIX for gap #4: the previous run_pipeline_with_persistence() only ever
saved Session, rejected FilterResultRecords, and RankedCandidateRecords —
ResumeRecord, JobDescriptionRecord's real structured_data, and
CandidateScoreRecord/CandidateSummaryRecord were imported but never
actually written. This meant a recruiter asking "what did candidate X's
resume actually say" or "what were the raw scores, not just the final
rank" a day later had no answer — the data simply wasn't there.

This module is the single place that knows how to save a COMPLETE pipeline
run, and how to read one back. Nothing else should write to these tables
directly — that keeps "what gets persisted" in one auditable place.
"""

from __future__ import annotations

import uuid
from typing import Optional

from loguru import logger
from sqlmodel import select

from app.core.database import get_session as get_db_session
from app.models.db_models import (
    Session,
    JobDescriptionRecord,
    ResumeRecord,
    FilterResultRecord,
    CandidateScoreRecord,
    RankedCandidateRecord,
    CandidateSummaryRecord,
    KnowledgeBriefRecord,
)
from app.models.schemas import ResumeData, JDData
from app.models.results import (
    FilterResult,
    CandidateScores,
    RankedCandidate,
    CandidateSummary,
    KnowledgeBrief,
)


def save_full_session(
    jd: JDData,
    jd_raw_text: str,
    resumes: list[ResumeData],
    resume_raw_texts: dict[str, str],
    filter_results: list[FilterResult],
    semantic_scores: list[CandidateScores],
    llm_scores: list[CandidateScores],
    ranked_candidates: list[RankedCandidate],
    summaries: list[CandidateSummary],
    weights_used: dict,
    knowledge_briefs: Optional[list[KnowledgeBrief]] = None,
    agency_id: Optional[str] = None,
) -> str:
    """
    Save EVERYTHING from one full pipeline run in a single transaction.
    Returns the session_id for later lookup.

    resume_raw_texts: {resume_id: raw_text} — the parsed text BEFORE
    extraction, for every resume that made it far enough to be extracted.
    Pass this through from pipeline.py, since run_pipeline() previously
    discarded raw text right after extraction — nowhere to recover it from
    afterward otherwise.

    Every table write happens here, explicitly, so it's obvious at a
    glance what does and doesn't get persisted — no more "imported but
    never .add()'d" gaps.
    """
    session_id = str(uuid.uuid4())

    with get_db_session() as db:
        # 1. Session record
        session_record = Session(
            id=session_id,
            agency_id=agency_id,
            status="processing",
            weights_used=weights_used,
        )
        db.add(session_record)

        # Force this INSERT to actually execute now, before anything that
        # foreign-keys to it. Normally SQLAlchemy's unit-of-work sorts
        # insert order automatically by dependency — this flush is a
        # defensive belt-and-suspenders fix on top of that, in case the
        # connection pooler (see DATABASE_URL — switch off Transaction
        # pooler mode) reassigns backends between statements in a way that
        # breaks that ordering guarantee.
        db.flush()

        # 2. JD — full structured data, not the {} placeholder from before
        db.add(JobDescriptionRecord(
            session_id=session_id,
            agency_id=agency_id,
            raw_text=jd_raw_text,
            structured_data=jd.model_dump(mode="json"),
        ))

        # 3. Every resume that was successfully extracted — structured
        # data AND the raw text it came from, so a recruiter can pull up
        # the original parsed content later, not just the JSON summary.
        for resume in resumes:
            db.add(ResumeRecord(
                id=resume.resume_id,
                session_id=session_id,
                agency_id=agency_id,
                candidate_name=resume.candidate_name,
                raw_text=resume_raw_texts.get(resume.resume_id, ""),
                structured_data=resume.model_dump(mode="json"),
            ))

        # 4. Every filter result — pass AND reject, so rejected candidates
        # are traceable with their reasons, not silently dropped.
        for fr in filter_results:
            if not fr.resume_id:
                continue  # e.g. a parsing-failure entry with no resume_id
            db.add(FilterResultRecord(
                session_id=session_id,
                resume_id=fr.resume_id,
                passed=fr.passed,
                reject_reasons=fr.reject_reasons,
                checks=fr.checks,
                rejection_summary=fr.rejection_summary,
                is_close_miss=fr.is_close_miss,
            ))

        # 5. Both scorer outputs, kept separate by method — this is what
        # lets you later query "how often did semantic and LLM scoring
        # disagree" instead of only ever seeing the already-blended result.
        for cs in semantic_scores + llm_scores:
            if not cs.resume_id:
                continue
            db.add(CandidateScoreRecord(
                session_id=session_id,
                resume_id=cs.resume_id,
                method=cs.method,
                section_scores={s.section_name: s.score for s in cs.section_scores},
            ))

        # 6. Final ranking
        for rc in ranked_candidates:
            if not rc.resume_id:
                continue
            db.add(RankedCandidateRecord(
                session_id=session_id,
                resume_id=rc.resume_id,
                rank=rc.rank,
                overall_score=rc.overall_score,
                score_breakdown=rc.section_scores,
            ))

        # 7. Summaries — was imported in the old pipeline.py but never
        # written at all.
        for summary in summaries:
            if not summary.resume_id:
                continue
            db.add(CandidateSummaryRecord(
                session_id=session_id,
                resume_id=summary.resume_id,
                strengths=summary.strengths,
                gaps=summary.gaps,
                recommendation=summary.recommendation,
            ))

        # 8. Knowledge briefs (top-N candidates only, per pipeline.py)
        for brief in (knowledge_briefs or []):
            if not brief.resume_id:
                continue
            db.add(KnowledgeBriefRecord(
                session_id=session_id,
                resume_id=brief.resume_id,
                brief_data=brief.model_dump(mode="json"),
            ))

        # Mark complete only after every write above succeeded.
        session_record.status = "complete"
        db.add(session_record)

        db.commit()

    logger.info(f"✓ Full session persisted | session_id={session_id} | resumes={len(resumes)} | ranked={len(ranked_candidates)}")
    return session_id


def list_sessions(agency_id: Optional[str] = None, limit: int = 20) -> list[dict]:
    """
    History view — 'what did we run yesterday'. Returns a lightweight
    summary per session, newest first, joined with the JD title so it's
    actually readable rather than just a list of UUIDs.
    """
    with get_db_session() as db:
        query = select(Session).order_by(Session.created_at.desc()).limit(limit)
        if agency_id:
            query = query.where(Session.agency_id == agency_id)
        sessions = db.exec(query).all()

        results = []
        for s in sessions:
            jd_record = db.exec(
                select(JobDescriptionRecord).where(JobDescriptionRecord.session_id == s.id)
            ).first()
            ranked_count = len(db.exec(
                select(RankedCandidateRecord).where(RankedCandidateRecord.session_id == s.id)
            ).all())

            results.append({
                "session_id": s.id,
                "created_at": s.created_at,
                "status": s.status,
                "jd_title": jd_record.structured_data.get("job_title") if jd_record else None,
                "candidates_ranked": ranked_count,
            })
        return results


def get_session_detail(session_id: str) -> dict:
    """
    Full detail view for ONE past session — everything a recruiter would
    want to see again: the JD, every resume, every filter result, both
    scorers' raw scores, final ranking, and summaries. This is the direct
    answer to 'what score did this candidate get last week and why'.
    """
    with get_db_session() as db:
        session_record = db.get(Session, session_id)
        if not session_record:
            raise ValueError(f"No session found with id={session_id}")

        jd_record = db.exec(
            select(JobDescriptionRecord).where(JobDescriptionRecord.session_id == session_id)
        ).first()

        resumes = db.exec(
            select(ResumeRecord).where(ResumeRecord.session_id == session_id)
        ).all()

        filter_results = db.exec(
            select(FilterResultRecord).where(FilterResultRecord.session_id == session_id)
        ).all()

        scores = db.exec(
            select(CandidateScoreRecord).where(CandidateScoreRecord.session_id == session_id)
        ).all()

        ranked = db.exec(
            select(RankedCandidateRecord)
            .where(RankedCandidateRecord.session_id == session_id)
            .order_by(RankedCandidateRecord.rank)
        ).all()

        summaries = db.exec(
            select(CandidateSummaryRecord).where(CandidateSummaryRecord.session_id == session_id)
        ).all()

        briefs = db.exec(
            select(KnowledgeBriefRecord).where(KnowledgeBriefRecord.session_id == session_id)
        ).all()

        return {
            "session": session_record,
            "jd": jd_record,
            "resumes": resumes,
            "filter_results": filter_results,
            "scores": scores,
            "ranked_candidates": ranked,
            "summaries": summaries,
            "knowledge_briefs": briefs,
        }