"""
Stage 7 — Pipeline Orchestration.

Wires together all stages into one callable function:
  Stage 1: Parse (extract raw text from PDF/DOCX)
  Stage 2: Extract (LLM → structured ResumeData/JDData)
  Stage 3: Hard Filter (knockout gate — reject unqualified candidates)
  Stage 4: Score (semantic + LLM scoring per SIMILARITY_MODE)
  Stage 5: Rank (weighted blending, deterministic sort)
  Stage 6: Summarise (LLM-generated strengths/gaps/recommendation)

FIX for persistence gap: previously run_pipeline_with_persistence() only
ever wrote Session, rejected FilterResultRecords, and RankedCandidateRecords
— ResumeRecord, real JD structured_data, CandidateScoreRecord, and
CandidateSummaryRecord were imported but never actually saved. This version
delegates ALL persistence to session_repo.save_full_session(), which saves
every table, in one place, so it's obvious what does and doesn't get
persisted.

Design:
  - _run_pipeline_core() does the actual work ONCE. Both run_pipeline() and
    run_pipeline_with_persistence() call it — no duplicated parsing/
    extraction/scoring between the two entry points (the old version would
    have had to re-run everything a second time to get data for
    persistence, which this avoids).
  - Per-resume error isolation: one bad file doesn't crash the whole batch.
  - All candidates accounted for: rejected candidates appear in output with
    reasons, not silently dropped.
  - Raw resume text is kept (resume_id -> text) all the way through, so
    persistence can save it — previously discarded right after extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from app.core.config import get_settings
from app.models.schemas import ResumeData, JDData
from app.models.results import (
    PipelineResult,
    FilterResult,
    CandidateScores,
    RankedCandidate,
    CandidateSummary,
)
from app.services.parser import parse_document
from app.services.extractor import extract_resume, extract_jd
from app.services.hard_filter import run_hard_filter_bulk
from app.services.scorer_semantic import score_candidates_semantic_bulk
from app.services.scorer_llm import score_candidates_llm_bulk
from app.services.ranker import rank_candidates, load_weights
from app.services.summariser import summarise_candidates_bulk, summarise_rejections_bulk
from app.services.knowledge_brief import generate_briefs_bulk

settings = get_settings()


@dataclass
class _PipelineRun:
    """Everything produced by one full pipeline execution — enough to both
    build the PipelineResult AND persist every stage's output, without
    re-running anything."""
    jd: JDData
    jd_raw_text: str
    extracted_resumes: list[ResumeData]
    resume_raw_texts: dict[str, str]
    parsing_failures: list[FilterResult]
    filter_results: list[FilterResult]
    passed_resumes: list[ResumeData]
    semantic_scores: list[CandidateScores]
    llm_scores: list[CandidateScores]
    ranked_candidates: list[RankedCandidate]
    summaries: list[CandidateSummary]
    knowledge_briefs: list = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)
    total_resumes_submitted: int = 0

    @property
    def all_rejected(self) -> list[FilterResult]:
        return self.parsing_failures + [r for r in self.filter_results if not r.passed]

    def to_pipeline_result(self) -> PipelineResult:
        return PipelineResult(
            jd_title=self.jd.job_title,
            total_resumes_processed=self.total_resumes_submitted,
            total_passed_filter=len(self.passed_resumes),
            total_rejected=len(self.all_rejected),
            rejected_candidates=self.all_rejected,
            ranked_candidates=self.ranked_candidates,
            knowledge_briefs=self.knowledge_briefs,
            similarity_mode=settings.similarity_mode,
        )


def _run_pipeline_core(
    resume_sources: list[tuple[bytes, str]],
    jd_source: tuple[bytes, str],
) -> _PipelineRun:
    """
    Does the actual work, once. Both public entry points below call this.
    """
    logger.info(
        f"Starting pipeline run | resumes={len(resume_sources)} | "
        f"mode={settings.similarity_mode}"
    )

    # ── Stage 1+2: JD ────────────────────────────────────────────────────────
    jd_raw_text = parse_document(jd_source[0], jd_source[1])
    if not jd_raw_text.strip():
        raise ValueError(f"Could not extract text from JD file: {jd_source[1]}")

    try:
        jd = extract_jd(jd_raw_text)
        logger.info(f"JD extracted: title={jd.job_title!r}")
    except Exception as e:
        logger.exception("JD extraction failed")
        raise ValueError(f"JD extraction failed: {e}") from e

    # ── Stage 1+2: Resumes ───────────────────────────────────────────────────
    parsing_failures: list[FilterResult] = []
    extracted_resumes: list[ResumeData] = []
    resume_raw_texts: dict[str, str] = {}  # resume_id -> raw text, kept for persistence

    for raw_bytes, filename in resume_sources:
        try:
            raw_text = parse_document(raw_bytes, filename)
            if not raw_text.strip():
                parsing_failures.append(FilterResult(
                    candidate_name=filename, passed=False,
                    reject_reasons=["Could not extract any text from document"],
                    checks={"parsing": False},
                ))
                continue

            resume = extract_resume(raw_text)
            extracted_resumes.append(resume)
            resume_raw_texts[resume.resume_id] = raw_text

        except Exception as e:
            logger.error(f"Resume parsing/extraction failed for {filename}: {e}")
            parsing_failures.append(FilterResult(
                candidate_name=filename, passed=False,
                reject_reasons=[f"Extraction failed: {e}"],
                checks={"extraction": False},
            ))

    logger.info(
        f"Resume extraction complete | success={len(extracted_resumes)} | "
        f"failed={len(parsing_failures)}"
    )

    empty_run = _PipelineRun(
        jd=jd, jd_raw_text=jd_raw_text,
        extracted_resumes=extracted_resumes, resume_raw_texts=resume_raw_texts,
        parsing_failures=parsing_failures, filter_results=[], passed_resumes=[],
        semantic_scores=[], llm_scores=[], ranked_candidates=[], summaries=[],
        weights_used=load_weights(), total_resumes_submitted=len(resume_sources),
    )

    if not extracted_resumes:
        logger.warning("No resumes successfully extracted.")
        return empty_run

    # ── Stage 3: Hard filter ─────────────────────────────────────────────────
    passed_resumes, filter_results = run_hard_filter_bulk(extracted_resumes, jd)
    logger.info(
        f"Hard filter complete | passed={len(passed_resumes)} | "
        f"rejected={len(filter_results) - len(passed_resumes)}"
    )

    # Generate a short "why + is this worth a second look" summary for every
    # rejected candidate — done unconditionally here (before the early-return
    # below) so this runs even in the "nobody passed" case, not just when
    # scoring/ranking also happens.
    summarise_rejections_bulk(extracted_resumes, jd, filter_results)

    if not passed_resumes:
        logger.warning("No candidates passed the hard filter.")
        empty_run.filter_results = filter_results
        return empty_run

    # ── Stage 4: Score ───────────────────────────────────────────────────────
    semantic_scores: list[CandidateScores] = []
    llm_scores: list[CandidateScores] = []
    mode = settings.similarity_mode

    if mode in ("semantic", "both"):
        semantic_scores = score_candidates_semantic_bulk(passed_resumes, jd)
        logger.info(f"Semantic scoring complete for {len(semantic_scores)} candidates")

    if mode in ("llm", "both"):
        llm_scores = score_candidates_llm_bulk(passed_resumes, jd)
        logger.info(f"LLM scoring complete for {len(llm_scores)} candidates")

    # ── Stage 5: Rank ────────────────────────────────────────────────────────
    ranked_candidates = rank_candidates(passed_resumes, semantic_scores, llm_scores)
    logger.info(
        f"Ranking complete | top="
        f"{ranked_candidates[0].candidate_name if ranked_candidates else 'N/A'}"
    )

    # ── Stage 6: Summarise ───────────────────────────────────────────────────
    summaries = summarise_candidates_bulk(passed_resumes, jd, ranked_candidates)

    summary_by_id = {s.resume_id: s for s in summaries}
    for ranked in ranked_candidates:
        summary = summary_by_id.get(ranked.resume_id)
        if summary:
            ranked.fit_summary = summary.recommendation

    logger.info(f"Summarised {len(summaries)} candidates")

    # ── Stage 7: Knowledge Briefs (top 3 only — this is interview prep, ────
    # not candidate evaluation, so only generate it for people you're
    # actually likely to interview, not the entire passed list.
    knowledge_briefs = generate_briefs_bulk(passed_resumes, jd, ranked_candidates, top_n=3)
    logger.info(f"Generated {len(knowledge_briefs)} knowledge briefs (top 3)")

    return _PipelineRun(
        jd=jd, jd_raw_text=jd_raw_text,
        extracted_resumes=extracted_resumes, resume_raw_texts=resume_raw_texts,
        parsing_failures=parsing_failures, filter_results=filter_results,
        passed_resumes=passed_resumes,
        semantic_scores=semantic_scores, llm_scores=llm_scores,
        ranked_candidates=ranked_candidates, summaries=summaries,
        knowledge_briefs=knowledge_briefs,
        weights_used=load_weights(), total_resumes_submitted=len(resume_sources),
    )


def run_pipeline(
    resume_sources: list[tuple[bytes, str]],
    jd_source: tuple[bytes, str],
) -> PipelineResult:
    """
    Public entry point — no persistence. Used by /pipeline/run when you
    just want the result back without saving it.
    """
    run = _run_pipeline_core(resume_sources, jd_source)
    result = run.to_pipeline_result()
    logger.info(
        f"Pipeline complete | processed={result.total_resumes_processed} | "
        f"passed={result.total_passed_filter} | rejected={result.total_rejected}"
    )
    return result


def run_pipeline_with_persistence(
    resume_sources: list[tuple[bytes, str]],
    jd_source: tuple[bytes, str],
    agency_id: str | None = None,
) -> tuple[PipelineResult, str]:
    """
    Run the pipeline AND save everything — JD, every resume, every filter
    result, both scorers' raw scores, final ranking, and summaries — via
    session_repo.save_full_session(). Returns (PipelineResult, session_id).

    If DATABASE_URL isn't configured, runs normally but skips persistence
    (returns session_id="") rather than failing the whole request — a
    recruiter should still get their ranked results even if the DB is
    temporarily unreachable.
    """
    run = _run_pipeline_core(resume_sources, jd_source)
    result = run.to_pipeline_result()

    if not settings.database_url:
        logger.warning("DATABASE_URL not set — skipping persistence.")
        return result, ""

    try:
        from app.services.session_repo import save_full_session
        session_id = save_full_session(
            jd=run.jd,
            jd_raw_text=run.jd_raw_text,
            resumes=run.extracted_resumes,
            resume_raw_texts=run.resume_raw_texts,
            filter_results=run.filter_results,
            semantic_scores=run.semantic_scores,
            llm_scores=run.llm_scores,
            ranked_candidates=run.ranked_candidates,
            summaries=run.summaries,
            knowledge_briefs=run.knowledge_briefs,
            weights_used=run.weights_used,
            agency_id=agency_id,
        )
        return result, session_id
    except Exception as e:
        logger.error(f"Failed to persist pipeline results: {e}")
        # Persistence failure shouldn't fail the whole run — the recruiter
        # still gets their ranked candidates back, just without history.
        return result, ""