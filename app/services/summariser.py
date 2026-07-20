"""
Stage 6 — Summariser.

Produces the final, recruiter-facing artifact: strengths, gaps, and a one-
paragraph recommendation per candidate. Grounded strictly in already-
computed facts (skills matched, scores, reject reasons) — never re-derives
or invents them, so a candidate's summary can never disagree with what
Stage 3/4/5 actually computed about them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.schemas import ResumeData, JDData
from app.models.results import RankedCandidate, CandidateSummary
from app.llm.factory import get_llm


class _LLMSummary(BaseModel):
    strengths: list[str] = Field(min_length=1, max_length=5)
    gaps: list[str] = Field(default_factory=list, max_length=4)
    recommendation: str


SUMMARY_SYSTEM_PROMPT = """
You are writing a candidate summary for a recruiter who is about to screen
this person. You will be given: the candidate's actual resume facts, the
JD's actual requirements, and scores that have ALREADY been computed by a
separate scoring system.

Rules — follow these exactly:
1. Do NOT recompute or restate any number differently than given (e.g. if
   given experience_months=6, never say "several years of experience").
2. Every strength must reference a SPECIFIC skill, project, or role from
   the resume — not a generic statement like "strong technical background."
3. Every gap must reference something SPECIFIC the JD asks for that the
   resume doesn't clearly show — not a vague "could improve" statement.
   If there are no real gaps, return an empty gaps list. Do not invent a
   gap just to have one.
4. The recommendation must be consistent with the overall_score given —
   a high score should read as a positive recommendation, a low score
   should read as a cautious one. Reference the score's implication in
   plain language, not the raw number itself.
5. Never mention information that isn't in the provided resume/JD facts.
"""


def _build_summary_prompt(
    resume: ResumeData, jd: JDData, ranked: RankedCandidate,
    reject_reasons: list[str] | None = None,
) -> str:
    experience_lines = [
        f"- {job.title} at {job.company} ({job.duration_months or '?'} months): "
        f"{'; '.join(job.responsibilities) if job.responsibilities else 'no listed responsibilities'}"
        for job in resume.work_experience
    ]
    project_lines = [f"- {p.name}: {p.description}" for p in resume.projects]
    scores_text = "\n".join(f"  {name}: {score:.2f}/1.0" for name, score in ranked.section_scores.items())
    reject_text = ""
    if reject_reasons:
        reject_text = f"\nNote: this candidate had the following filter concerns: {'; '.join(reject_reasons)}"

    return f"""
<job_description>
Title: {jd.job_title}
Must-have skills: {', '.join(jd.must_have_skills) or 'none specified'}
Must-have skill groups (any one per group): {jd.must_have_skill_groups or 'none'}
Nice-to-have skills: {', '.join(jd.nice_to_have_skills) or 'none specified'}
Responsibilities: {'; '.join(jd.responsibilities) or 'none specified'}
</job_description>

<candidate_resume>
Name: {resume.candidate_name}
Total experience: {resume.total_experience_months or 0} months
Skills: {', '.join(resume.skills) or 'none listed'}
Work experience:
{chr(10).join(experience_lines) or '  none listed'}
Projects:
{chr(10).join(project_lines) or '  none listed'}
</candidate_resume>

<computed_scores>
Overall score: {ranked.overall_score:.2f}/1.0 (rank #{ranked.rank})
{scores_text}
</computed_scores>{reject_text}

Write the strengths, gaps, and recommendation now, grounded strictly in the above.
""".strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_summariser(system: str, user: str) -> _LLMSummary:
    llm = get_llm()
    return llm.structured_completion(
        system_prompt=system, user_prompt=user, response_model=_LLMSummary, temperature=0.0,
    )


def summarise_candidate(
    resume: ResumeData, jd: JDData, ranked: RankedCandidate,
    reject_reasons: list[str] | None = None,
) -> CandidateSummary:
    logger.debug(f"Summarising candidate={resume.candidate_name!r} (rank={ranked.rank})")
    raw = _call_llm_summariser(
        system=SUMMARY_SYSTEM_PROMPT,
        user=_build_summary_prompt(resume, jd, ranked, reject_reasons),
    )
    summary = CandidateSummary(
        resume_id=resume.resume_id, candidate_name=resume.candidate_name,
        strengths=raw.strengths, gaps=raw.gaps, recommendation=raw.recommendation,
    )
    logger.info(f"✓ Summarised {resume.candidate_name!r}: {len(raw.strengths)} strengths, {len(raw.gaps)} gaps")
    return summary


def summarise_candidates_bulk(
    resumes: list[ResumeData], jd: JDData, ranked_candidates: list[RankedCandidate],
) -> list[CandidateSummary]:
    resumes_by_id = {r.resume_id: r for r in resumes}
    summaries: list[CandidateSummary] = []
    for ranked in ranked_candidates:
        resume = resumes_by_id.get(ranked.resume_id)
        if resume is None:
            logger.error(f"No resume found for resume_id={ranked.resume_id} — skipping summary.")
            continue
        try:
            summaries.append(summarise_candidate(resume, jd, ranked))
        except Exception as e:
            logger.error(f"Summary generation failed for {resume.candidate_name!r}: {e}")
            summaries.append(
                CandidateSummary(
                    resume_id=resume.resume_id, candidate_name=resume.candidate_name,
                    strengths=[], gaps=[],
                    recommendation=f"Summary generation failed: {e}. Please review this candidate manually.",
                )
            )
    return summaries


# ── Rejection summaries — the recruiter-facing "why, and is it worth a second look" ──

class _LLMRejectionSummary(BaseModel):
    is_close_miss: bool = Field(
        description=(
            "True ONLY if the candidate is missing just 1-2 requirements and is "
            "otherwise clearly well-qualified — genuinely worth a recruiter's second "
            "look. False if the mismatch is broad (missing several requirements, "
            "wrong domain entirely, far below experience level, etc.)."
        )
    )
    summary: str = Field(
        description=(
            "2-3 sentences for a recruiter: state plainly why this candidate was "
            "rejected (reference the SPECIFIC missing requirement(s)), then note "
            "what relevant strengths they DO have despite the rejection, if any. "
            "If is_close_miss is True, say so explicitly and suggest it may be "
            "worth a manual look."
        )
    )


REJECTION_SUMMARY_SYSTEM_PROMPT = """
You are writing a short note for a recruiter explaining why a candidate was
automatically rejected by an exact-match hard filter, and whether the
rejection might be worth double-checking manually.

Rules:
1. State the SPECIFIC missing requirement(s) — never a vague "doesn't meet
   requirements."
2. If the candidate clearly has strong, directly relevant experience/skills
   despite the technical rejection, say so — a hard filter rejects on exact
   text matching and can be wrong (e.g. a candidate may have equivalent
   experience phrased differently, or be missing something genuinely minor).
3. Set is_close_miss=True ONLY when the gap is small (1-2 missing items, and
   the candidate is otherwise a strong, clear match) — not for every
   rejection. Most rejections are NOT close misses; be honest and
   conservative here, don't inflate this to be reassuring.
4. Never invent facts not present in the resume or JD provided.
"""


def _build_rejection_prompt(resume: ResumeData, jd: JDData, reject_reasons: list[str]) -> str:
    return f"""
<job_description>
Title: {jd.job_title}
Must-have skills: {', '.join(jd.must_have_skills) or 'none specified'}
Must-have skill groups (any one per group): {jd.must_have_skill_groups or 'none'}
Min experience: {jd.min_experience_months or 0} months
</job_description>

<candidate_resume>
Name: {resume.candidate_name}
Total experience: {resume.total_experience_months or 0} months
Skills: {', '.join(resume.skills) or 'none listed'}
</candidate_resume>

<rejection_reasons_from_hard_filter>
{chr(10).join('- ' + r for r in reject_reasons)}
</rejection_reasons_from_hard_filter>

Write the rejection summary now.
""".strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _call_llm_rejection_summariser(system: str, user: str) -> _LLMRejectionSummary:
    llm = get_llm()
    return llm.structured_completion(
        system_prompt=system, user_prompt=user, response_model=_LLMRejectionSummary, temperature=0.0,
    )


def summarise_rejection(resume: ResumeData, jd: JDData, reject_reasons: list[str]) -> tuple[str, bool]:
    """
    Returns (summary_text, is_close_miss) for ONE rejected candidate.
    Called from pipeline.py for every hard-filter reject that has a real
    ResumeData behind it (not parsing failures, which have nothing to
    summarise against).
    """
    logger.debug(f"Summarising rejection for {resume.candidate_name!r}")
    try:
        raw = _call_llm_rejection_summariser(
            system=REJECTION_SUMMARY_SYSTEM_PROMPT,
            user=_build_rejection_prompt(resume, jd, reject_reasons),
        )
        return raw.summary, raw.is_close_miss
    except Exception as e:
        logger.error(f"Rejection summary failed for {resume.candidate_name!r}: {e}")
        # Fall back to the raw reject reasons rather than nothing — still
        # useful to a recruiter even without the LLM writeup.
        return f"Rejected: {'; '.join(reject_reasons)}", False


def summarise_rejections_bulk(
    resumes: list[ResumeData], jd: JDData, rejected_filter_results: list,
) -> None:
    """
    Mutates each FilterResult in-place, setting rejection_summary and
    is_close_miss. Only processes results that have both a resume_id AND
    a matching ResumeData (parsing failures have neither and are skipped).
    """
    resumes_by_id = {r.resume_id: r for r in resumes}
    for fr in rejected_filter_results:
        if fr.passed or not fr.resume_id:
            continue
        resume = resumes_by_id.get(fr.resume_id)
        if resume is None:
            continue
        summary, is_close_miss = summarise_rejection(resume, jd, fr.reject_reasons)
        fr.rejection_summary = summary
        fr.is_close_miss = is_close_miss