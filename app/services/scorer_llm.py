"""
Stage 4B — LLM Scoring (provider-agnostic version).

Uses LLM to score each component with temperature=0 for determinism.
Provider-agnostic via app.llm.factory.get_llm() — works with Groq, OpenAI,
or Anthropic without any branching logic in this file.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from app.models.schemas import ResumeData, JDData
from app.models.results import CandidateScores, SectionScore
from app.llm.factory import get_llm


class LLMScoringResponse(BaseModel):
    must_have_skills_score: float = Field(..., ge=0.0, le=1.0)
    good_to_have_skills_score: float = Field(..., ge=0.0, le=1.0)
    domain_knowledge_score: float = Field(..., ge=0.0, le=1.0)
    certifications_score: float = Field(..., ge=0.0, le=1.0)
    total_experience_score: float = Field(..., ge=0.0, le=1.0)
    role_match_score: float = Field(..., ge=0.0, le=1.0)
    project_relevance_score: float = Field(..., ge=0.0, le=1.0)
    summary_score: float = Field(..., ge=0.0, le=1.0)
    skills_reasoning: str = Field(default="")
    experience_reasoning: str = Field(default="")


def _call_llm_scorer(resume: ResumeData, jd: JDData) -> LLMScoringResponse:
    resume_summary = f"""
Candidate: {resume.candidate_name}
Total Experience: {resume.total_experience_months or 0} months
Skills: {', '.join(resume.skills[:30])}
Programming Languages: {', '.join(resume.programming_languages)}
Frameworks & Tools: {', '.join(resume.frameworks_and_tools)}
Projects: {len(resume.projects)} projects
Certifications: {', '.join(c.name for c in resume.certifications) if resume.certifications else 'None'}
"""
    if resume.work_experience:
        resume_summary += "\nWork Experience:\n"
        for job in resume.work_experience[:5]:
            resume_summary += f"  - {job.title} at {job.company} ({job.duration_months or 0} months)\n"

    if resume.projects:
        resume_summary += "\nProjects:\n"
        for proj in resume.projects[:3]:
            resume_summary += f"  - {proj.name}: {', '.join(proj.technologies[:5])}\n"

    jd_summary = f"""
Job Title: {jd.job_title}
Company: {jd.company or 'Not specified'}
Location: {jd.location or 'Not specified'}
Min Experience: {jd.min_experience_months or 0} months
Domain: {jd.domain or 'Not specified'}

Must-Have Skills: {', '.join(jd.must_have_skills)}
Nice-to-Have Skills: {', '.join(jd.nice_to_have_skills)}
Required Certifications: {', '.join(jd.required_certifications) if jd.required_certifications else 'None'}

Responsibilities:
{chr(10).join('- ' + r for r in jd.responsibilities[:5])}
"""

    system_prompt = """You are an expert recruiter scoring candidates against job requirements.

Score each component from 0.0 to 1.0:
- 0.0 = Completely missing/inadequate
- 0.5 = Partially meets requirements
- 1.0 = Fully meets or exceeds requirements

CRITICAL RULE — empty/unspecified requirements: if a requirement category
is EMPTY or not specified in the job description, that means NOTHING is
required in that category, so the candidate cannot be missing anything.
Score it accordingly:
  - Must-Have Skills is empty/none listed  -> must_have_skills_score = 1.0
  - Nice-to-Have Skills is empty/none listed -> good_to_have_skills_score = 1.0
  - Required Certifications is empty/none listed -> certifications_score = 1.0
  - Domain is "Not specified" -> domain_knowledge_score = 0.5 (neutral —
    there's nothing stated to judge alignment against, so this is neither
    a strength nor a weakness)
Do NOT score these low just because the candidate's resume doesn't happen
to mention something the JD never actually asked for — that would
incorrectly penalize a fully qualified candidate for a requirement that
doesn't exist. This exact mistake has caused real, wrongly-low scores in
testing — treat this rule as non-negotiable, not a suggestion.

Be objective and consistent. Consider:
- Skill coverage (exact matches and related technologies)
- Experience adequacy relative to requirements
- Project relevance to the role
- Domain alignment

Return scores in the structured format."""

    user_prompt = f"""Score this candidate against the job requirements.

RESUME:
{resume_summary}

JOB DESCRIPTION:
{jd_summary}

Provide scores for each component."""

    llm = get_llm()
    return llm.structured_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMScoringResponse,
        temperature=0.0,
    )


def score_candidate_llm(resume: ResumeData, jd: JDData) -> CandidateScores:
    logger.debug(f"LLM-scoring candidate='{resume.candidate_name}' against jd='{jd.job_title}'")

    llm_failed = False
    failure_reason = ""
    try:
        llm_response = _call_llm_scorer(resume, jd)
    except Exception as e:
        # FIX (gap #3): the old fallback silently returned 0.5 across every
        # component with generic reasoning "LLM scoring failed" — a 0.5
        # looks exactly like a real, mediocre score. A recruiter (or the
        # ranker sorting by overall_score) has no way to tell "this
        # candidate is average" apart from "scoring never actually ran for
        # this candidate." Now: score=0.0 (impossible to mistake for a real
        # middling result) + the actual exception message in reasoning, AND
        # a WARNING-level log (not just error) so it surfaces clearly.
        logger.warning(f"LLM scoring FAILED for {resume.candidate_name!r} — using 0.0 fallback, NOT 0.5. Error: {e}")
        llm_failed = True
        failure_reason = str(e)
        llm_response = LLMScoringResponse(
            must_have_skills_score=0.0,
            good_to_have_skills_score=0.0,
            domain_knowledge_score=0.0,
            certifications_score=0.0,
            total_experience_score=0.0,
            role_match_score=0.0,
            project_relevance_score=0.0,
            summary_score=0.0,
            skills_reasoning=f"LLM SCORING FAILED (not a real score): {failure_reason}",
            experience_reasoning=f"LLM SCORING FAILED (not a real score): {failure_reason}",
        )

    fail_note = " [LLM SCORING FAILED]" if llm_failed else ""

    section_scores = [
        SectionScore(
            section_name="must_have_skills", score=llm_response.must_have_skills_score, method="llm",
            raw_score=llm_response.must_have_skills_score * 10, reasoning=llm_response.skills_reasoning,
        ),
        SectionScore(
            section_name="good_to_have_skills", score=llm_response.good_to_have_skills_score, method="llm",
            reasoning=f"Nice-to-have skills coverage{fail_note}",
        ),
        SectionScore(
            section_name="domain_knowledge", score=llm_response.domain_knowledge_score, method="llm",
            reasoning=f"Domain alignment assessment{fail_note}",
        ),
        SectionScore(
            section_name="certifications", score=llm_response.certifications_score, method="llm",
            reasoning=f"Certification match{fail_note}",
        ),
        SectionScore(
            section_name="total_experience", score=llm_response.total_experience_score, method="llm",
            raw_score=llm_response.total_experience_score * 10, reasoning=llm_response.experience_reasoning,
        ),
        SectionScore(
            section_name="role_match", score=llm_response.role_match_score, method="llm",
            reasoning=f"Role/title similarity{fail_note}",
        ),
        SectionScore(
            section_name="project_relevance", score=llm_response.project_relevance_score, method="llm",
            reasoning=f"Project relevance to JD{fail_note}",
        ),
        SectionScore(
            section_name="semantic_boost", score=0.5, method="llm",
            reasoning="Computed by semantic engine",
        ),
        SectionScore(
            section_name="summary_score", score=llm_response.summary_score, method="llm",
            reasoning=f"Summary alignment{fail_note}",
        ),
    ]

    if llm_failed:
        logger.error(f"✗ LLM scoring FAILED for '{resume.candidate_name}' — all scores are 0.0 fallback, not real scores.")
    else:
        logger.info(
            f"✓ LLM-scored '{resume.candidate_name}': "
            f"must_have={llm_response.must_have_skills_score:.2f}, "
            f"experience={llm_response.total_experience_score:.2f}"
        )

    return CandidateScores(
        resume_id=resume.resume_id,
        candidate_name=resume.candidate_name,
        method="llm",
        section_scores=section_scores,
    )


def score_candidates_llm_bulk(resumes: list[ResumeData], jd: JDData) -> list[CandidateScores]:
    return [score_candidate_llm(resume, jd) for resume in resumes]