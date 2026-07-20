"""
Stage 7 — Knowledge Brief Engine.

Generates a comprehensive knowledge brief for top-ranked candidates
that recruiters can use to prepare for interviews.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel
from typing import Optional

from app.models.schemas import ResumeData, JDData
from app.models.results import RankedCandidate, KnowledgeBrief
from app.llm.factory import get_llm


class LLMBriefResponse(BaseModel):
    role_overview: str
    team_context: Optional[str] = None
    career_summary: str
    key_achievements: list[str]
    technical_strengths: list[str]
    alignment_highlights: list[str]
    experience_relevance: str
    areas_to_probe: list[str]
    suggested_talking_points: list[str]


def _call_llm_brief(resume: ResumeData, jd: JDData, ranked: RankedCandidate) -> LLMBriefResponse:
    resume_context = f"""
CANDIDATE: {resume.candidate_name}
TOTAL EXPERIENCE: {resume.total_experience_months or 0} months ({(resume.total_experience_months or 0) // 12} years)

SKILLS:
{', '.join(resume.skills[:30])}

PROGRAMMING LANGUAGES:
{', '.join(resume.programming_languages)}

FRAMEWORKS & TOOLS:
{', '.join(resume.frameworks_and_tools)}

WORK EXPERIENCE:
"""
    for job in resume.work_experience[:5]:
        resume_context += f"\n• {job.title} at {job.company} ({job.duration_months or 0} months)"
        if job.technologies:
            resume_context += f"\n  Technologies: {', '.join(job.technologies)}"
        if job.responsibilities:
            resume_context += f"\n  Key work: {'; '.join(job.responsibilities[:3])}"

    if resume.projects:
        resume_context += "\n\nNOTABLE PROJECTS:"
        for proj in resume.projects[:3]:
            resume_context += f"\n• {proj.name}: {proj.description}"
            if proj.impact:
                resume_context += f" | Impact: {proj.impact}"

    if resume.certifications:
        resume_context += f"\n\nCERTIFICATIONS: {', '.join(c.name for c in resume.certifications)}"

    if resume.education:
        resume_context += "\n\nEDUCATION:"
        for edu in resume.education[:2]:
            resume_context += f"\n• {edu.degree} in {edu.field_of_study} from {edu.institution}"

    jd_context = f"""
ROLE: {jd.job_title}
COMPANY: {jd.company or 'Not specified'}
LOCATION: {jd.location or 'Not specified'}
MINIMUM EXPERIENCE: {jd.min_experience_months or 0} months

MUST-HAVE SKILLS:
{', '.join(jd.must_have_skills)}

NICE-TO-HAVE SKILLS:
{', '.join(jd.nice_to_have_skills) if jd.nice_to_have_skills else 'None specified'}

DOMAIN: {jd.domain or 'Not specified'}

KEY RESPONSIBILITIES:
{chr(10).join('• ' + r for r in jd.responsibilities[:5])}

REQUIRED CERTIFICATIONS:
{', '.join(jd.required_certifications) if jd.required_certifications else 'None'}
"""

    candidate_scores = f"""
CANDIDATE FIT SCORES:
Overall Score: {ranked.overall_score:.2%}
"""
    for section, score in ranked.section_scores.items():
        candidate_scores += f"\n• {section.replace('_', ' ').title()}: {score:.2%}"

    system_prompt = """You are an expert recruitment analyst creating a knowledge brief for a recruiter.

The recruiter will use this brief to prepare for an interview with the candidate.
Be specific, factual, and avoid generic statements. Quote specific achievements or skills when possible.

The brief should:
1. Summarize the role context
2. Provide a clear picture of the candidate's background
3. Highlight specific alignment with the role
4. Suggest areas to probe deeper during the interview
5. Provide talking points for meaningful discussion

Be concise but informative. The recruiter has limited prep time."""

    user_prompt = f"""Create a knowledge brief for this candidate-role pair.

JOB DESCRIPTION:
{jd_context}

CANDIDATE RESUME:
{resume_context}

FIT ANALYSIS:
{candidate_scores}

Generate a comprehensive knowledge brief for recruiter interview prep."""

    llm = get_llm()
    return llm.structured_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_model=LLMBriefResponse,
        temperature=0.0,
    )


def generate_knowledge_brief(resume: ResumeData, jd: JDData, ranked: RankedCandidate) -> KnowledgeBrief:
    logger.debug(f"Generating knowledge brief for candidate='{resume.candidate_name}' role='{jd.job_title}'")

    try:
        llm_response = _call_llm_brief(resume, jd, ranked)
    except Exception as e:
        logger.error(f"Knowledge brief generation failed for {resume.candidate_name}: {e}")
        llm_response = LLMBriefResponse(
            role_overview=f"Role: {jd.job_title} at {jd.company or 'Company'}",
            team_context=None,
            career_summary=f"{resume.candidate_name} has {resume.total_experience_months or 0} months of experience.",
            key_achievements=["Brief generation failed - please review resume directly"],
            technical_strengths=resume.skills[:5] if resume.skills else [],
            alignment_highlights=[],
            experience_relevance="Could not generate - please review manually",
            areas_to_probe=["Review resume manually due to generation error"],
            suggested_talking_points=["Discuss experience and skills directly"],
        )

    brief = KnowledgeBrief(
        resume_id=resume.resume_id,
        candidate_name=resume.candidate_name,
        role_overview=llm_response.role_overview,
        team_context=llm_response.team_context,
        career_summary=llm_response.career_summary,
        key_achievements=llm_response.key_achievements,
        technical_strengths=llm_response.technical_strengths,
        alignment_highlights=llm_response.alignment_highlights,
        experience_relevance=llm_response.experience_relevance,
        areas_to_probe=llm_response.areas_to_probe,
        suggested_talking_points=llm_response.suggested_talking_points,
        years_of_experience=(resume.total_experience_months or 0) // 12,
        current_or_last_role=resume.work_experience[0].title if resume.work_experience else None,
        education_highlight=f"{resume.education[0].degree} in {resume.education[0].field_of_study}" if resume.education else None,
        certifications=[c.name for c in resume.certifications],
        location=resume.location,
    )

    logger.info(
        f"✓ Knowledge brief generated for '{resume.candidate_name}': "
        f"{len(brief.key_achievements)} achievements, {len(brief.areas_to_probe)} probe areas"
    )
    return brief


def generate_briefs_bulk(
    resumes: list[ResumeData], jd: JDData, ranked_candidates: list[RankedCandidate], top_n: int = 3,
) -> list[KnowledgeBrief]:
    resume_by_id = {r.resume_id: r for r in resumes}
    briefs = []
    for rank in ranked_candidates[:top_n]:
        resume = resume_by_id.get(rank.resume_id)
        if not resume:
            logger.warning(f"Resume not found for ranked candidate {rank.resume_id}")
            continue
        briefs.append(generate_knowledge_brief(resume, jd, rank))
    logger.info(f"Generated {len(briefs)} knowledge briefs")
    return briefs