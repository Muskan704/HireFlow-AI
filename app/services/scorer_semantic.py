"""
Stage 4A — Semantic Scoring (README-aligned version).

Computes detailed component scores matching the README specification:

STAGE 2: SKILL MATCHING ENGINE
├── must_have_skills_score    = matched/total × 0.40
├── good_to_have_skills_score = matched/total × 0.10
├── domain_knowledge_score    = semantic match × 0.10
└── certifications_score      = matched/total × 0.05

STAGE 3: EXPERIENCE MATCHING ENGINE
├── total_exp_score   = f(actual, required) × 0.15
├── role_match_score  = title_similarity × 0.05
└── project_score     = relevant/total × 0.05

STAGE 4: SEMANTIC SIMILARITY ENGINE
└── semantic_similarity_boost = cosine sim for unmatched skills × 0.08

Uses local sentence-transformers model (all-MiniLM-L6-v2) by default.
"""

from __future__ import annotations

from loguru import logger

from app.services.embedder import embed_text, cosine_similarity
from app.models.schemas import ResumeData, JDData
from app.models.results import CandidateScores, SectionScore
from app.services.skill_aliases import normalize_skill_list


# ── Skill Matching Engine ────────────────────────────────────────────────────────

def _compute_must_have_skills_score(resume: ResumeData, jd: JDData) -> float:
    """
    must_have_skills_score = matched_skills / total_must_have × weight
    
    Exact match on normalized skills first.
    """
    if not jd.must_have_skills:
        return 1.0  # No requirements = full score
    
    resume_skills = set(resume.normalised_skills)
    jd_skills = set(jd.normalised_must_have_skills)
    
    matched = jd_skills.intersection(resume_skills)
    score = len(matched) / len(jd_skills) if jd_skills else 1.0
    
    return round(score, 4)


def _compute_good_to_have_skills_score(resume: ResumeData, jd: JDData) -> float:
    """
    good_to_have_skills_score = matched_skills / total_nice_to_have × weight
    """
    if not jd.nice_to_have_skills:
        return 1.0  # No nice-to-have = full score
    
    resume_skills = set(resume.normalised_skills)
    nice_to_have_normalized = normalize_skill_list(jd.nice_to_have_skills)
    
    matched = [s for s in nice_to_have_normalized if s in resume_skills]
    score = len(matched) / len(nice_to_have_normalized) if nice_to_have_normalized else 1.0
    
    return round(score, 4)


def _compute_domain_knowledge_score(resume: ResumeData, jd: JDData) -> float:
    """
    Domain knowledge score based on semantic similarity between JD domain
    and candidate's experience/skills.
    """
    if not jd.domain:
        return 0.5  # Neutral score if no domain specified
    
    # Combine resume skills and experience for domain matching
    resume_text = " ".join(resume.skills)
    for job in resume.work_experience:
        resume_text += " " + " ".join(job.technologies)
    
    if not resume_text.strip():
        return 0.0
    
    # Semantic similarity between domain and resume content
    domain_vec = embed_text(jd.domain)
    resume_vec = embed_text(resume_text)
    
    cos_sim = cosine_similarity(domain_vec, resume_vec)
    # Normalize from [-1, 1] to [0, 1]
    score = (cos_sim + 1) / 2
    
    return round(score, 4)


def _compute_certifications_score(resume: ResumeData, jd: JDData) -> float:
    """
    certifications_score = matched_certs / total_required_certs × weight
    
    Note: This is a SCORING function, not a filter.
    Hard filter already handles mandatory cert rejection.
    """
    if not jd.required_certifications:
        return 1.0  # No required certs = full score
    
    candidate_certs = {cert.name.lower().strip() for cert in resume.certifications}
    required_certs = [c.lower().strip() for c in jd.required_certifications]
    
    matched = [c for c in required_certs if c in candidate_certs]
    score = len(matched) / len(required_certs) if required_certs else 1.0
    
    return round(score, 4)


# ── Experience Matching Engine ────────────────────────────────────────────────────

def _compute_total_experience_score(resume: ResumeData, jd: JDData) -> float:
    """
    total_exp_score based on how well candidate's experience matches requirements.
    
    Scoring logic:
    - Below minimum: proportional penalty
    - At minimum: 0.7
    - Above minimum: up to 1.0 based on surplus
    """
    candidate_exp = resume.total_experience_months or 0
    required_exp = jd.min_experience_months or 0
    preferred_exp = jd.preferred_experience_months or required_exp
    
    if required_exp == 0:
        # No requirement - score based on having any experience
        if candidate_exp == 0:
            return 0.3
        elif candidate_exp < 12:
            return 0.5
        elif candidate_exp < 36:
            return 0.7
        else:
            return 0.9
    
    if candidate_exp < required_exp:
        # Below minimum - should be filtered, but score for completeness
        return round(candidate_exp / required_exp * 0.5, 4)
    elif candidate_exp < preferred_exp:
        # Between min and preferred
        ratio = (candidate_exp - required_exp) / (preferred_exp - required_exp) if preferred_exp > required_exp else 0
        return round(0.7 + 0.2 * ratio, 4)
    else:
        # Meets or exceeds preferred
        surplus_ratio = min((candidate_exp - preferred_exp) / preferred_exp, 0.1) if preferred_exp > 0 else 0
        return round(min(0.9 + surplus_ratio, 1.0), 4)


def _compute_role_match_score(resume: ResumeData, jd: JDData) -> float:
    """
    role_match_score = semantic similarity between JD title and candidate's titles
    """
    if not jd.job_title:
        return 0.5
    
    # Get all job titles from candidate
    candidate_titles = [job.title for job in resume.work_experience if job.title]
    if not candidate_titles:
        return 0.0
    
    # Find best matching title using semantic similarity
    jd_title_vec = embed_text(jd.job_title)
    
    best_score = 0.0
    for title in candidate_titles:
        title_vec = embed_text(title)
        cos_sim = cosine_similarity(jd_title_vec, title_vec)
        normalized = (cos_sim + 1) / 2
        best_score = max(best_score, normalized)
    
    return round(best_score, 4)


def _compute_project_relevance_score(resume: ResumeData, jd: JDData) -> float:
    """
    project_score = relevant_projects / total_projects
    
    Project is "relevant" if its technologies overlap with JD requirements.
    """
    if not resume.projects:
        # No Projects section is normal for many fields (accounting, finance,
        # ops, etc.) — it's a formatting convention, not a sign of weaker
        # candidates. Scoring this LOW (0.3) systematically penalizes every
        # non-engineering candidate by 5% of their total score for something
        # that isn't actually a gap. Neutral (0.5) = "no signal either way,"
        # not "bad signal" — confirmed via real testing on an accountant
        # candidate whose otherwise-strong profile was being dragged down.
        return 0.5
    
    jd_skills = set(jd.normalised_must_have_skills)
    jd_skills.update(s.lower().strip() for s in jd.nice_to_have_skills)
    
    if not jd_skills:
        return 0.5  # No JD skills to match against
    
    relevant_count = 0
    for project in resume.projects:
        project_techs = {t.lower().strip() for t in project.technologies}
        if project_techs.intersection(jd_skills):
            relevant_count += 1
    
    score = relevant_count / len(resume.projects)
    return round(score, 4)


# ── Semantic Similarity Engine ────────────────────────────────────────────────────

def _compute_semantic_boost_score(resume: ResumeData, jd: JDData) -> float:
    """
    Semantic similarity boost for skills NOT matched by exact/alias lookup.
    Only unmatched must-have skills get this treatment.
    """
    resume_skills = set(resume.normalised_skills)
    jd_skills = set(jd.normalised_must_have_skills)
    
    # Find unmatched JD skills
    unmatched_skills = jd_skills - resume_skills
    
    if not unmatched_skills:
        return 1.0  # All matched = full boost
    
    if not resume.skills:
        return 0.0
    
    # Get embeddings for unmatched skills and resume text
    resume_text = " ".join(resume.skills)
    resume_vec = embed_text(resume_text)
    
    matched_scores = []
    for skill in unmatched_skills:
        skill_vec = embed_text(skill)
        cos_sim = cosine_similarity(skill_vec, resume_vec)
        # Threshold: > 0.72 counts as soft match
        if cos_sim > 0.72:
            matched_scores.append((cos_sim + 1) / 2)
    
    if not matched_scores:
        return 0.0
    
    # Average soft match score
    return round(sum(matched_scores) / len(unmatched_skills), 4)


def _compute_summary_score(resume: ResumeData, jd: JDData) -> float:
    """
    summary_score = semantic match between candidate summary and JD.
    Very low weight (0.02).
    """
    if not resume.summary:
        return 0.3
    
    # Create JD summary text
    jd_text = f"{jd.job_title} {' '.join(jd.responsibilities)} {jd.domain or ''}"
    
    summary_vec = embed_text(resume.summary)
    jd_vec = embed_text(jd_text)
    
    cos_sim = cosine_similarity(summary_vec, jd_vec)
    score = (cos_sim + 1) / 2
    
    return round(score, 4)


# ── Main Scoring Function ────────────────────────────────────────────────────────

def score_candidate_semantic(resume: ResumeData, jd: JDData) -> CandidateScores:
    """
    Compute all component scores for a candidate using semantic matching.
    
    Returns CandidateScores with detailed section scores matching README spec.
    """
    section_scores = [
        # Skill Matching Engine
        SectionScore(
            section_name="must_have_skills",
            score=_compute_must_have_skills_score(resume, jd),
            method="semantic",
            reasoning="Exact match on normalized must-have skills"
        ),
        SectionScore(
            section_name="good_to_have_skills",
            score=_compute_good_to_have_skills_score(resume, jd),
            method="semantic",
            reasoning="Match on nice-to-have skills"
        ),
        SectionScore(
            section_name="domain_knowledge",
            score=_compute_domain_knowledge_score(resume, jd),
            method="semantic",
            reasoning="Semantic similarity between JD domain and candidate experience"
        ),
        SectionScore(
            section_name="certifications",
            score=_compute_certifications_score(resume, jd),
            method="semantic",
            reasoning="Match on required certifications"
        ),
        
        # Experience Matching Engine
        SectionScore(
            section_name="total_experience",
            score=_compute_total_experience_score(resume, jd),
            method="semantic",
            reasoning="Total experience vs requirements"
        ),
        SectionScore(
            section_name="role_match",
            score=_compute_role_match_score(resume, jd),
            method="semantic",
            reasoning="Semantic similarity of job titles"
        ),
        SectionScore(
            section_name="project_relevance",
            score=_compute_project_relevance_score(resume, jd),
            method="semantic",
            reasoning="Projects with technologies matching JD"
        ),
        
        # Semantic Similarity Engine
        SectionScore(
            section_name="semantic_boost",
            score=_compute_semantic_boost_score(resume, jd),
            method="semantic",
            reasoning="Soft matches for unmatched skills (threshold > 0.72)"
        ),
        SectionScore(
            section_name="summary_score",
            score=_compute_summary_score(resume, jd),
            method="semantic",
            reasoning="Resume summary alignment with JD"
        ),
    ]
    
    logger.debug(
        f"Semantic scores for {resume.candidate_name}: "
        f"must_have={section_scores[0].score:.2f}, "
        f"good_to_have={section_scores[1].score:.2f}, "
        f"domain={section_scores[2].score:.2f}, "
        f"experience={section_scores[4].score:.2f}"
    )
    
    return CandidateScores(
        resume_id=resume.resume_id,
        candidate_name=resume.candidate_name,
        method="semantic",
        section_scores=section_scores,
    )


def score_candidates_semantic_bulk(
    resumes: list[ResumeData], jd: JDData
) -> list[CandidateScores]:
    """Bulk helper — processes all resumes."""
    return [score_candidate_semantic(resume, jd) for resume in resumes]