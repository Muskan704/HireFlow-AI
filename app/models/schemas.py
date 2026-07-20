"""
Pydantic v2 schemas — single source of truth for the entire pipeline.

Every stage reads from and writes to these models.

Key design decisions:
  - All fields Optional where the LLM might not find them (prevents crashes)
  - normalised_skills is a computed field — derived automatically from skills,
    so the hard filter always has clean lowercase data to work with
  - raw_text_snippet is excluded from serialisation (debug only)
"""
from __future__ import annotations
import uuid
from pydantic import BaseModel, Field, computed_field
from typing import Optional

from app.services.skill_aliases import normalize_skill_list


# ── Resume sub-models ──────────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    company: str
    title: str
    duration_months: Optional[int] = Field(
        default=None,
        description="Total duration in months. Compute from dates if given.",
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)
    impact: Optional[str] = None


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None


# ── ResumeData ─────────────────────────────────────────────────────────────────

class ResumeData(BaseModel):
    """
    Structured output from a single resume.
    Produced by Stage 2 (LLM extraction).
    Consumed by Stage 3 (hard filter) and Stage 4 (similarity scoring).
    """
    resume_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Stable unique ID for this resume, generated at extraction time. "
            "candidate_name alone is NOT a safe key — two candidates can share "
            "a name. Every downstream stage (filter, scores, ranking, summary) "
            "should carry this ID forward for reliable joins/persistence."
        ),
    )
    candidate_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None

    total_experience_months: Optional[int] = Field(
        default=None,
        description=(
            "Sum of ALL work experience durations in months. "
            "Compute this — do not leave None if dates are present."
        ),
    )

    skills: list[str] = Field(
        default_factory=list,
        description="All technical and soft skills found anywhere in the resume.",
    )
    programming_languages: list[str] = Field(default_factory=list)
    frameworks_and_tools: list[str] = Field(default_factory=list)

    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)

    summary: Optional[str] = Field(
        default=None,
        description="Candidate's own summary/objective section, verbatim.",
    )

    @computed_field
    @property
    def normalised_skills(self) -> list[str]:
        """
        Lowercase, stripped version of all skills.

        Why this exists:
          The LLM returns "Python" on one resume and "python" on another.
          The hard filter uses this list so it always compares
          lowercase → lowercase. No false rejections from case differences.

        Example:
          skills            = ["Python", "FastAPI", "DOCKER"]
          normalised_skills = ["python", "fastapi", "docker"]

        Also applies the deterministic skill-alias table (skill_aliases.py)
        as a backstop underneath the LLM's own normalization instructions.
        The LLM prompt already asks it to expand "k8s" -> "Kubernetes" etc.,
        but that's a best-effort instruction, not a guarantee — this makes
        the common cases deterministic regardless of what the LLM actually
        returned.
        """
        return normalize_skill_list(self.skills)

    raw_text_snippet: Optional[str] = Field(
        default=None,
        description="First 300 chars of raw text — debugging only.",
        exclude=True,
    )


# ── JDData ─────────────────────────────────────────────────────────────────────

class JDData(BaseModel):
    """
    Structured output from a job description.
    Produced by Stage 2 (LLM extraction).
    Consumed by Stage 3 (hard filter) and Stage 4 (similarity scoring).
    """
    job_title: str
    company: Optional[str] = None
    location: Optional[str] = None
    remote_policy: Optional[str] = None

    # ── Hard filter fields (Stage 3 reads these) ───────────────────────────────
    min_experience_months: Optional[int] = Field(
        default=None,
        description=(
            "Minimum total work experience required, in months. "
            "Convert '3 years' → 36. Use the MINIMUM of any range."
        ),
    )

    must_have_skills: list[str] = Field(
        default_factory=list,
        description="Non-negotiable skills. Absence = automatic reject.",
    )

    must_have_skill_groups: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Groups of alternative skills where AT LEAST ONE from each group is "
            "required. Use this for 'X, Y, or Z (required)' style requirements — "
            "e.g. a JD requiring 'GCP, AWS, or Azure' should produce "
            "must_have_skill_groups=[['GCP','AWS','Azure']], NOT three separate "
            "entries in must_have_skills."
        ),
    )

    required_certifications: list[str] = Field(
        default_factory=list,
        description=(
            "Certifications that are mandatory. e.g. ['AWS Solutions Architect']. "
            "Absence = automatic reject. Leave empty if none required."
        ),
    )

    location: Optional[str] = Field(
        default=None,
        description="Job location e.g. 'Bengaluru', 'Remote', 'New York'.",
    )

    location_strict: bool = Field(
        default=False,
        description=(
            "If True, candidate location must match JD location — reject if mismatch. "
            "If False, location is noted but never causes rejection. "
            "Default False because most JDs in our test set are remote-friendly."
        ),
    )
    
    required_education: Optional[str] = Field(
        default=None,
        description="Minimum degree level e.g. 'B.Tech', 'Master's'.",
    )

    @computed_field
    @property
    def normalised_must_have_skills(self) -> list[str]:
        """
        Lowercase must-have skills for hard filter comparison.
        Mirrors normalised_skills on ResumeData — both sides normalized
        through the same alias table, so "K8s" on one side and "Kubernetes"
        on the other resolve to the same canonical string.
        """
        return normalize_skill_list(self.must_have_skills)

    # ── Scoring fields (Stage 4+5 reads these) ────────────────────────────────
    nice_to_have_skills: list[str] = Field(default_factory=list)
    preferred_experience_months: Optional[int] = Field(
        default=None,
        description="Preferred (not required) experience in months.",
    )
    responsibilities: list[str] = Field(default_factory=list)
    domain: Optional[str] = Field(
        default=None,
        description="Primary domain e.g. 'machine learning', 'backend'.",
    )

    raw_text_snippet: Optional[str] = Field(
        default=None,
        exclude=True,
    )