"""
app/models/results.py
----------------------
Output schemas for Stage 3 through Stage 6.

Why this file exists separately from schemas.py:
  schemas.py defines INPUT data (ResumeData, JDData) — what the LLM extracts.
  results.py defines OUTPUT data — what each pipeline stage produces.

  Keeping them separate makes it obvious which models flow INTO the pipeline
  vs which ones flow OUT of each stage. Every stage from here on reads one
  of these models and returns another.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal


# ── Stage 3: Hard Filter output ─────────────────────────────────────────────────

class FilterResult(BaseModel):
    """
    Output of the hard filter (Stage 3) for ONE candidate.

    This is a knockout gate — either the candidate passes and moves to
    Stage 4, or they're rejected here and the pipeline stops for them.
    No scoring happens until this returns passed=True.
    """
    resume_id: Optional[str] = Field(
        default=None,
        description="Links back to ResumeData.resume_id — use this for persistence, not candidate_name.",
    )
    candidate_name: Optional[str] = None
    passed: bool

    # Empty list if passed=True. Each string is a human-readable reason,
    # e.g. "Missing required skill: aws"
    reject_reasons: list[str] = Field(default_factory=list)

    # Populated only for rejected candidates, by summariser.py's
    # summarise_rejection() — a short grounded explanation of why they
    # were rejected, plus whether it's a close miss worth a second look
    # (1-2 gaps) vs a broad mismatch. None for passed candidates, and None
    # for parsing failures (no resume to summarise against).
    rejection_summary: Optional[str] = Field(
        default=None,
        description="Short explanation of why this candidate was rejected, for recruiter review.",
    )
    is_close_miss: Optional[bool] = Field(
        default=None,
        description="True if only 1-2 requirements were missing — worth a manual second look.",
    )

    # Which specific checks were run and their individual pass/fail —
    # useful for debugging WHY a candidate was rejected, not just THAT
    # they were rejected.
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "e.g. {'min_experience': True, 'must_have_skills': False, "
            "'required_education': True}"
        ),
    )


# ── Stage 4: Similarity Scoring output ──────────────────────────────────────────

class SectionScore(BaseModel):
    """
    Score for ONE section (e.g. 'skills', 'experience', 'other')
    for ONE candidate, from ONE scoring path (semantic or LLM).

    Both scorer_semantic.py and scorer_llm.py return this same shape —
    that's what lets the ranker treat them interchangeably.
    """
    section_name: str                    # "skills" | "experience" | "other"
    score: float = Field(ge=0.0, le=1.0)  # always normalised to 0-1, even if LLM gave 0-10
    raw_score: Optional[float] = None     # original scale before normalising, for debugging
    method: Literal["semantic", "llm"]    # which path produced this
    reasoning: Optional[str] = Field(
        default=None,
        description="LLM's justification string. None for semantic scores — cosine similarity has no 'reasoning'.",
    )


class CandidateScores(BaseModel):
    """
    All section scores for ONE candidate, from ONE scoring path.

    We'll have up to two of these per candidate during the Stage 4
    experiment — one from scorer_semantic.py, one from scorer_llm.py.
    """
    resume_id: Optional[str] = None
    candidate_name: Optional[str] = None
    method: Literal["semantic", "llm"]
    section_scores: list[SectionScore]

    def get_section(self, name: str) -> Optional[SectionScore]:
        """Helper: find a section score by name without looping manually elsewhere."""
        for s in self.section_scores:
            if s.section_name == name:
                return s
        return None


# ── Stage 5: Weighted Ranking output ────────────────────────────────────────────

class RankedCandidate(BaseModel):
    """
    Final ranked output for ONE candidate — Stage 5's product.

    This is what the API will eventually return for each candidate
    in the ranked list. overall_score is the single sortable number;
    section_scores is the breakdown so the result is explainable,
    not a black box.
    """
    rank: int
    resume_id: Optional[str] = None
    candidate_name: Optional[str] = None
    overall_score: float = Field(ge=0.0, le=1.0)

    # Breakdown by section so a recruiter can see WHY this score happened
    section_scores: dict[str, float] = Field(
        default_factory=dict,
        description="e.g. {'skills': 0.85, 'experience': 0.70, 'other': 0.60}",
    )

    # Which weight config produced this ranking — lets you compare
    # different weights.json runs side by side later
    weights_used: dict[str, float] = Field(default_factory=dict)

    # Filled in by Stage 6 — None until the summary stage runs
    fit_summary: Optional[str] = None


# ── Stage 6: Per-candidate Summary output ───────────────────────────────────────

class CandidateSummary(BaseModel):
    """
    Output of the LLM summary call (Stage 6) for ONE ranked candidate.

    Kept separate from RankedCandidate so Stage 5 and Stage 6 stay
    decoupled — ranker.py never needs to know about LLM summaries,
    and summariser.py never needs to know about scoring weights.
    """
    resume_id: Optional[str] = None
    candidate_name: Optional[str] = None
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendation: str = Field(
        description="One paragraph: overall fit assessment and recommendation."
    )
    agency_notes: str = Field(
        default="",
        description=(
            "Detailed, submission-ready explanation of fit — 5-8 sentences, each "
            "explicitly matching a SPECIFIC JD requirement to SPECIFIC resume "
            "evidence (e.g. 'JD requires X; candidate demonstrated this by doing Y "
            "at Company Z'). Written in professional agency-notes style, reusable "
            "verbatim by another system/microservice as submission notes — not "
            "just a summary for internal scoring review."
        ),
    )


# ── Stage 7: Knowledge Brief output ──────────────────────────────────────────────

class KnowledgeBrief(BaseModel):
    """
    Comprehensive knowledge brief for a top-ranked candidate.
    
    Generated for recruiter interview prep, not candidate evaluation.
    Combines JD context, resume digest, and interview guidance.
    """
    resume_id: Optional[str] = None
    candidate_name: Optional[str] = None
    
    # JD Context
    role_overview: str = Field(
        default="",
        description="Brief overview of the role and its key requirements"
    )
    team_context: Optional[str] = Field(
        default=None,
        description="Team/company context if available from JD"
    )
    
    # Candidate Snapshot
    career_summary: str = Field(
        default="",
        description="2-3 sentence summary of candidate's career trajectory"
    )
    key_achievements: list[str] = Field(
        default_factory=list,
        description="Top 3-5 notable achievements from resume"
    )
    technical_strengths: list[str] = Field(
        default_factory=list,
        description="Core technical competencies"
    )
    
    # Fit Analysis
    alignment_highlights: list[str] = Field(
        default_factory=list,
        description="Specific ways candidate aligns with JD requirements"
    )
    experience_relevance: str = Field(
        default="",
        description="How candidate's experience maps to role requirements"
    )
    
    # Interview Prep
    areas_to_probe: list[str] = Field(
        default_factory=list,
        description="Areas to explore deeper in interview (gaps, questions)"
    )
    suggested_talking_points: list[str] = Field(
        default_factory=list,
        description="Recommended topics to discuss with candidate"
    )
    
    # Quick Reference
    years_of_experience: Optional[int] = None
    current_or_last_role: Optional[str] = None
    education_highlight: Optional[str] = None
    certifications: list[str] = Field(default_factory=list)
    location: Optional[str] = None


class ResumeFullDetail(BaseModel):
    """
    Everything about ONE resume, for the GET /resumes/{resume_id} endpoint.
    This is the single response shape a frontend needs to render a full
    candidate detail view — no follow-up calls required.
    """
    resume_id: str
    session_id: str
    candidate_name: Optional[str] = None
    company: Optional[str] = None
    job_title: Optional[str] = None

    # Stage 3
    passed_filter: bool
    reject_reasons: list[str] = Field(default_factory=list)
    rejection_summary: Optional[str] = None
    is_close_miss: Optional[bool] = None

    # Stage 4 — both raw scorer outputs, kept separate for transparency
    semantic_scores: dict[str, float] = Field(default_factory=dict)
    llm_scores: dict[str, float] = Field(default_factory=dict)

    # Stage 5 — final blended result
    rank: Optional[int] = None
    overall_score: Optional[float] = None
    final_section_scores: dict[str, float] = Field(default_factory=dict)
    weights_used: dict = Field(default_factory=dict)  # raw weights.json structure, not flattened

    # Stage 6
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendation: Optional[str] = None
    agency_notes: Optional[str] = Field(
        default=None,
        description="Detailed, submission-ready JD-matching notes — see CandidateSummary.agency_notes.",
    )

    # Stage 7
    knowledge_brief: Optional[KnowledgeBrief] = None

    # Raw material, for a recruiter who wants to see the source
    resume_structured_data: dict = Field(default_factory=dict)
    resume_raw_text: Optional[str] = None

    created_at: Optional[str] = None


# ── Final pipeline output ───────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """
    The complete output of one end-to-end pipeline run.
    This is what gets serialised to JSON and returned by the API.
    """
    jd_title: Optional[str] = None
    total_resumes_processed: int
    total_passed_filter: int
    total_rejected: int

    rejected_candidates: list[FilterResult] = Field(default_factory=list)
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    
    # Stage 7: Knowledge briefs for top N candidates
    knowledge_briefs: list[KnowledgeBrief] = Field(
        default_factory=list,
        description="Detailed briefs for top-ranked candidates (interview prep)"
    )

    similarity_mode: str = Field(
        description="'semantic' | 'llm' | 'both' — which Stage 4 path(s) were used"
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Set when persistence succeeded — fetch this session again via /sessions/{session_id}/candidates.",
    )