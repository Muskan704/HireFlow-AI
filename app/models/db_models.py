"""
Persistence-layer ORM models.

Uses SQLModel (built on Pydantic v2 + SQLAlchemy) so these models share the
same validation primitives as your existing schemas.py / results.py, and map
cleanly to/from ResumeData, JDData, FilterResult, CandidateScores,
RankedCandidate, CandidateSummary without a separate translation layer.

Every table carries agency_id (nullable for now, non-null once auth exists)
and session_id — this is the fix for the "1 JD -> N resumes, no tenant
scoping" gap. Adding it now costs nothing; retrofitting it after real data
exists means a migration touching every table.

Run `python -m app.core.database` (see database.py) to create tables in a
fresh Postgres instance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import SQLModel, Field, JSON, Column


def _uuid_str() -> str:
    return str(uuid.uuid4())


class Session(SQLModel, table=True):
    """One Pre-Call Setup run: one JD + N resumes."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    agency_id: Optional[str] = Field(default=None, index=True)
    status: str = Field(default="processing")  # processing | complete | failed
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    weights_used: dict = Field(default_factory=dict, sa_column=Column(JSON))


class JobDescriptionRecord(SQLModel, table=True):
    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    agency_id: Optional[str] = Field(default=None, index=True)
    raw_text: str
    structured_data: dict = Field(sa_column=Column(JSON))  # JDData.model_dump()
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ResumeRecord(SQLModel, table=True):
    # NOTE: no default_factory here on purpose. This id must be set to
    # ResumeData.resume_id when saving — it's the SAME id, not a second one.
    # Generating a separate UUID here would defeat the point of adding
    # resume_id to ResumeData in the first place (one stable id per resume,
    # used everywhere from extraction through to the DB row).
    id: str = Field(primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    agency_id: Optional[str] = Field(default=None, index=True)
    candidate_name: Optional[str] = Field(default=None, index=True)
    raw_text: str
    structured_data: dict = Field(sa_column=Column(JSON))  # ResumeData.model_dump()
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FilterResultRecord(SQLModel, table=True):
    """Stage 3 output — persisted so 'why was this candidate rejected' is answerable later."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    resume_id: str = Field(index=True, foreign_key="resumerecord.id")
    passed: bool = Field(index=True)
    reject_reasons: list = Field(default_factory=list, sa_column=Column(JSON))
    checks: dict = Field(default_factory=dict, sa_column=Column(JSON))
    rejection_summary: Optional[str] = Field(default=None)
    is_close_miss: Optional[bool] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateScoreRecord(SQLModel, table=True):
    """Stage 4 output — kept per scoring method so the semantic-vs-LLM comparison stays queryable."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    resume_id: str = Field(index=True, foreign_key="resumerecord.id")
    method: str = Field(index=True)  # "semantic" | "llm"
    section_scores: dict = Field(sa_column=Column(JSON))  # {"skills": 8.5, "experience": 7.0, ...}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RankedCandidateRecord(SQLModel, table=True):
    """Stage 5 output."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    resume_id: str = Field(index=True, foreign_key="resumerecord.id")
    rank: int = Field(index=True)
    overall_score: float
    score_breakdown: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateSummaryRecord(SQLModel, table=True):
    """Stage 6 output."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    resume_id: str = Field(index=True, foreign_key="resumerecord.id")
    strengths: list = Field(default_factory=list, sa_column=Column(JSON))
    gaps: list = Field(default_factory=list, sa_column=Column(JSON))
    recommendation: str
    grounded_on: dict = Field(default_factory=dict, sa_column=Column(JSON))  # trace back to source fields
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WeightsConfigRecord(SQLModel, table=True):
    """Versioned weights.json snapshots — so you can tell which weights produced which ranking."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    label: str = Field(index=True)  # e.g. "default", "experiment-2"
    weights: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBriefRecord(SQLModel, table=True):
    """Stage 7 output — interview-prep briefs for top-ranked candidates."""

    id: str = Field(default_factory=_uuid_str, primary_key=True)
    session_id: str = Field(index=True, foreign_key="session.id")
    resume_id: str = Field(index=True, foreign_key="resumerecord.id")
    brief_data: dict = Field(sa_column=Column(JSON))  # KnowledgeBrief.model_dump()
    created_at: datetime = Field(default_factory=datetime.utcnow)