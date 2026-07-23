"""
Business-facing orchestration for candidate ranking.

This layer keeps FastAPI request handling separate from the recruitment
pipeline internals. The pipeline remains the source of truth for all parsing,
extraction, filtering, scoring, ranking, summarisation, and brief generation.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import UploadFile

from app.core.config import get_settings
from app.models.results import PipelineResult, ResumeFullDetail
from app.services.pipeline import run_pipeline_with_persistence_async
from app.services.session_repo import get_pipeline_result, get_resume_detail


class CandidateRankingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def process(
        self,
        jd: Optional[UploadFile],
        jd_text: Optional[str],
        resumes: list[UploadFile],
    ) -> PipelineResult:
        if not resumes:
            raise ValueError("At least one resume file is required.")

        jd_source = await self._build_jd_source(jd=jd, jd_text=jd_text)
        resume_sources = await asyncio.gather(*[
            self._read_upload(resume, "Resume")
            for resume in resumes
        ])

        result, session_id = await run_pipeline_with_persistence_async(
            resume_sources=resume_sources,
            jd_source=jd_source,
        )
        result.session_id = session_id or None
        return result

    async def _build_jd_source(
        self,
        jd: Optional[UploadFile],
        jd_text: Optional[str],
    ) -> tuple[bytes, str]:
        has_jd_file = jd is not None and bool(jd.filename)
        has_jd_text = bool(jd_text and jd_text.strip())

        if has_jd_file and has_jd_text:
            raise ValueError("Provide either jd file or jd_text, not both.")

        if has_jd_text:
            return jd_text.strip().encode("utf-8"), "job_description.txt"

        if has_jd_file and jd is not None:
            return await self._read_upload(jd, "Job description")

        raise ValueError("Either jd file or jd_text is required.")

    def get_result(self, session_id: str) -> PipelineResult:
        return get_pipeline_result(session_id)

    def get_candidate_detail(self, session_id: str, resume_id: str) -> ResumeFullDetail:
        detail = get_resume_detail(resume_id)
        if detail.session_id != session_id:
            raise ValueError(
                f"Resume id={resume_id} does not belong to session id={session_id}"
            )
        return detail

    async def _read_upload(self, file: UploadFile, label: str) -> tuple[bytes, str]:
        filename = file.filename or ""
        self._validate_filename(filename, label)

        raw = await file.read()
        max_bytes = self.settings.max_resume_size_mb * 1024 * 1024
        if len(raw) > max_bytes:
            raise ValueError(
                f"{filename!r} exceeds {self.settings.max_resume_size_mb}MB limit."
            )
        return raw, filename

    @staticmethod
    def _validate_filename(filename: str, label: str) -> None:
        allowed_extensions = (".pdf", ".docx")
        if not filename.lower().endswith(allowed_extensions):
            raise ValueError(f"{label} must be a PDF or DOCX file.")
