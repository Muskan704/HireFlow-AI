"""
FastAPI app for the Recruitment Intelligence Platform.

Public API surface:
  - health check
  - candidate ranking business endpoints
  - deprecated pipeline endpoint kept for backward compatibility
"""

import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.models.results import PipelineResult, ResumeFullDetail
from app.services.candidate_ranking import CandidateRankingService
from app.services.pipeline import run_pipeline_with_persistence


settings = get_settings()

logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

app = FastAPI(
    title="Recruitment Intelligence Platform",
    description="AI-powered resume screening and ranking - Pre-Call Module",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

candidate_ranking_service = CandidateRankingService()


@app.get("/health")
def health():
    return {"status": "ok", "active_llm": settings.active_llm}


@app.post(
    "/candidate-ranking",
    response_model=PipelineResult,
    tags=["Candidate Ranking"],
)
async def create_candidate_ranking(
    jd: UploadFile = File(..., description="One job description file (PDF/DOCX)"),
    resumes: list[UploadFile] = File(..., description="Multiple resume files (PDF/DOCX)"),
):
    """
    Primary business endpoint for recruiters and consuming microservices.
    """
    try:
        return await candidate_ranking_service.process(jd=jd, resumes=resumes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Candidate ranking failed")
        raise HTTPException(500, f"Candidate ranking failed: {e}")


@app.get(
    "/candidate-ranking/{session_id}",
    response_model=PipelineResult,
    tags=["Candidate Ranking"],
)
async def get_candidate_ranking(session_id: str):
    """
    Retrieve a previously generated ranking result by session_id.
    """
    try:
        return candidate_ranking_service.get_result(session_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("Candidate ranking retrieval failed")
        raise HTTPException(500, f"Candidate ranking retrieval failed: {e}")


@app.get(
    "/candidate-ranking/{session_id}/candidate/{resume_id}",
    response_model=ResumeFullDetail,
    tags=["Candidate Ranking"],
)
async def get_candidate_ranking_detail(session_id: str, resume_id: str):
    """
    Detailed candidate view for a candidate already processed in this session.
    """
    try:
        return candidate_ranking_service.get_candidate_detail(session_id, resume_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception("Candidate detail retrieval failed")
        raise HTTPException(500, f"Candidate detail retrieval failed: {e}")


@app.post(
    "/pipeline/run",
    response_model=PipelineResult,
    tags=["Pipeline"],
    deprecated=True,
)
async def run_pipeline_endpoint(
    resumes: list[UploadFile] = File(..., description="Multiple resume files (PDF/DOCX)"),
    jd: UploadFile = File(..., description="One job description file (PDF/DOCX)"),
):
    """
    Deprecated compatibility endpoint. Prefer POST /candidate-ranking.
    """
    if not resumes:
        raise HTTPException(422, "At least one resume file is required.")

    max_bytes = settings.max_resume_size_mb * 1024 * 1024
    resume_sources: list[tuple[bytes, str]] = []
    for file in resumes:
        raw = await file.read()
        if len(raw) > max_bytes:
            raise HTTPException(
                413,
                f"{file.filename!r} exceeds {settings.max_resume_size_mb}MB limit.",
            )
        resume_sources.append((raw, file.filename or ""))

    jd_raw = await jd.read()
    jd_source = (jd_raw, jd.filename or "")

    try:
        result, session_id = run_pipeline_with_persistence(resume_sources, jd_source)
        result.session_id = session_id or None
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(500, f"Pipeline run failed: {e}")
