'''
"""
FastAPI app — Recruitment Intelligence Platform
Currently exposes Stage 1+2: parse + extract endpoints.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import get_settings
from app.services.parser import parse_document
from app.services.extractor import extract_resume, extract_jd
from app.models.schemas import ResumeData, JDData

settings = get_settings()

# ── Logging setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

app = FastAPI(
    title="Recruitment Intelligence Platform",
    description="AI-powered resume screening and ranking — Pre-Call Module",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_llm": settings.active_llm}


# ── Stage 1+2: Single resume ───────────────────────────────────────────────────

@app.post("/resume/extract", response_model=ResumeData, tags=["Stage 1-2"])
async def extract_resume_endpoint(file: UploadFile = File(...)):
    """
    Upload a single resume (PDF/DOCX/TXT).
    Returns structured JSON from Stage 1 (parse) + Stage 2 (LLM extraction).
    """
    max_bytes = settings.max_resume_size_mb * 1024 * 1024
    raw = await file.read()

    if len(raw) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_resume_size_mb}MB limit.")

    try:
        raw_text = parse_document(raw, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not raw_text.strip():
        raise HTTPException(422, "Could not extract any text from the document.")

    try:
        result = extract_resume(raw_text)
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")

    return result


@app.post("/resume/extract-text", response_model=ResumeData, tags=["Stage 1-2"])
async def extract_resume_from_text(text: str = Form(...)):
    """
    Submit raw resume text directly (for manual/pasted input).
    """
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty.")
    try:
        return extract_resume(text)
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


# ── Stage 1+2: JD ─────────────────────────────────────────────────────────────

@app.post("/jd/extract", response_model=JDData, tags=["Stage 1-2"])
async def extract_jd_endpoint(file: UploadFile = File(...)):
    """Upload a JD file and get structured JSON back."""
    raw = await file.read()
    try:
        raw_text = parse_document(raw, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not raw_text.strip():
        raise HTTPException(422, "Could not extract any text from the document.")

    try:
        return extract_jd(raw_text)
    except Exception as e:
        logger.exception("JD extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


@app.post("/jd/extract-text", response_model=JDData, tags=["Stage 1-2"])
async def extract_jd_from_text(text: str = Form(...)):
    """Submit raw JD text directly."""
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty.")
    try:
        return extract_jd(text)
    except Exception as e:
        logger.exception("JD extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")
'''

'''
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
'''

"""
FastAPI app — Recruitment Intelligence Platform
Exposes Stage 1+2 (parse + extract) endpoints for single-document testing,
plus the Stage 7 /pipeline/run endpoint that runs the full Pre-Call Setup
module end-to-end (bulk resumes + one JD -> ranked, scored, summarised
output).
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import get_settings
from app.services.parser import parse_document
from app.services.extractor import extract_resume, extract_jd
from app.services.pipeline import run_pipeline_with_persistence
from app.models.schemas import ResumeData, JDData
from app.models.results import PipelineResult, ResumeFullDetail, RankedCandidate

settings = get_settings()

# ── Logging setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level=settings.log_level)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

app = FastAPI(
    title="Recruitment Intelligence Platform",
    description="AI-powered resume screening and ranking — Pre-Call Module",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "active_llm": settings.active_llm}


# ── Stage 1+2: Single resume ───────────────────────────────────────────────────

@app.post("/resume/extract", response_model=ResumeData, tags=["Stage 1-2"])
async def extract_resume_endpoint(file: UploadFile = File(...)):
    """
    Upload a single resume (PDF/DOCX/TXT).
    Returns structured JSON from Stage 1 (parse) + Stage 2 (LLM extraction).
    """
    max_bytes = settings.max_resume_size_mb * 1024 * 1024
    raw = await file.read()

    if len(raw) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_resume_size_mb}MB limit.")

    try:
        raw_text = parse_document(raw, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not raw_text.strip():
        raise HTTPException(422, "Could not extract any text from the document.")

    try:
        result = extract_resume(raw_text)
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")

    return result


@app.post("/resume/extract-text", response_model=ResumeData, tags=["Stage 1-2"])
async def extract_resume_from_text(text: str = Form(...)):
    """
    Submit raw resume text directly (for manual/pasted input).
    """
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty.")
    try:
        return extract_resume(text)
    except Exception as e:
        logger.exception("LLM extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


# ── Stage 1+2: JD ─────────────────────────────────────────────────────────────

@app.post("/jd/extract", response_model=JDData, tags=["Stage 1-2"])
async def extract_jd_endpoint(file: UploadFile = File(...)):
    """Upload a JD file and get structured JSON back."""
    raw = await file.read()
    try:
        raw_text = parse_document(raw, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not raw_text.strip():
        raise HTTPException(422, "Could not extract any text from the document.")

    try:
        return extract_jd(raw_text)
    except Exception as e:
        logger.exception("JD extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


@app.post("/jd/extract-text", response_model=JDData, tags=["Stage 1-2"])
async def extract_jd_from_text(text: str = Form(...)):
    """Submit raw JD text directly."""
    if not text.strip():
        raise HTTPException(422, "Text cannot be empty.")
    try:
        return extract_jd(text)
    except Exception as e:
        logger.exception("JD extraction failed")
        raise HTTPException(500, f"Extraction failed: {e}")


# ── Stage 7: full pipeline (bulk resumes + one JD -> ranked report) ──────────

@app.post("/pipeline/run", response_model=PipelineResult, tags=["Pipeline"])
async def run_pipeline_endpoint(
    resumes: list[UploadFile] = File(..., description="Multiple resume files (PDF/DOCX)"),
    jd: UploadFile = File(..., description="One job description file (PDF/DOCX)"),
):
    """
    The end-to-end Pre-Call Setup run: parse + extract every resume and the
    JD, hard-filter, score (semantic/LLM per SIMILARITY_MODE), rank, and
    summarise — all in one call.

    Every submitted resume is accounted for in the response: either in
    ranked_candidates (passed the hard filter) or rejected_candidates
    (failed to parse, failed extraction, or failed the hard filter, each
    with a human-readable reason).

    Size limits are enforced per-file, same as /resume/extract, so one
    oversized file in a batch doesn't silently balloon memory usage.
    """
    if not resumes:
        raise HTTPException(422, "At least one resume file is required.")

    max_bytes = settings.max_resume_size_mb * 1024 * 1024
    resume_sources: list[tuple[bytes, str]] = []
    for f in resumes:
        raw = await f.read()
        if len(raw) > max_bytes:
            raise HTTPException(
                413, f"{f.filename!r} exceeds {settings.max_resume_size_mb}MB limit."
            )
        resume_sources.append((raw, f.filename or ""))

    jd_raw = await jd.read()
    jd_source = (jd_raw, jd.filename or "")

    try:
        result, session_id = run_pipeline_with_persistence(resume_sources, jd_source)
        result.session_id = session_id or None
        return result
    except ValueError as e:
        # e.g. JD produced no extractable text — a client-fixable input problem
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(500, f"Pipeline run failed: {e}")


# ── Task 2: company/JD-scoped read API for the frontend ──────────────────────

@app.get("/companies/{company}/sessions", tags=["History"])
async def list_sessions_for_company(company: str, limit: int = 50):
    """
    Every JD posting (session) run for this company, most recent first.
    Case-insensitive partial match on company name.
    """
    from app.services.session_repo import list_sessions_by_company
    return list_sessions_by_company(company, limit=limit)


@app.get(
    "/sessions/{session_id}/candidates",
    response_model=list[RankedCandidate],
    tags=["History"],
)
async def get_ranked_candidates(session_id: str, include_rejected: bool = False):
    """
    The core 'one JD vs N resumes' view: every candidate for this session,
    sorted best-to-worst by overall_score — this is what the frontend's
    main candidate list should call.

    include_rejected=true also appends rejected candidates (rank=0) after
    the ranked ones, for an admin/full-visibility view.
    """
    from app.services.session_repo import list_resumes_for_session
    return list_resumes_for_session(session_id, passed_only=not include_rejected)


@app.get(
    "/resumes/{resume_id}",
    response_model=ResumeFullDetail,
    tags=["History"],
)
async def get_resume_full_detail(resume_id: str):
    """
    Everything about ONE resume — scores from both scorers, final rank,
    strengths/gaps, detailed agency notes, knowledge brief, and the raw
    resume text. This is the single call a frontend needs to render a
    complete candidate detail page from a resume_id.
    """
    from app.services.session_repo import get_resume_detail
    try:
        return get_resume_detail(resume_id)
    except ValueError as e:
        raise HTTPException(404, str(e))