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
