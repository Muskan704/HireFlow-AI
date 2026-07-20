"""
Stage 1 — Document Parser
Extracts raw text from PDF, DOCX, or plain text files.
OCR fallback for image-heavy PDFs.
"""
from __future__ import annotations
import io
from pathlib import Path
from loguru import logger


def parse_document(source: str | Path | bytes, filename: str = "") -> str:
    """
    Accept a file path, a Path object, or raw bytes.
    Returns clean extracted text.
    Raises ValueError if the format is unsupported.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        filename = filename or path.name
        raw_bytes = path.read_bytes()
    else:
        raw_bytes = source

    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(raw_bytes, filename)
    elif ext in (".docx", ".doc"):
        return _parse_docx(raw_bytes)
    elif ext in (".txt", ".md", ""):
        return raw_bytes.decode("utf-8", errors="replace").strip()
    else:
        # Try PDF first, then DOCX — handles files uploaded without extension
        for fn in (_parse_pdf, _parse_docx):
            try:
                text = fn(raw_bytes, filename) if fn == _parse_pdf else fn(raw_bytes)
                if text.strip():
                    return text
            except Exception:
                continue
        raise ValueError(f"Unsupported or unreadable file: {filename!r}")


def _parse_pdf(raw_bytes: bytes, filename: str = "") -> str:
    """
    Primary: PyMuPDF text extraction.
    Fallback: pytesseract OCR if extracted text is suspiciously short.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    pages_text: list[str] = []

    for page in doc:
        pages_text.append(page.get_text("text"))

    full_text = "\n".join(pages_text).strip()

    # OCR fallback — if we got almost nothing, the PDF is likely image-based
    if len(full_text) < 100:
        logger.warning(
            f"PDF {filename!r} yielded <100 chars — attempting OCR fallback."
        )
        full_text = _ocr_pdf(doc) or full_text

    doc.close()
    return full_text


def _ocr_pdf(doc) -> str:
    """Rasterise each page and run pytesseract."""
    try:
        import pytesseract
        from PIL import Image

        texts: list[str] = []
        import fitz  # PyMuPDF — needed here for fitz.Matrix

        for page in doc:
            # Render at 2x resolution for better OCR accuracy.
            # NOTE: previous version called page.get_pixmap().irect.Matrix(2, 2),
            # which is not a valid PyMuPDF call and would raise AttributeError
            # the first time OCR fallback actually triggered (image-heavy PDF).
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            texts.append(pytesseract.image_to_string(img))
        return "\n".join(texts).strip()
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def _parse_docx(raw_bytes: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()


def parse_many(
    sources: list[tuple[bytes, str]],
) -> list[tuple[str, str | None]]:
    """
    Bulk parse.
    sources: list of (raw_bytes, filename)
    Returns list of (extracted_text, error_message | None)
    """
    results = []
    for raw_bytes, filename in sources:
        try:
            text = parse_document(raw_bytes, filename)
            results.append((text, None))
            logger.debug(f"Parsed {filename!r}: {len(text)} chars")
        except Exception as e:
            logger.error(f"Failed to parse {filename!r}: {e}")
            results.append(("", str(e)))
    return results