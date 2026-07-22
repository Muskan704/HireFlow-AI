"""
Text cleanup helpers for parsed PDF/DOCX content.

The parser should extract text faithfully. This module removes transport
noise, especially email/export wrappers from Outlook PDFs, before the text is
sent to the LLM extraction stage.
"""

from __future__ import annotations

import re


EMAIL_HEADER_RE = re.compile(
    r"^\s*(from|sent|to|cc|bcc|subject|date)\s*:\s*.+$",
    re.IGNORECASE,
)

EMAIL_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*outlook\s*$", re.IGNORECASE),
    re.compile(r"^\s*external\s*[-:]?.*$", re.IGNORECASE),
    re.compile(r"^\s*original message\s*$", re.IGNORECASE),
    re.compile(r"^\s*-{2,}\s*(original message|forwarded message)\s*-{2,}\s*$", re.IGNORECASE),
    re.compile(r"^\s*on .+ wrote:\s*$", re.IGNORECASE),
    re.compile(r"^\s*CAUTION:.*$", re.IGNORECASE),
    re.compile(r"^\s*This email originated from outside.*$", re.IGNORECASE),
    re.compile(r"^\s*This message and any attachments.*$", re.IGNORECASE),
    re.compile(r"^\s*This e-mail and any attachments.*$", re.IGNORECASE),
    re.compile(r"^\s*Confidentiality notice.*$", re.IGNORECASE),
    re.compile(r"^\s*NOTICE:.*confidential.*$", re.IGNORECASE),
]


def clean_document_text(raw_text: str) -> str:
    """
    Remove common email-thread wrappers from parsed document text.

    Keeps business content such as job title, responsibilities, skills, and
    resume details. Drops transport metadata, reply headers, obvious warning
    banners, and repeated blank lines.
    """
    if not raw_text:
        return ""

    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned: list[str] = []

    skip_quoted_block = False
    for line in lines:
        stripped = line.strip()

        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        if _starts_quoted_history(stripped):
            skip_quoted_block = True
            continue

        if skip_quoted_block and _looks_like_new_content_boundary(stripped):
            skip_quoted_block = False

        if skip_quoted_block:
            continue

        if _is_email_noise(stripped):
            continue

        cleaned.append(stripped)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _is_email_noise(line: str) -> bool:
    if EMAIL_HEADER_RE.match(line):
        return True

    if line.startswith(">"):
        return True

    if EMAIL_ADDRESS_RE.fullmatch(line):
        return True

    return any(pattern.match(line) for pattern in BOILERPLATE_PATTERNS)


def _starts_quoted_history(line: str) -> bool:
    lowered = line.lower()
    return (
        lowered.startswith("-----original message-----")
        or lowered.startswith("----- forwarded message -----")
        or lowered == "original message"
        or (lowered.startswith("from:") and "sent:" in lowered)
    )


def _looks_like_new_content_boundary(line: str) -> bool:
    """
    If a PDF extraction interleaves email history and actual attachments,
    allow useful content to resume at obvious document section headings.
    """
    lowered = line.lower().rstrip(":")
    return lowered in {
        "job description",
        "role",
        "responsibilities",
        "requirements",
        "qualifications",
        "skills",
        "professional summary",
        "work experience",
        "education",
        "projects",
    }
