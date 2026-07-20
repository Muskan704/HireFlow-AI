"""
Stage 4A — Embedding service.

Purpose
-------
This is the layer that generalizes to ARBITRARY resumes and JDs — the exact
concern raised: "we're only handling the built-in example." Nothing here is
hardcoded. It embeds whatever text is actually extracted from whatever file
was actually uploaded, and compares by meaning, not by string.

Model: sentence-transformers/all-MiniLM-L6-v2
- Free, local, no API key, no rate limit, ~80MB, runs on CPU fine.
- Loaded once as a module-level singleton (NOT per-request — reloading a
  transformer model per call is the #1 accidental performance killer here).

Section-level, not whole-document
----------------------------------
We embed skills / experience / other separately, per resume and per JD, so:
  1. We can report *which* section drove a similarity score (explainability).
  2. A resume that's strong on skills but weak on experience phrasing doesn't
     get its skill-match signal diluted by averaging into one big vector.

Integration point
------------------
Called from scorer_semantic.py (next file) with the already-extracted
ResumeData / JDData objects. This module has no knowledge of hard_filter.py
or scoring weights — it only turns text into vectors and compares them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Singleton loader — model is loaded once per process, not per call."""
    return SentenceTransformer(_MODEL_NAME)


def _section_text(items: list[str]) -> str:
    """
    Join a list of free-text items (skills, experience bullet points, etc.)
    into one string for embedding. Works for ANY content — there is no
    assumption here about what the skills/experience actually say.
    """
    return ". ".join(item.strip() for item in items if item and item.strip())


def embed_text(text: str) -> np.ndarray:
    """Embed a single arbitrary string. Returns a normalized vector."""
    if not text.strip():
        # Empty section (e.g. resume has no listed certifications) — return
        # a zero vector rather than crashing. Callers should treat a zero
        # vector's similarity score as "no signal", not "bad match".
        model = _get_model()
        return np.zeros(model.get_sentence_embedding_dimension())
    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector


def embed_resume_sections(
    skills: list[str],
    experience_bullets: list[str],
    other_text: Optional[list[str]] = None,
) -> dict[str, np.ndarray]:
    """
    Embed a resume's sections independently.

    Parameters are plain lists of strings pulled from the already-extracted
    ResumeData object — this function makes no assumptions about their
    content, so it works identically whether the resume says "Kubernetes",
    "K8s", "container orchestration", or something neither of us has thought
    of yet.
    """
    return {
        "skills": embed_text(_section_text(skills)),
        "experience": embed_text(_section_text(experience_bullets)),
        "other": embed_text(_section_text(other_text or [])),
    }


def embed_jd_sections(
    must_have_skills: list[str],
    good_to_have_skills: list[str],
    role_context: str = "",
) -> dict[str, np.ndarray]:
    """Embed a JD's sections independently, same principle as above."""
    return {
        "skills": embed_text(_section_text(must_have_skills + good_to_have_skills)),
        "role_context": embed_text(role_context),
    }


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors, already normalized on encode so
    this reduces to a dot product. Returns 0.0 (no signal) if either vector
    is a zero vector, instead of NaN.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clip for floating point safety — cosine similarity must be in [-1, 1].
    return max(-1.0, min(1.0, similarity))