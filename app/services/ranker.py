"""
Stage 5 — Ranking (README-aligned version).

Applies weighted formula from weights.json to compute final scores:

final_score = (
  0.40 × must_have_skill_coverage +
  0.20 × experience_score +           (0.15 total_exp + 0.05 role_match)
  0.10 × good_to_have_coverage +
  0.10 × domain_knowledge_coverage +
  0.08 × semantic_similarity_boost +
  0.05 × certification_coverage +
  0.05 × project_relevance +
  0.02 × summary_score
)

Deterministic ranking: same inputs → same order every time.
Tiebreaker: higher must_have_skills score wins.
"""

from __future__ import annotations

import json
from pathlib import Path
from loguru import logger

from app.models.schemas import ResumeData
from app.models.results import CandidateScores, RankedCandidate, SectionScore
from app.core.config import get_settings

settings = get_settings()


# ── Weights Loading ──────────────────────────────────────────────────────────────

def load_weights() -> dict:
    """Load weights from weights.json."""
    weights_path = Path(__file__).parent.parent.parent / "weights.json"
    try:
        with open(weights_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("weights.json not found, using defaults")
        return {
            "skill_matching": {
                "must_have_skills": 0.40,
                "good_to_have_skills": 0.10,
                "domain_knowledge": 0.10
            },
            "experience_matching": {
                "total_experience": 0.15,
                "role_match": 0.05
            },
            "additional_components": {
                "semantic_similarity_boost": 0.08,
                "certifications": 0.05,
                "project_relevance": 0.05,
                "summary_score": 0.02
            },
            "blend_ratio": 0.5
        }


# ── Score Extraction Helper ─────────────────────────────────────────────────────

def _get_section_score(scores: CandidateScores, section_name: str) -> float:
    """Extract a section score by name, defaulting to 0.0 if not found."""
    for section in scores.section_scores:
        if section.section_name == section_name:
            return section.score
    return 0.0


# ── Blending Function ────────────────────────────────────────────────────────────

def _blend_section_score(
    section_name: str,
    semantic_scores: CandidateScores,
    llm_scores: CandidateScores | None,
    mode: str,
    blend_ratio: float,
) -> float:
    """
    Blend semantic and LLM scores for a section based on mode.
    
    Modes:
      - "semantic": use only semantic score
      - "llm": use only LLM score
      - "both": blend with ratio (default 0.5 each)
    """
    semantic_score = _get_section_score(semantic_scores, section_name)
    
    if mode == "semantic":
        return semantic_score
    
    if llm_scores is None:
        return semantic_score
    
    llm_score = _get_section_score(llm_scores, section_name)
    
    if mode == "llm":
        return llm_score
    
    # "both" mode: weighted blend
    return blend_ratio * llm_score + (1 - blend_ratio) * semantic_score


# ── Final Score Computation ──────────────────────────────────────────────────────

def _compute_weighted_score(
    semantic_scores: CandidateScores,
    llm_scores: CandidateScores | None,
    weights: dict,
    mode: str,
) -> tuple[float, dict[str, float]]:
    """
    Compute the final weighted score using the README formula.
    
    Returns (final_score, score_breakdown) for debug/transparency.
    """
    blend_ratio = weights.get("blend_ratio", 0.5)
    
    skill_weights = weights.get("skill_matching", {})
    exp_weights = weights.get("experience_matching", {})
    add_weights = weights.get("additional_components", {})
    
    # Compute each component
    must_have = _blend_section_score("must_have_skills", semantic_scores, llm_scores, mode, blend_ratio)
    good_to_have = _blend_section_score("good_to_have_skills", semantic_scores, llm_scores, mode, blend_ratio)
    domain = _blend_section_score("domain_knowledge", semantic_scores, llm_scores, mode, blend_ratio)
    certs = _blend_section_score("certifications", semantic_scores, llm_scores, mode, blend_ratio)
    
    total_exp = _blend_section_score("total_experience", semantic_scores, llm_scores, mode, blend_ratio)
    role_match = _blend_section_score("role_match", semantic_scores, llm_scores, mode, blend_ratio)
    
    semantic_boost = _blend_section_score("semantic_boost", semantic_scores, llm_scores, mode, blend_ratio)
    project_rel = _blend_section_score("project_relevance", semantic_scores, llm_scores, mode, blend_ratio)
    summary = _blend_section_score("summary_score", semantic_scores, llm_scores, mode, blend_ratio)
    
    # Weighted sum
    final_score = (
        skill_weights.get("must_have_skills", 0.40) * must_have +
        skill_weights.get("good_to_have_skills", 0.10) * good_to_have +
        skill_weights.get("domain_knowledge", 0.10) * domain +
        add_weights.get("certifications", 0.05) * certs +
        exp_weights.get("total_experience", 0.15) * total_exp +
        exp_weights.get("role_match", 0.05) * role_match +
        add_weights.get("semantic_similarity_boost", 0.08) * semantic_boost +
        add_weights.get("project_relevance", 0.05) * project_rel +
        add_weights.get("summary_score", 0.02) * summary
    )
    
    # Build breakdown for transparency
    breakdown = {
        "must_have_skills": must_have,
        "good_to_have_skills": good_to_have,
        "domain_knowledge": domain,
        "certifications": certs,
        "total_experience": total_exp,
        "role_match": role_match,
        "semantic_boost": semantic_boost,
        "project_relevance": project_rel,
        "summary_score": summary,
    }
    
    return round(final_score, 4), breakdown


# ── Main Ranking Function ────────────────────────────────────────────────────────

def rank_candidates(
    resumes: list[ResumeData],
    semantic_scores: list[CandidateScores],
    llm_scores: list[CandidateScores],
) -> list[RankedCandidate]:
    """
    Rank candidates by weighted score.
    
    Deterministic: same inputs → same order.
    Tiebreaker: higher must_have_skills score wins.
    """
    weights = load_weights()
    mode = settings.similarity_mode

    # Flatten the nested weights.json structure into one dict for
    # RankedCandidate.weights_used — done once here, not per-candidate,
    # since it's the same for every candidate in this ranking run.
    flat_weights_used: dict[str, float] = {
        **weights.get("skill_matching", {}),
        **weights.get("experience_matching", {}),
        **weights.get("additional_components", {}),
        "blend_ratio": weights.get("blend_ratio", 0.5),
    }
    
    # Index scores by resume_id for lookup
    semantic_by_id = {s.resume_id: s for s in semantic_scores}
    llm_by_id = {s.resume_id: s for s in llm_scores}
    
    ranked: list[RankedCandidate] = []
    
    for resume in resumes:
        semantic = semantic_by_id.get(resume.resume_id)
        llm = llm_by_id.get(resume.resume_id)
        
        if not semantic:
            logger.warning(f"No semantic scores for {resume.candidate_name}, skipping")
            continue
        
        # Compute final weighted score
        final_score, breakdown = _compute_weighted_score(
            semantic, llm, weights, mode
        )
        
        ranked.append(
            RankedCandidate(
                resume_id=resume.resume_id,
                candidate_name=resume.candidate_name,
                rank=0,  # Assigned after sorting
                overall_score=final_score,
                section_scores=breakdown,
                weights_used=flat_weights_used,
            )
        )
    
    # Sort by score descending, with deterministic tiebreaker
    # Tiebreaker: higher must_have_skills score
    ranked.sort(key=lambda c: (c.overall_score, c.section_scores.get("must_have_skills", 0)), reverse=True)
    
    # Assign ranks
    for i, candidate in enumerate(ranked, 1):
        candidate.rank = i
    
    logger.info(
        f"Ranked {len(ranked)} candidates | mode={mode} | "
        f"weights=skill_matching({weights.get('skill_matching', {})}) | "
        f"blend_ratio={weights.get('blend_ratio', 0.5)}"
    )
    
    if ranked:
        logger.info(
            f"#{ranked[0].rank}: {ranked[0].candidate_name} "
            f"(score={ranked[0].overall_score:.4f})"
        )
    
    return ranked