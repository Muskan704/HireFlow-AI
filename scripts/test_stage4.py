"""
Smoke test for Stage 4 (both scoring paths).

Runs the full chain: parse -> extract -> hard filter -> semantic score -> LLM score,
using the SAME sample resume/JD as test_stage12.py, so you can directly compare
this candidate's scores against what you already know about them (strong Python/
GCP/ML fit, JD asks for Python/TensorFlow/GCP/SQL).

Usage:
  python scripts/test_stage4.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.extractor import extract_resume, extract_jd
from app.services.hard_filter import run_hard_filter
from app.services.scorer_semantic import score_candidate_semantic
from app.services.scorer_llm import score_candidate_llm

settings = get_settings()

# Same sample data as test_stage12.py, on purpose — this candidate should
# clearly PASS the hard filter (has Python, GCP, SQL) and score reasonably
# high on both scorers, since the JD's must-haves are directly present.
SAMPLE_RESUME = """
Priya Sharma
priya.sharma@email.com | Bengaluru, India

EXPERIENCE
Senior Data Scientist — Flipkart, Bengaluru
Jan 2022 – Present (30 months)
- Built recommendation engine serving 10M+ users
- Tech: Python, TensorFlow, Spark, Kubeflow, BigQuery

Data Scientist — Razorpay, Bengaluru
Jul 2020 – Dec 2021 (18 months)
- Fraud detection model, deployed via FastAPI on GCP
- Tech: Python, scikit-learn, XGBoost, FastAPI, GCP

EDUCATION
B.Tech Computer Science — IIT Bombay, 2020

SKILLS
Python, TensorFlow, PyTorch, scikit-learn, XGBoost, Spark,
FastAPI, GCP, SQL, Docker, Kubernetes, MLflow

CERTIFICATIONS
Google Professional ML Engineer (2023)
"""

SAMPLE_JD = """
Senior Data Scientist — FinTech AI Team

Requirements (MUST HAVE):
- 3+ years of experience in data science (required)
- Python (required)
- TensorFlow or PyTorch (required)
- GCP, AWS, or Azure (required)
- SQL (required)

Nice to have:
- MLOps tools (Kubeflow, MLflow)
- Fintech domain experience
"""


def _print_scores(label: str, scores):
    print(f"\n── {label} ──────────────────────────────────────")
    for section in scores.section_scores:
        reasoning = f"  ({section.reasoning})" if section.reasoning else ""
        print(f"  {section.section_name:12s} score={section.score:.2f}  raw={section.raw_score}{reasoning}")


def run():
    print(f"\n{'='*55}")
    print(f"  Stage 4 Smoke Test (semantic + LLM scoring)")
    print(f"  Provider : {settings.active_llm}")
    print(f"  Embedding: {settings.embedding_provider}")
    print(f"{'='*55}")

    # Stage 1+2 (already verified by test_stage12.py — re-running here just
    # to get real ResumeData/JDData objects to feed into Stage 4)
    resume = extract_resume(SAMPLE_RESUME)
    jd = extract_jd(SAMPLE_JD)
    print(f"\nExtracted resume_id: {resume.resume_id}")
    print(f"Candidate: {resume.candidate_name}")

    # Stage 3 — this candidate should PASS
    filter_result = run_hard_filter(resume, jd)
    print(f"\n── HARD FILTER ─────────────────────────────────────")
    print(f"  passed: {filter_result.passed}")
    if not filter_result.passed:
        print(f"  reject_reasons: {filter_result.reject_reasons}")
        print("\n  ⚠ Expected this candidate to PASS. Check must-have skills match.")
        return
    print("  ✓ Passed as expected — proceeding to scoring")

    # Stage 4A — semantic (local, free, no API call)
    semantic_scores = score_candidate_semantic(resume, jd)
    _print_scores("SEMANTIC SCORES (Stage 4A)", semantic_scores)

    # Stage 4B — LLM (costs one API call, uses your ACTIVE_LLM provider)
    llm_scores = score_candidate_llm(resume, jd)
    _print_scores("LLM SCORES (Stage 4B)", llm_scores)

    # Sanity checks
    assert semantic_scores.resume_id == resume.resume_id
    assert llm_scores.resume_id == resume.resume_id
    assert all(0.0 <= s.score <= 1.0 for s in semantic_scores.section_scores)
    assert all(0.0 <= s.score <= 1.0 for s in llm_scores.section_scores)
    print("\n✓ resume_id matches across both scorers")
    print("✓ all scores within 0.0-1.0 range")

    print(f"\n{'='*55}")
    print("  ✓ Stage 4 smoke test passed")
    print(f"  Compare the semantic vs LLM scores above —")
    print(f"  they should be directionally similar (both high,")
    print(f"  since this candidate is a strong match) even if")
    print(f"  the exact numbers differ.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run()