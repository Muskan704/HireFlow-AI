"""
Smoke test for Stage 1+2.
Run this after pasting all files to confirm everything works.

Usage:
  python scripts/test_stage12.py                     # uses built-in sample
  python scripts/test_stage12.py path/to/resume.pdf  # uses a real file
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.services.extractor import extract_resume, extract_jd

settings = get_settings()

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


def run():
    print(f"\n{'='*55}")
    print(f"  Stage 1+2 Smoke Test")
    print(f"  Provider : {settings.active_llm}")
    print(f"{'='*55}\n")

    # Resume
    if len(sys.argv) > 1:
        from app.services.parser import parse_document
        raw = parse_document(Path(sys.argv[1]))
        resume = extract_resume(raw)
    else:
        resume = extract_resume(SAMPLE_RESUME)

    print("── RESUME ────────────────────────────────────────────")
    print(json.dumps(resume.model_dump(exclude_none=True), indent=2))

    print("\n── COMPUTED FIELDS ───────────────────────────────────")
    print(f"normalised_skills : {resume.normalised_skills[:5]}...")
    print(f"total_exp_months  : {resume.total_experience_months}")

    # Verify normalisation
    assert resume.normalised_skills == [s.lower().strip() for s in resume.skills]
    print("✓ normalised_skills correct")

    # JD
    jd = extract_jd(SAMPLE_JD)
    print("\n── JD ────────────────────────────────────────────────")
    print(json.dumps(jd.model_dump(exclude_none=True), indent=2))
    print(f"\nnormalised_must_have : {jd.normalised_must_have_skills}")
    assert jd.normalised_must_have_skills == [s.lower().strip() for s in jd.must_have_skills]
    print("✓ normalised_must_have_skills correct")

    print(f"\n{'='*55}")
    print("  ✓ All checks passed")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run() 

    