"""
scripts/test_stage3.py
------------------------
Tests Stage 3 (hard filter) — ALL four checks from the architecture diagram:
  1. missing mandatory skill    → REJECT
  2. experience below minimum   → REJECT
  3. missing mandatory cert     → REJECT
  4. location mismatch (if flag)→ REJECT

No LLM calls — pure logic test, runs in milliseconds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import ResumeData, JDData
from app.models.schemas import Certification
from app.services.hard_filter import run_hard_filter, run_hard_filter_bulk


# ── Base JD used across all tests ───────────────────────────────────────────────
JD = JDData(
    job_title="Senior Data Scientist",
    min_experience_months=36,
    must_have_skills=["Python", "TensorFlow", "AWS"],
    required_certifications=["AWS Solutions Architect"],
    location="Bengaluru",
    location_strict=False,  # default — location won't cause rejection
)


def test_1_full_pass():
    """All checks pass — candidate moves to Stage 4."""
    resume = ResumeData(
        candidate_name="Candidate A — Full Pass",
        total_experience_months=48,
        skills=["Python", "TensorFlow", "AWS", "Docker"],
        certifications=[Certification(name="AWS Solutions Architect")],
        location="Bengaluru, India",
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is True
    assert result.checks["min_experience"] is True
    assert result.checks["must_have_skills"] is True
    assert result.checks["required_certifications"] is True
    assert result.checks["location_match"] is True
    print(f"✓ Test 1 — Full pass: {result.candidate_name}")


def test_2_experience_fail():
    """Has all skills and certs but not enough experience."""
    resume = ResumeData(
        candidate_name="Candidate B — Exp Fail",
        total_experience_months=18,
        skills=["Python", "TensorFlow", "AWS"],
        certifications=[Certification(name="AWS Solutions Architect")],
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is False
    assert result.checks["min_experience"] is False
    assert result.checks["must_have_skills"] is True
    assert "Experience too low" in result.reject_reasons[0]
    print(f"✓ Test 2 — Exp fail: {result.reject_reasons}")


def test_3_skills_fail():
    """Has experience and cert but missing AWS (has GCP instead)."""
    resume = ResumeData(
        candidate_name="Candidate C — Skills Fail",
        total_experience_months=60,
        skills=["Python", "TensorFlow", "GCP"],  # GCP ≠ AWS (strict match)
        certifications=[Certification(name="AWS Solutions Architect")],
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is False
    assert result.checks["must_have_skills"] is False
    assert "aws" in result.reject_reasons[0].lower()
    print(f"✓ Test 3 — Skills fail: {result.reject_reasons}")
    print(f"   Confirmed: GCP does NOT substitute for AWS (strict matching)")


def test_4_cert_fail():
    """Has experience and skills but missing the mandatory certification."""
    resume = ResumeData(
        candidate_name="Candidate D — Cert Fail",
        total_experience_months=48,
        skills=["Python", "TensorFlow", "AWS"],
        certifications=[],  # no certs at all
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is False
    assert result.checks["required_certifications"] is False
    assert "certification" in result.reject_reasons[0].lower()
    print(f"✓ Test 4 — Cert fail: {result.reject_reasons}")


def test_5_location_fail():
    """Location mismatch rejects only when location_strict=True."""
    jd_strict = JDData(
        job_title="Senior Data Scientist",
        min_experience_months=36,
        must_have_skills=["Python"],
        required_certifications=[],
        location="Bengaluru",
        location_strict=True,  # now active
    )
    resume = ResumeData(
        candidate_name="Candidate E — Location Fail",
        total_experience_months=48,
        skills=["Python", "TensorFlow", "AWS"],
        location="Mumbai",
    )
    result = run_hard_filter(resume, jd_strict)
    assert result.passed is False
    assert result.checks["location_match"] is False
    assert "Location mismatch" in result.reject_reasons[0]
    print(f"✓ Test 5 — Location fail (strict=True): {result.reject_reasons}")


def test_6_location_ignored_when_not_strict():
    """Location mismatch does NOT reject when location_strict=False (default)."""
    resume = ResumeData(
        candidate_name="Candidate F — Location ignored",
        total_experience_months=48,
        skills=["Python", "TensorFlow", "AWS"],
        certifications=[Certification(name="AWS Solutions Architect")],
        location="Mumbai",  # different city but location_strict=False on JD
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is True  # location difference ignored
    assert result.checks["location_match"] is True
    print(f"✓ Test 6 — Location ignored when strict=False: passed correctly")


def test_7_multiple_failures():
    """Candidate fails experience, skills, AND cert — all reasons recorded."""
    resume = ResumeData(
        candidate_name="Candidate G — Multiple Fails",
        total_experience_months=6,
        skills=["Java"],
        certifications=[],
    )
    result = run_hard_filter(resume, JD)
    assert result.passed is False
    assert len(result.reject_reasons) == 3
    print(f"✓ Test 7 — Multiple failures ({len(result.reject_reasons)} reasons):")
    for r in result.reject_reasons:
        print(f"   - {r}")


def test_8_bulk():
    """Bulk filter processes a mixed batch and correctly splits pass/reject."""
    resumes = [
        ResumeData(candidate_name="Bulk Pass 1", total_experience_months=48,
                   skills=["Python", "TensorFlow", "AWS"],
                   certifications=[Certification(name="AWS Solutions Architect")]),
        ResumeData(candidate_name="Bulk Reject — exp", total_experience_months=12,
                   skills=["Python", "TensorFlow", "AWS"],
                   certifications=[Certification(name="AWS Solutions Architect")]),
        ResumeData(candidate_name="Bulk Pass 2", total_experience_months=60,
                   skills=["Python", "TensorFlow", "AWS", "Spark"],
                   certifications=[Certification(name="AWS Solutions Architect")]),
        ResumeData(candidate_name="Bulk Reject — skills + cert",
                   total_experience_months=48,
                   skills=["Java"], certifications=[]),
    ]
    passed, all_results = run_hard_filter_bulk(resumes, JD)
    assert len(all_results) == 4
    assert len(passed) == 2
    print(f"✓ Test 8 — Bulk: {len(passed)}/{len(resumes)} passed correctly")


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Stage 3 Hard Filter — Full Test Suite")
    print(f"  All 4 checks from architecture diagram")
    print(f"{'='*55}\n")

    test_1_full_pass()
    test_2_experience_fail()
    test_3_skills_fail()
    test_4_cert_fail()
    test_5_location_fail()
    test_6_location_ignored_when_not_strict()
    test_7_multiple_failures()
    test_8_bulk()

    print(f"\n{'='*55}")
    print(f"  ✓ All 8 tests passed — Stage 3 complete")
    print(f"{'='*55}\n")