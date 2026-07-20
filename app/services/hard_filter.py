"""
app/services/hard_filter.py
-----------------------------
Stage 3 — Hard Filter (knockout gate)

Pure Python logic. No LLM calls, no network calls, no randomness.
This means it's fast (microseconds) and 100% testable with
hand-written fixtures — no API key needed.

Checks (all derived from your architecture diagram):
  1. min_experience_months  — candidate total exp >= JD requirement
  2. must_have_skills       — every JD must-have skill in candidate skills
  3. required_certifications — every mandatory cert present in candidate certs
  4. location_mismatch      — only rejects if jd.location_strict=True

What this does NOT check (MVP scope):
  - required_education — informational only, never a reject reason.
    Education requirements in real JDs are usually soft ("Bachelor's preferred")
    and hard-rejecting on this risks too many false negatives.

Matching is STRICT for skills and certs:
  "AWS" in JD must_have_skills → candidate must have "aws" exactly.
  No skill-equivalence groups yet (AWS ≠ GCP at this stage).
"""
from __future__ import annotations
from loguru import logger

from app.models.schemas import ResumeData, JDData
from app.models.results import FilterResult


def run_hard_filter(resume: ResumeData, jd: JDData) -> FilterResult:
    """
    Check ONE candidate against ONE JD's non-negotiable requirements.

    Returns FilterResult with:
      passed=True  → candidate moves to Stage 4
      passed=False → candidate rejected, pipeline stops here for them

    The `checks` dict records EVERY individual check (pass AND fail)
    so you can see the full picture, not just the first failure.
    """
    checks: dict[str, bool] = {}
    reject_reasons: list[str] = []

    # ── Check 1: Minimum experience ─────────────────────────────────────────────
    # Directly from diagram: "experience below minimum? → REJECT"
    if jd.min_experience_months is not None:
        candidate_exp = resume.total_experience_months or 0
        exp_ok = candidate_exp >= jd.min_experience_months
        checks["min_experience"] = exp_ok

        if not exp_ok:
            reject_reasons.append(
                f"Experience too low: candidate has {candidate_exp} months, "
                f"JD requires minimum {jd.min_experience_months} months"
            )
    else:
        checks["min_experience"] = True  # JD has no exp requirement

    # ── Check 2: Must-have skills (strict exact match) ──────────────────────────
    # Directly from diagram: "missing mandatory skill? → REJECT"
    if jd.normalised_must_have_skills:
        candidate_skill_set = set(resume.normalised_skills)
        missing_skills = [
            skill for skill in jd.normalised_must_have_skills
            if skill not in candidate_skill_set
        ]
        skills_ok = len(missing_skills) == 0
        checks["must_have_skills"] = skills_ok

        if not skills_ok:
            reject_reasons.append(
                f"Missing required skills: {', '.join(missing_skills)}"
            )
    else:
        checks["must_have_skills"] = True  # JD has no must-have skills

    # ── Check 2b: Skill groups — "X, Y, or Z (required)" style ──────────────────
    # jd.must_have_skill_groups already existed in the schema for this, but was
    # never actually checked here — meaning "GCP, AWS, or Azure (required)"
    # JDs were silently falling through to Check 2 as three separate AND'd
    # requirements, rejecting perfectly qualified candidates who only had one
    # of the three. This check fixes that: each group needs only ONE match.
    if jd.must_have_skill_groups:
        candidate_skill_set = set(resume.normalised_skills)
        failed_groups: list[list[str]] = []

        for group in jd.must_have_skill_groups:
            normalised_group = [s.lower().strip() for s in group]
            group_satisfied = any(
                skill in candidate_skill_set for skill in normalised_group
            )
            if not group_satisfied:
                failed_groups.append(group)

        groups_ok = len(failed_groups) == 0
        checks["must_have_skill_groups"] = groups_ok

        if not groups_ok:
            for group in failed_groups:
                reject_reasons.append(
                    f"Missing required skill (need at least one of): {', '.join(group)}"
                )
    else:
        checks["must_have_skill_groups"] = True  # JD has no OR-style requirements

    # ── Check 3: Mandatory certifications ───────────────────────────────────────
    # Directly from diagram: "missing mandatory cert? → REJECT"
    if jd.required_certifications:
        # Normalise candidate cert names the same way we normalise skills
        candidate_certs = {
            cert.name.lower().strip()
            for cert in resume.certifications
        }

        # Normalise JD required cert names
        missing_certs = [
            cert for cert in jd.required_certifications
            if cert.lower().strip() not in candidate_certs
        ]
        certs_ok = len(missing_certs) == 0
        checks["required_certifications"] = certs_ok

        if not certs_ok:
            reject_reasons.append(
                f"Missing mandatory certifications: {', '.join(missing_certs)}"
            )
    else:
        checks["required_certifications"] = True  # JD requires no specific certs

    # ── Check 4: Location (only active if location_strict=True in JD) ───────────
    # Directly from diagram: "location mismatch (if flag)? → REJECT"
    # The "if flag" part is jd.location_strict — defaults to False
    if jd.location and jd.location_strict:
        candidate_location = (resume.location or "").lower().strip()
        jd_location = jd.location.lower().strip()

        # Simple substring check: "bengaluru" matches "bengaluru, india"
        # and "remote" matches anything containing "remote"
        location_ok = (
            jd_location in candidate_location
            or candidate_location in jd_location
            or "remote" in candidate_location
            or "remote" in jd_location
        )
        checks["location_match"] = location_ok

        if not location_ok:
            reject_reasons.append(
                f"Location mismatch: candidate is in '{resume.location}', "
                f"JD requires '{jd.location}'"
            )
    else:
        # location_strict is False (default) — log it but never reject on it
        checks["location_match"] = True

    # ── Informational only — education ──────────────────────────────────────────
    # NOT in reject_reasons, just recorded for reference
    if jd.required_education:
        checks["has_education_listed"] = len(resume.education) > 0

    # ── Final verdict ────────────────────────────────────────────────────────────
    passed = len(reject_reasons) == 0

    result = FilterResult(
        resume_id=resume.resume_id,
        candidate_name=resume.candidate_name,
        passed=passed,
        reject_reasons=reject_reasons,
        checks=checks,
    )

    if passed:
        logger.info(
            f"✓ PASSED  | {resume.candidate_name} | "
            f"exp={resume.total_experience_months}mo | "
            f"checks={checks}"
        )
    else:
        logger.info(
            f"✗ REJECTED | {resume.candidate_name} | "
            f"reasons={reject_reasons}"
        )

    return result


def run_hard_filter_bulk(
    resumes: list[ResumeData],
    jd: JDData,
) -> tuple[list[ResumeData], list[FilterResult]]:
    """
    Run the hard filter across ALL resumes for one JD.

    Returns:
      passed_resumes  — only candidates who passed, forwarded to Stage 4
      all_results     — every result (pass + reject) for the final report

    Why return both?
    Stage 4 only processes passed_resumes (saves LLM/embedding cost).
    But the final pipeline output needs all_results so rejected candidates
    still appear in the output with their reject reasons.
    """
    passed_resumes: list[ResumeData] = []
    all_results: list[FilterResult] = []

    for resume in resumes:
        result = run_hard_filter(resume, jd)
        all_results.append(result)
        if result.passed:
            passed_resumes.append(resume)

    total = len(resumes)
    passed = len(passed_resumes)
    logger.info(
        f"Hard filter complete | "
        f"{passed} passed / {total - passed} rejected / {total} total"
    )

    return passed_resumes, all_results