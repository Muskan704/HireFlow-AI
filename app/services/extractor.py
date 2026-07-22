"""
Stage 2 — LLM Extraction
Converts raw document text into validated Pydantic v2 objects using Instructor.
"""
from __future__ import annotations
from app.llm.factory import get_llm
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_result

from app.core.config import get_settings
from app.models.schemas import ResumeData, JDData

settings = get_settings()

RESUME_SYSTEM_PROMPT = """
You are an expert recruitment analyst. Extract structured information from the
resume text provided. Follow these rules precisely:

1. Compute total_experience_months by summing all individual work durations.
   Example: 30 months at Company A + 18 months at Company B = 48 months total.

2. Normalise ALL skill names to their full common form:
   "Py" → "Python", "TF" → "TensorFlow", "k8s" → "Kubernetes", "JS" → "JavaScript"

3. Put skills in BOTH the specific list AND the main skills list:
   programming_languages: ["Python", "SQL"]
   frameworks_and_tools:  ["FastAPI", "Docker"]
   skills:                ["Python", "SQL", "FastAPI", "Docker"]  ← union of everything

4. If a field is genuinely absent, return null or [] — do NOT hallucinate.

5. Convert all date ranges to months:
   "Jan 2021 – Jun 2023" = 29 months
   "2020 – Present"      = calculate up to current year

6. COMPLETENESS IS CRITICAL — this skills list is used for automated
   pass/fail candidate screening. Extract EVERY technology, tool, protocol,
   or skill explicitly named anywhere in the resume, including ones
   embedded inside a comma-separated list (e.g. if the text says
   "MongoDB, RESTful APIs, MySQL", ALL THREE must appear as separate
   entries — do not skip or merge any of them, even if the list is long).
   Missing a skill here can incorrectly disqualify a real candidate.
   Re-check your skills list against the source text before finishing:
   every technology word mentioned in the resume should appear somewhere
   in your output.
"""


def _resume_extraction_looks_incomplete(result: ResumeData) -> bool:
    """
    Detects a suspiciously thin extraction — the kind of run that produced
    21 skills on one pass and 91-98 on another for the IDENTICAL resume
    text. This can't be caught by tenacity's normal exception-based retry,
    since the call didn't fail, it just produced a low-quality result.

    Heuristic, not a guarantee: fewer than 5 skills AND fewer than 2 work
    experience entries on a resume that clearly has real content (we only
    call this after a real resume was already successfully parsed with
    meaningful text) is a strong signal of a bad extraction, not a
    genuinely sparse resume. This will occasionally trigger an unnecessary
    retry on a real one-job/few-skills resume — that costs a few extra
    seconds and one more API call, which is a much smaller cost than
    silently accepting a broken extraction that could wrongly reject a
    real candidate at Stage 3.
    """
    return len(result.skills) < 5 and len(result.work_experience) < 2


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    retry=(retry_if_exception_type(Exception) | retry_if_result(_resume_extraction_looks_incomplete)),
)
def extract_resume(raw_text: str) -> ResumeData:
    llm = get_llm()
    provider = settings.active_llm

    logger.debug(f"Extracting resume ({len(raw_text)} chars) via {provider}")

    result: ResumeData = llm.structured_completion(
        system_prompt=RESUME_SYSTEM_PROMPT,
        user_prompt=f"<resume>\n{raw_text}\n</resume>",
        response_model=ResumeData,
        temperature=0.1,
        # Keep enough output room for detailed structured resumes, but avoid
        # reserving so many tokens that Groq rejects the request before
        # generation starts on the on-demand TPM tier.
        max_tokens=3500,
    )

    result.raw_text_snippet = raw_text[:300]

    if _resume_extraction_looks_incomplete(result):
        logger.warning(
            f"Extraction for {result.candidate_name!r} looks suspiciously thin "
            f"(skills={len(result.skills)}, jobs={len(result.work_experience)}) "
            f"on a {len(raw_text)}-char resume — tenacity will retry this."
        )

    import uuid
    result.resume_id = str(uuid.uuid4())

    logger.info(
        f"✓ Resume extracted: name={result.candidate_name!r} | "
        f"exp={result.total_experience_months}mo | "
        f"skills={len(result.skills)} | provider={provider}"
    )
    return result


JD_SYSTEM_PROMPT = """
You are an expert recruitment analyst. Extract structured requirements from the
job description provided. Follow these rules precisely:

1. Separate must-have from nice-to-have skills:
   must_have_skills    = skills described as "required", "mandatory", "must have"
   nice_to_have_skills = skills described as "preferred", "nice to have", "a plus"

1a. CRITICAL — must_have_skills is a HARD REJECT GATE using EXACT TEXT
    MATCHING against the resume. A candidate is automatically rejected if
    ANY entry here doesn't literally appear (or alias-match) in their
    resume. This means must_have_skills may ONLY contain concrete,
    specific, individually-nameable technologies, tools, software,
    languages, platforms, or certifications — things a resume would
    plausibly state as an exact phrase (e.g. "Excel", "Oracle", "SQL",
    "AWS", "SAP"). Getting this wrong causes REAL, QUALIFIED CANDIDATES TO
    BE WRONGLY REJECTED — this is the single most important rule in this
    prompt.

    NEVER put these into must_have_skills, even if the JD calls them
    "required" or "must have":
      - Education/degree requirements (e.g. "Bachelor's degree in
        accounting", "degree in Business Administration") — these belong
        ONLY in required_education, never in must_have_skills.
      - Soft skills, behavioral traits, or general working style (e.g.
        "team environment experience", "strong communication skills",
        "ability to multitask", "self-motivated", "detail oriented",
        "organized", "works independently") — these cannot be verified by
        exact text matching and must NEVER cause an automatic rejection.
        Omit them entirely — do not put them anywhere in must_have_skills,
        must_have_skill_groups, or nice_to_have_skills.
      - Vague qualifiers or "or similar/equivalent" phrasing — exact
        matching cannot fairly judge "or similar" or skill *depth*. For
        these: extract ONLY the concrete named technology and put it in
        nice_to_have_skills — it must NOT appear in must_have_skills at
        all, under any circumstances.

        WORKED EXAMPLE (this exact pattern has caused real false
        rejections — follow it precisely):
          JD text: "Oracle or similar ERP system experience"
          WRONG:   must_have_skills = ["Oracle"]   <- causes false rejects
          RIGHT:   must_have_skills = []  (nothing added for this line)
                   nice_to_have_skills = ["Oracle"]
          Reasoning: a candidate with SAP, NetSuite, or any other ERP
          system experience but not literally "Oracle" is CLEARLY
          qualified for this requirement, but exact-match hard filtering
          would incorrectly reject them if "Oracle" were in
          must_have_skills. The word "similar" in the JD is explicit
          permission that alternatives are acceptable — must_have_skills
          cannot represent "acceptable alternatives," only
          must_have_skill_groups or nice_to_have_skills can.

        Second example: "advanced level Excel skills" ->
          must_have_skills = [] for this line, nice_to_have_skills = ["Excel"]
          (the word "advanced" is a depth qualifier, not exact-matchable)

    When genuinely unsure whether something belongs in must_have_skills,
    default to nice_to_have_skills instead. A missed nice-to-have costs a
    few scoring points; a wrongly-placed must-have silently rejects a
    real, qualified candidate before a human ever sees them — the two
    mistakes are not equally bad, always err toward the smaller one.

1b. CRITICAL — distinguish AND-requirements from OR-requirements:
    If the JD requires a candidate to have ALL of several specific skills
    individually, put each one in must_have_skills.
    If the JD requires ANY ONE of several alternatives (e.g. "GCP, AWS, or
    Azure", "Python or Java", "React or Vue experience"), that is an
    OR-group — put it in must_have_skill_groups as one inner list, e.g.
    must_have_skill_groups = [["GCP", "AWS", "Azure"]]
    Do NOT put OR-alternatives into must_have_skills as separate entries —
    that incorrectly requires the candidate to have ALL of them, rejecting
    candidates who satisfy the JD with just one. A JD can have both:
    must_have_skills for individually-required skills, AND
    must_have_skill_groups for "one of these" requirements, at the same time.

2. Convert experience to months:
   "3+ years" → 36,  "2-5 years" → 24 (use the minimum),  "5 years" → 60

3. min_experience_months = the absolute MINIMUM stated.
   If JD says "3-5 years", use 36. This feeds the hard filter — be conservative.

4. Normalise skill names the same way as resumes.

5. If a field is genuinely absent, return null or [] — do NOT hallucinate.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def extract_jd(raw_text: str) -> JDData:
    llm = get_llm()
    provider = settings.active_llm

    logger.debug(f"Extracting JD ({len(raw_text)} chars) via {provider}")

    result: JDData = llm.structured_completion(
        system_prompt=JD_SYSTEM_PROMPT,
        user_prompt=f"<job_description>\n{raw_text}\n</job_description>",
        response_model=JDData,
        temperature=0.1,
        max_tokens=2500,
    )

    result.raw_text_snippet = raw_text[:300]

    logger.info(
        f"✓ JD extracted: title={result.job_title!r} | "
        f"min_exp={result.min_experience_months}mo | "
        f"must_have={result.must_have_skills} | "
        f"skill_groups={result.must_have_skill_groups} | provider={provider}"
    )

    return result


def extract_resumes_bulk(raw_texts: list[str]) -> list[tuple[ResumeData | None, str | None]]:
    results = []
    for i, text in enumerate(raw_texts):
        try:
            data = extract_resume(text)
            results.append((data, None))
        except Exception as e:
            logger.error(f"Resume #{i} extraction failed: {e}")
            results.append((None, str(e)))
    return results
