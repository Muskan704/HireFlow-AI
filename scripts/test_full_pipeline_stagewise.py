"""
Stage-by-stage pipeline walkthrough.

Runs the ENTIRE pipeline one stage at a time, printing the actual input and
output object at every transition — this is the "see under the hood" view:
what Stage 1 hands to Stage 2, what Stage 2 hands to Stage 3, and so on,
all the way through persistence.

Usage:
  python scripts/test_full_pipeline_stagewise.py
  python scripts/test_full_pipeline_stagewise.py path/to/resume.pdf path/to/jd.pdf

Defaults to Resume.pdf / JD.pdf in the scripts/ folder if no args given —
same convention as test_scoring.py.
"""
import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from app.core.config import get_settings
from app.services.parser import parse_document
from app.services.extractor import extract_resume, extract_jd
from app.services.hard_filter import run_hard_filter
from app.services.scorer_semantic import score_candidate_semantic
from app.services.scorer_llm import score_candidate_llm
from app.services.ranker import rank_candidates, load_weights
from app.services.summariser import summarise_candidate

settings = get_settings()


def _banner(stage_num: str, title: str):
    print(f"\n{'═' * 70}")
    print(f"  STAGE {stage_num} — {title}")
    print(f"{'═' * 70}")


def _pretty(obj, max_chars: int = 2000):
    """Pretty-print a Pydantic object as JSON, truncated so huge fields
    (like raw_text_snippet) don't flood the terminal."""
    text = json.dumps(obj.model_dump(mode="json"), indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n  ... [truncated, {len(text)} total chars]"
    print(text)


def run(resume_path: Path, jd_path: Path):
    print(f"\nProvider: {settings.active_llm} | Similarity mode: {settings.similarity_mode}")
    print(f"Resume: {resume_path} | JD: {jd_path}")

    # ── STAGE 1: Parse ──────────────────────────────────────────────────────
    _banner("1", "PARSE — file bytes → plain text")
    resume_raw_text = parse_document(resume_path)
    jd_raw_text = parse_document(jd_path)
    print(f"INPUT:  {resume_path.name} ({resume_path.stat().st_size} bytes)")
    print(f"OUTPUT: {len(resume_raw_text)} chars of plain text")
    print(f"--- first 300 chars ---\n{resume_raw_text[:300]}...")

    # ── STAGE 2: Extract ────────────────────────────────────────────────────
    _banner("2", "EXTRACT — plain text → structured JSON (LLM)")
    print(f"INPUT:  {len(resume_raw_text)} chars of resume text")
    resume = extract_resume(resume_raw_text)
    print(f"OUTPUT: ResumeData —")
    _pretty(resume)

    print(f"\nINPUT:  {len(jd_raw_text)} chars of JD text")
    jd = extract_jd(jd_raw_text)
    print(f"OUTPUT: JDData —")
    _pretty(jd)

    # ── STAGE 3: Hard Filter ────────────────────────────────────────────────
    _banner("3", "HARD FILTER — knockout gate")
    print(f"INPUT:  ResumeData({resume.candidate_name!r}) + JDData({jd.job_title!r})")
    filter_result = run_hard_filter(resume, jd)
    print(f"OUTPUT: FilterResult —")
    _pretty(filter_result)

    if not filter_result.passed:
        print("\n⚠ Candidate REJECTED at Stage 3 — pipeline stops here for this candidate.")
        print("  (Stages 4-6 never run for a rejected candidate — this is the whole point")
        print("   of the hard filter: don't spend LLM calls scoring someone disqualified.)")
        return

    print("\n✓ PASSED — proceeding to Stage 4")

    # ── STAGE 4A: Semantic Scoring ──────────────────────────────────────────
    _banner("4A", "SEMANTIC SCORING — embeddings, no LLM call, free")
    print(f"INPUT:  ResumeData + JDData (same objects as Stage 3)")
    semantic_scores = score_candidate_semantic(resume, jd)
    print(f"OUTPUT: CandidateScores(method='semantic') —")
    _pretty(semantic_scores)

    # ── STAGE 4B: LLM Scoring ───────────────────────────────────────────────
    _banner("4B", "LLM SCORING — one Groq/OpenAI/Anthropic call, temperature=0")
    print(f"INPUT:  ResumeData + JDData (same objects as Stage 3)")
    llm_scores = score_candidate_llm(resume, jd)
    print(f"OUTPUT: CandidateScores(method='llm') —")
    _pretty(llm_scores)

    # ── STAGE 5: Rank ───────────────────────────────────────────────────────
    _banner("5", "RANK — blend both scorers, apply weights.json, sort")
    print(f"INPUT:  [resume], [semantic_scores], [llm_scores]")
    print(f"Weights loaded from weights.json: {json.dumps(load_weights(), indent=2)}")
    ranked = rank_candidates([resume], [semantic_scores], [llm_scores])
    print(f"OUTPUT: RankedCandidate —")
    _pretty(ranked[0])

    # ── STAGE 6: Summarise ──────────────────────────────────────────────────
    _banner("6", "SUMMARISE — grounded LLM writeup of the already-computed facts")
    print(f"INPUT:  ResumeData + JDData + RankedCandidate (rank={ranked[0].rank}, score={ranked[0].overall_score})")
    summary = summarise_candidate(resume, jd, ranked[0])
    print(f"OUTPUT: CandidateSummary —")
    _pretty(summary)

    # ── STAGE 7: Persistence (only if DATABASE_URL is configured) ──────────
    _banner("7", "PERSIST — save everything, then read it back to prove it worked")
    if not settings.database_url:
        print("DATABASE_URL not set in .env — skipping persistence stage.")
        print("(Not an error — the pipeline works fine without it, you just get no history.)")
    else:
        from app.services.session_repo import save_full_session, get_session_detail

        print("Saving full session to the database...")
        session_id = save_full_session(
            jd=jd,
            jd_raw_text=jd_raw_text,
            resumes=[resume],
            resume_raw_texts={resume.resume_id: resume_raw_text},
            filter_results=[filter_result],
            semantic_scores=[semantic_scores],
            llm_scores=[llm_scores],
            ranked_candidates=ranked,
            summaries=[summary],
            weights_used=load_weights(),
        )
        print(f"✓ Saved. session_id = {session_id}")

        print("\nReading it back out of the database (proves persistence actually worked,")
        print("not just that the write call didn't crash) —")
        detail = get_session_detail(session_id)
        print(f"  Session status:      {detail['session'].status}")
        print(f"  JD title (from DB):  {detail['jd'].structured_data.get('job_title')}")
        print(f"  Resumes saved:       {len(detail['resumes'])}")
        print(f"  Filter results:      {len(detail['filter_results'])}")
        print(f"  Score records:       {len(detail['scores'])} (should be 2: one semantic, one llm)")
        print(f"  Ranked candidates:   {len(detail['ranked_candidates'])}")
        print(f"  Summaries:           {len(detail['summaries'])}")

    print(f"\n{'═' * 70}")
    print("  ✓ Full pipeline walkthrough complete — every stage ran and was inspected.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        resume_arg = Path(sys.argv[1])
        jd_arg = Path(sys.argv[2])
    else:
        resume_arg = SCRIPT_DIR / "Resume.pdf"
        jd_arg = SCRIPT_DIR / "JD.pdf"

    if not resume_arg.exists() or not jd_arg.exists():
        print(f"Could not find {resume_arg} or {jd_arg}.")
        print("Usage: python scripts/test_full_pipeline_stagewise.py <resume_path> <jd_path>")
        sys.exit(1)

    run(resume_arg, jd_arg)