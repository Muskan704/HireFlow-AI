"""
Batch stage-by-stage pipeline test — runs MULTIPLE resume/JD pairs through
all 7 stages and writes a single readable report file, so you can review
several real candidates side by side instead of re-running the single-pair
script one at a time.

Usage:
  Edit the PAIRS list below to point at your actual files, then:
  python scripts/test_batch_stagewise.py

Output:
  Prints progress to the terminal AND writes a full report to
  scripts/batch_stagewise_report.txt — share that file back for review,
  same way you shared the Word doc, but now covering every stage, not
  just wherever the run happened to stop.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

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

# ── Edit this list to point at your real files ──────────────────────────────
PAIRS = [
    {"label": "Tim Grindland — Wabtec Controllership", "resume": SCRIPT_DIR / "TimGrindlandresume.pdf", "jd": SCRIPT_DIR / "TimGrindlandJD.pdf"},
    {"label": "Divya Pramod — Novo Nordisk Data Steward", "resume": SCRIPT_DIR / "divyaresume.pdf", "jd": SCRIPT_DIR / "divyajd.pdf"},
]


class _Tee:
    """Writes to both the terminal and the report file at once."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _pretty(obj, max_chars: int = 3000) -> str:
    text = json.dumps(obj.model_dump(mode="json"), indent=2, default=str)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n  ... [truncated, {len(text)} total chars]"
    return text


def run_one_pair(label: str, resume_path: Path, jd_path: Path):
    print(f"\n{'#' * 78}")
    print(f"#  {label}")
    print(f"{'#' * 78}")

    if not resume_path.exists() or not jd_path.exists():
        print(f"⚠ SKIPPED — missing file(s): {resume_path} / {jd_path}")
        return

    # Stage 1: Parse
    print(f"\n── STAGE 1: PARSE ──")
    resume_raw = parse_document(resume_path)
    jd_raw = parse_document(jd_path)
    print(f"Resume: {len(resume_raw)} chars parsed from {resume_path.name}")
    print(f"JD:     {len(jd_raw)} chars parsed from {jd_path.name}")

    # Stage 2: Extract
    print(f"\n── STAGE 2: EXTRACT ──")
    resume = extract_resume(resume_raw)
    print(f"ResumeData:\n{_pretty(resume)}")
    jd = extract_jd(jd_raw)
    print(f"\nJDData:\n{_pretty(jd)}")

    # Stage 3: Hard Filter
    print(f"\n── STAGE 3: HARD FILTER ──")
    filter_result = run_hard_filter(resume, jd)
    print(f"FilterResult:\n{_pretty(filter_result)}")

    if not filter_result.passed:
        print(f"\n⚠ REJECTED — stages 4-7 skipped for {label}")
        return

    # Stage 4A + 4B
    print(f"\n── STAGE 4A: SEMANTIC SCORING ──")
    semantic = score_candidate_semantic(resume, jd)
    print(_pretty(semantic))

    print(f"\n── STAGE 4B: LLM SCORING ──")
    llm = score_candidate_llm(resume, jd)
    print(_pretty(llm))

    # Stage 5: Rank
    print(f"\n── STAGE 5: RANK ──")
    ranked = rank_candidates([resume], [semantic], [llm])
    print(_pretty(ranked[0]))

    # Stage 6: Summarise
    print(f"\n── STAGE 6: SUMMARISE ──")
    summary = summarise_candidate(resume, jd, ranked[0])
    print(_pretty(summary))

    print(f"\n✓ {label} — PASSED, overall_score={ranked[0].overall_score}")


def main():
    report_path = SCRIPT_DIR / "batch_stagewise_report.txt"
    with open(report_path, "w", encoding="utf-8") as report_file:
        tee = _Tee(sys.stdout, report_file)
        original_stdout = sys.stdout
        sys.stdout = tee

        print(f"Batch stagewise report — generated {datetime.now().isoformat()}")
        print(f"Provider: {settings.active_llm} | Similarity mode: {settings.similarity_mode}")

        for pair in PAIRS:
            run_one_pair(pair["label"], pair["resume"], pair["jd"])

        sys.stdout = original_stdout

    print(f"\n\nFull report written to: {report_path}")
    print("Share that file back for review — it has every stage for every pair.")


if __name__ == "__main__":
    main()