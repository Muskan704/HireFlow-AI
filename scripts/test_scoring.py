import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
# Add project root to Python path
sys.path.insert(0, str(SCRIPT_DIR.parent))

from app.services.parser import parse_document
from app.services.extractor import extract_resume, extract_jd
from app.services.hard_filter import run_hard_filter
from app.services.scorer_semantic import score_candidate_semantic
from app.services.scorer_llm import score_candidate_llm

# Use absolute paths relative to the script location
resume_text = parse_document(SCRIPT_DIR / "Resume.pdf")
jd_text = parse_document(SCRIPT_DIR / "JD.pdf")

resume = extract_resume(resume_text)
print("RESUME SKILLS:", resume.skills)
print("RESUME NORMALISED:", resume.normalised_skills)
jd = extract_jd(jd_text)
print("JD MUST-HAVE NORMALISED:", jd.normalised_must_have_skills)

result = run_hard_filter(resume, jd)
print(result.passed, result.reject_reasons)

from app.services.ranker import rank_candidates

semantic_scores = [score_candidate_semantic(resume, jd)]
llm_scores = [score_candidate_llm(resume, jd)]

ranked = rank_candidates([resume], semantic_scores, llm_scores)
for c in ranked:
    print(c.rank, c.candidate_name, c.overall_score, c.section_scores)


from app.services.summariser import summarise_candidate

summary = summarise_candidate(resume, jd, ranked[0])
print("STRENGTHS:", summary.strengths)
print("GAPS:", summary.gaps)
print("RECOMMENDATION:", summary.recommendation)