# Recruitment Intelligence Platform

AI-assisted resume screening service for ranking multiple candidates against a single job description.

The public API is intentionally business-focused: upload one JD and many resumes, receive a ranked candidate report. Internal parsing, extraction, scoring, ranking, summarisation, and knowledge brief stages are hidden behind the `/candidate-ranking` endpoint.

## Current Public API

FastAPI app:

```bash
uvicorn app.api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Exposed endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/candidate-ranking` | Run the full ranking pipeline for one JD and many resumes |
| GET | `/candidate-ranking/{session_id}` | Retrieve a previously saved pipeline result |
| GET | `/candidate-ranking/{session_id}/candidate/{resume_id}` | Retrieve detailed candidate data from a saved session |
| POST | `/pipeline/run` | Deprecated compatibility endpoint; prefer `/candidate-ranking` |

Removed from the public API:

```text
/resume/extract
/resume/extract-text
/jd/extract
/jd/extract-text
/companies/{company}/sessions
/sessions/{session_id}/candidates
/resumes/{resume_id}
```

Those stages still exist internally. They are no longer exposed as recruiter-facing endpoints.

## Architecture Flow

```text
POST /candidate-ranking
        |
        v
CandidateRankingService
        |
        v
Pipeline Orchestrator
        |
        v
Stage 1: Parse PDF/DOCX
        |
        v
Text Cleaner: remove Outlook/email boilerplate
        |
        v
Stage 2: LLM Extraction
        |-- JDData
        |-- ResumeData[]
        |
        v
Stage 3: Hard Filter
        |
        |-- rejected candidates -> rejection summaries
        |
        v
Stage 4A: Semantic Scoring
        |
        v
Stage 4B: LLM Scoring
        |
        v
Stage 5: Weighted Ranking
        |
        v
Stage 6: Candidate Summaries
        |
        v
Stage 7: Knowledge Briefs for top 3
        |
        v
Persistence
        |
        v
PipelineResult
```

## Main Modules

| Area | File |
|---|---|
| FastAPI app and routes | `app/api/main.py` |
| Business service layer | `app/services/candidate_ranking.py` |
| End-to-end orchestration | `app/services/pipeline.py` |
| PDF/DOCX/text parsing | `app/services/parser.py` |
| Email/PDF boilerplate cleanup | `app/services/text_cleaner.py` |
| LLM extraction | `app/services/extractor.py` |
| Deterministic hard filter | `app/services/hard_filter.py` |
| Local embedding scorer | `app/services/scorer_semantic.py` |
| LLM section scorer | `app/services/scorer_llm.py` |
| Weighted ranking | `app/services/ranker.py` |
| Candidate/rejection summaries | `app/services/summariser.py` |
| Knowledge briefs | `app/services/knowledge_brief.py` |
| Persistence repository | `app/services/session_repo.py` |
| SQLModel tables | `app/models/db_models.py` |
| Extraction schemas | `app/models/schemas.py` |
| Pipeline response schemas | `app/models/results.py` |
| LLM providers | `app/llm/` |

## Stage-by-Stage Details

### 1. Upload and Request Validation

Endpoint:

```http
POST /candidate-ranking
```

Request type:

```text
multipart/form-data
```

Fields:

```text
jd: optional JD PDF or DOCX
jd_text: optional plain-text JD
resumes: one or more PDF/DOCX files
```

Provide exactly one JD source:

```text
Use jd for file upload, or jd_text for pasted/raw JD text.
Do not send both jd and jd_text in the same request.
```

`CandidateRankingService` validates file extensions, reads upload bytes, checks max file size, and passes file sources into the pipeline. For pasted JDs, it converts `jd_text` into an internal `.txt` source so the same pipeline path is reused:

```python
jd_source = (jd_bytes, jd_filename)
# or
jd_source = (jd_text.encode("utf-8"), "job_description.txt")
resume_sources = [(resume_bytes, resume_filename), ...]
```

### 2. Stage 1: Document Parsing

Parser:

```python
parse_document(source, filename)
```

Supported formats:

```text
.pdf
.docx
.doc
.txt
.md
```

PDF flow:

```text
PyMuPDF text extraction
OCR fallback with pytesseract if extracted text is suspiciously short
```

Output:

```python
raw_text: str
```

LLM calls: `0`

### 3. Text Cleaner

Cleaner:

```python
clean_document_text(raw_text)
```

This runs immediately after parsing and before LLM extraction.

It removes common email/PDF transport noise:

```text
Outlook
EXTERNAL warnings
From / Sent / To / Cc / Bcc / Subject / Date headers
Original Message blocks
Forwarded Message blocks
quoted reply lines
email-only lines
common confidentiality disclaimers
extra blank lines
```

Output:

```python
cleaned_text: str
```

The pipeline logs cleanup when text length changes:

```text
Cleaned email/PDF boilerplate from 'file.pdf' | chars 12000 -> 7800
```

LLM calls: `0`

### 4. Stage 2: LLM Extraction

Extraction uses the active provider from `.env`:

```env
ACTIVE_LLM=groq
GROQ_MODEL=llama-3.3-70b-versatile
```

Provider selection is centralized in:

```python
app/llm/factory.py
```

Groq structured completion is implemented in:

```python
app/llm/groq_provider.py
```

#### JD Extraction

Function:

```python
extract_jd(cleaned_jd_text)
```

Output model:

```python
JDData
```

Important fields:

```text
job_title
company
location
remote_policy
min_experience_months
must_have_skills
must_have_skill_groups
required_certifications
required_education
nice_to_have_skills
preferred_experience_months
responsibilities
domain
normalised_must_have_skills
```

LLM calls:

```text
1 per uploaded JD
```

Current output token budget:

```python
max_tokens=2500
```

#### Resume Extraction

Function:

```python
extract_resume(cleaned_resume_text)
```

Output model:

```python
ResumeData
```

Important fields:

```text
resume_id
candidate_name
email
phone
location
total_experience_months
skills
programming_languages
frameworks_and_tools
work_experience
education
projects
certifications
summary
normalised_skills
```

LLM calls:

```text
1 per uploaded resume
```

Current output token budget:

```python
max_tokens=3500
```

Resume extraction retries up to 3 times if the call fails or the extracted result looks suspiciously incomplete.

### 5. Stage 3: Hard Filter

Function:

```python
run_hard_filter_bulk(extracted_resumes, jd)
```

Input:

```python
list[ResumeData]
JDData
```

Checks:

```text
minimum experience
must-have skills
must-have skill groups, meaning one-of-many alternatives
required certifications
strict location matching only if location_strict=True
education presence is informational only
```

Output:

```python
passed_resumes: list[ResumeData]
filter_results: list[FilterResult]
```

`FilterResult` fields:

```text
resume_id
candidate_name
passed
reject_reasons
rejection_summary
is_close_miss
checks
```

LLM calls for hard filter logic: `0`

### 6. Rejection Summaries

Rejected candidates with valid extracted resume data are passed to:

```python
summarise_rejection(resume, jd, reject_reasons)
```

Output added to each rejected `FilterResult`:

```text
rejection_summary
is_close_miss
```

LLM calls:

```text
1 per hard-filter rejected candidate
```

Parsing/extraction failures do not get LLM rejection summaries because there is no reliable structured resume to summarize.

### 7. Stage 4A: Semantic Scoring

Function:

```python
score_candidate_semantic(resume, jd)
```

Runs for candidates who passed the hard filter.

Output:

```python
CandidateScores(method="semantic")
```

Section scores:

```text
must_have_skills
good_to_have_skills
domain_knowledge
certifications
total_experience
role_match
project_relevance
semantic_boost
summary_score
```

LLM calls: `0`

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This runs locally and does not use Groq tokens.

### 8. Stage 4B: LLM Scoring

Function:

```python
score_candidate_llm(resume, jd)
```

Runs only when:

```env
SIMILARITY_MODE=llm
```

or:

```env
SIMILARITY_MODE=both
```

Default:

```env
SIMILARITY_MODE=both
```

Output:

```python
CandidateScores(method="llm")
```

LLM scoring fields:

```text
must_have_skills_score
good_to_have_skills_score
domain_knowledge_score
certifications_score
total_experience_score
role_match_score
project_relevance_score
summary_score
skills_reasoning
experience_reasoning
```

LLM calls:

```text
1 per passed candidate
```

Current output token budget:

```python
default max_tokens=4096
```

If LLM scoring fails, the code returns a 0.0 fallback score for that candidate's LLM scoring path and logs the failure.

### 9. Stage 5: Weighted Ranking

Function:

```python
rank_candidates(passed_resumes, semantic_scores, llm_scores)
```

Weights are loaded from:

```text
weights.json
```

Current weights:

```text
must_have_skills: 0.40
good_to_have_skills: 0.10
domain_knowledge: 0.10
certifications: 0.05
total_experience: 0.15
role_match: 0.05
semantic_similarity_boost: 0.08
project_relevance: 0.05
summary_score: 0.02
blend_ratio: 0.5
```

Formula:

```text
final_score =
  0.40 * must_have_skills
+ 0.10 * good_to_have_skills
+ 0.10 * domain_knowledge
+ 0.05 * certifications
+ 0.15 * total_experience
+ 0.05 * role_match
+ 0.08 * semantic_boost
+ 0.05 * project_relevance
+ 0.02 * summary_score
```

When `SIMILARITY_MODE=both`, section scores are blended:

```text
final_section_score = 0.5 * llm_score + 0.5 * semantic_score
```

Output:

```python
list[RankedCandidate]
```

Fields:

```text
rank
resume_id
candidate_name
overall_score
section_scores
weights_used
fit_summary
```

LLM calls: `0`

### 10. Stage 6: Candidate Summaries

Function:

```python
summarise_candidate(resume, jd, ranked)
```

Runs for every ranked candidate.

Output:

```python
CandidateSummary
```

Fields:

```text
resume_id
candidate_name
strengths
gaps
recommendation
agency_notes
```

The candidate's ranking object is then enriched:

```python
ranked.fit_summary = summary.recommendation
```

LLM calls:

```text
1 per ranked candidate
```

Current output token budget:

```python
default max_tokens=4096
```

### 11. Stage 7: Knowledge Briefs

Function:

```python
generate_briefs_bulk(passed_resumes, jd, ranked_candidates, top_n=3)
```

Runs only for top 3 ranked candidates.

Output:

```python
KnowledgeBrief
```

Fields:

```text
resume_id
candidate_name
role_overview
team_context
career_summary
key_achievements
technical_strengths
alignment_highlights
experience_relevance
areas_to_probe
suggested_talking_points
years_of_experience
current_or_last_role
education_highlight
certifications
location
```

LLM calls:

```text
1 per top ranked candidate, maximum 3
```

Current output token budget:

```python
default max_tokens=4096
```

### 12. Persistence

Function:

```python
save_full_session()
```

Runs after the pipeline has completed.

Tables:

```text
Session
JobDescriptionRecord
ResumeRecord
FilterResultRecord
CandidateScoreRecord
RankedCandidateRecord
CandidateSummaryRecord
KnowledgeBriefRecord
```

If `DATABASE_URL` is not configured, the pipeline still returns results but does not save session history.

Retrieval:

```http
GET /candidate-ranking/{session_id}
GET /candidate-ranking/{session_id}/candidate/{resume_id}
```

These reconstruct public response models from saved records.

## Final Response Contract

`POST /candidate-ranking` returns:

```python
PipelineResult
```

Example:

```json
{
  "jd_title": "Senior Accountant",
  "total_resumes_processed": 3,
  "total_passed_filter": 2,
  "total_rejected": 1,
  "rejected_candidates": [
    {
      "resume_id": "uuid",
      "candidate_name": "Candidate Name",
      "passed": false,
      "reject_reasons": ["Missing required skills: Excel"],
      "rejection_summary": "Candidate was rejected because...",
      "is_close_miss": true,
      "checks": {
        "min_experience": true,
        "must_have_skills": false
      }
    }
  ],
  "ranked_candidates": [
    {
      "rank": 1,
      "resume_id": "uuid",
      "candidate_name": "Candidate Name",
      "overall_score": 0.82,
      "section_scores": {
        "must_have_skills": 1.0,
        "good_to_have_skills": 0.7,
        "domain_knowledge": 0.8,
        "certifications": 1.0,
        "total_experience": 0.9,
        "role_match": 0.75,
        "semantic_boost": 0.8,
        "project_relevance": 0.5,
        "summary_score": 0.7
      },
      "weights_used": {
        "must_have_skills": 0.4,
        "good_to_have_skills": 0.1,
        "domain_knowledge": 0.1,
        "total_experience": 0.15,
        "role_match": 0.05,
        "semantic_similarity_boost": 0.08,
        "certifications": 0.05,
        "project_relevance": 0.05,
        "summary_score": 0.02,
        "blend_ratio": 0.5
      },
      "fit_summary": "Recommended for recruiter review..."
    }
  ],
  "knowledge_briefs": [],
  "similarity_mode": "both",
  "session_id": "uuid"
}
```

## LLM Call Count

Let:

```text
N = uploaded resumes
E = successfully extracted resumes
P = candidates that passed the hard filter
R = extracted candidates rejected by hard filter
K = min(3, P)
```

With default:

```env
SIMILARITY_MODE=both
```

Base LLM calls:

```text
1 JD extraction
+ N resume extractions
+ R rejection summaries
+ P LLM scoring calls
+ P candidate summaries
+ K knowledge briefs
```

Formula:

```text
total_llm_calls = 1 + N + R + P + P + K
total_llm_calls = 1 + N + R + 2P + K
```

Example where 3 resumes all pass:

```text
N=3, R=0, P=3, K=3
total = 1 + 3 + 0 + 6 + 3 = 13 LLM calls
```

Example where 3 resumes are uploaded, 2 pass and 1 is rejected:

```text
N=3, R=1, P=2, K=2
total = 1 + 3 + 1 + 4 + 2 = 11 LLM calls
```

Retries can increase actual calls.

## Token Usage

The application does not currently calculate exact token usage before each request. Approximate token usage is:

```text
prompt_tokens ~= characters / 4
request_tokens ~= prompt_tokens + max_tokens
```

Current output token reservations:

| Stage | LLM calls | max_tokens |
|---|---:|---:|
| JD extraction | 1 per JD | 2500 |
| Resume extraction | 1 per resume | 3500 |
| LLM scoring | 1 per passed candidate | 4096 |
| Rejection summary | 1 per hard-filter reject | 4096 |
| Candidate summary | 1 per ranked candidate | 4096 |
| Knowledge brief | 1 per top candidate, max 3 | 4096 |

Groq calculates limits roughly against:

```text
system prompt + user prompt + reserved output tokens
```

Common errors:

```text
413 Request too large
```

The single request is too large for the current Groq TPM tier.

```text
429 Rate limit reached
```

The rolling tokens-per-minute window is already partially used. The Groq provider now reads messages like `try again in 32.79s`, waits, and retries.

## Async and Rate Limit Behavior

`POST /candidate-ranking` uses async orchestration.

Resume parsing can run concurrently, but LLM calls are protected by:

```python
asyncio.Semaphore(1)
```

This means Groq-backed LLM calls run one at a time to reduce TPM failures.

Local semantic scoring can still run concurrently because it uses local embeddings, not Groq.

## Setup

### 1. Clone

```bash
git clone <your-repo-url>
cd Recruitment-platform
```

### 2. Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Minimum Groq setup:

```env
ACTIVE_LLM=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
SIMILARITY_MODE=both
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
MAX_RESUME_SIZE_MB=10
```

Optional database setup:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

If no `DATABASE_URL` is set, processing still works, but retrieval by `session_id` will not work because results are not saved.

### 5. Initialize Database

Only needed if using persistence:

```bash
python -m app.core.database
```

### 6. Run FastAPI

```bash
uvicorn app.api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 7. Run Candidate Ranking

Use Swagger UI:

1. Open `/docs`
2. Expand `POST /candidate-ranking`
3. Upload one JD PDF/DOCX or paste JD content into `jd_text`
4. Upload one or more resume PDF/DOCX files
5. Execute

Or use curl:

```bash
curl -X POST "http://127.0.0.1:8000/candidate-ranking" \
  -F "jd=@JD.pdf" \
  -F "resumes=@Resume1.pdf" \
  -F "resumes=@Resume2.pdf"
```

With pasted JD text:

```bash
curl -X POST "http://127.0.0.1:8000/candidate-ranking" \
  -F "jd_text=Senior Accountant role requiring month-end close, reconciliations, Excel, and ERP experience." \
  -F "resumes=@Resume1.pdf" \
  -F "resumes=@Resume2.pdf"
```

Windows PowerShell alternative:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/candidate-ranking" `
  -F "jd=@JD.pdf" `
  -F "resumes=@Resume1.pdf" `
  -F "resumes=@Resume2.pdf"
```

Windows PowerShell with pasted JD text:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/candidate-ranking" `
  -F "jd_text=Senior Accountant role requiring month-end close, reconciliations, Excel, and ERP experience." `
  -F "resumes=@Resume1.pdf" `
  -F "resumes=@Resume2.pdf"
```

## Running Tests

Run the focused text cleaner test:

```bash
python -m pytest tests/test_text_cleaner.py
```

Run all tests:

```bash
python -m pytest
```

Some older tests may need updates if they assert old stage internals or old score shapes.

## Troubleshooting

### `/` returns 404

This is normal. Use:

```text
/docs
/health
/candidate-ranking
```

### Swagger still shows old endpoints

Restart Uvicorn completely. With `--reload`, stale child processes can sometimes hold the old app briefly.

### Groq 413: Request too large

Cause:

```text
prompt tokens + max_tokens > Groq TPM tier limit
```

Fixes:

```text
Use cleaner JD/resume files
Remove email chains and signatures
Reduce max_tokens
Use a higher Groq tier
Use another provider with larger limits
```

### Groq 429: Rate limit reached

Cause:

```text
Too many tokens used in the rolling minute window
```

Current mitigation:

```text
LLM calls are throttled one at a time
Groq provider waits and retries when Groq gives "try again in Xs"
```

### `tenacity.RetryError`

This means a retried LLM operation failed after all retry attempts. Look above the `RetryError` in logs for the real cause, usually a Groq `413`, `429`, invalid JSON, or provider failure.

### OCR does not work

For scanned PDFs, `pytesseract` requires the Tesseract executable to be installed on the machine, not just the Python package.

## Development Notes

Important constraints:

```text
Do not change ranking formulas casually
Do not expose internal stage endpoints
Keep PipelineResult stable
Keep hard filter deterministic
Keep LLM provider access centralized through app/llm/factory.py
```

Recommended API surface for other services:

```text
POST /candidate-ranking
GET /candidate-ranking/{session_id}
GET /candidate-ranking/{session_id}/candidate/{resume_id}
```
