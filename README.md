# Recruitment Intelligence Platform

AI-assisted resume screening backend for ranking multiple candidates against one job description.

The public API is business-focused: a recruiter or consuming service uploads one JD and one or more resumes, then receives rejected candidates, ranked candidates, scoring details, and interview knowledge briefs. Internal stage APIs for raw resume/JD extraction and old history endpoints have been removed from Swagger.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI | HTTP API, Swagger UI, multipart uploads |
| Server | Uvicorn | Local/dev ASGI server |
| Validation/models | Pydantic v2 | Structured request/response and LLM output models |
| Settings | pydantic-settings + `.env` | Runtime configuration for LLMs, embeddings, database, logging |
| LLM structured output | Instructor | Forces LLM responses into Pydantic schemas |
| Main LLM provider | Groq | Default extraction, scoring, summaries, briefs |
| Optional LLM providers | OpenAI, Anthropic, Gemini, Ollama | Pluggable alternatives through the provider factory |
| Retry handling | Tenacity | Retries failed or weak LLM extraction calls |
| Document parsing | PyMuPDF, python-docx | PDF and DOCX text extraction |
| OCR fallback | pytesseract + Pillow | Fallback for scanned/low-text PDFs |
| Local embeddings | sentence-transformers | Semantic similarity without using LLM tokens |
| ML utilities | scikit-learn, numpy | Cosine similarity and score calculations |
| Persistence | SQLModel, SQLAlchemy | Session, JD, resume, score, summary, and brief storage |
| Database driver | psycopg2-binary | PostgreSQL connection when `DATABASE_URL` is configured |
| Logging | loguru | Console and rotating file logs |
| Tests | pytest | Unit and stage-level testing |

Default runtime choices:

```env
ACTIVE_LLM=groq
GROQ_MODEL=llama-3.3-70b-versatile
SIMILARITY_MODE=both
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Public API

Run the app:

```bash
uvicorn app.api.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

Current exposed endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/candidate-ranking` | Full JD/resume ranking pipeline |
| GET | `/candidate-ranking/{session_id}` | Retrieve a saved ranking result |
| GET | `/candidate-ranking/{session_id}/candidate/{resume_id}` | Retrieve one candidate's saved detail |
| POST | `/pipeline/run` | Deprecated compatibility endpoint |

Removed public endpoints:

```text
/resume/extract
/resume/extract-text
/jd/extract
/jd/extract-text
/companies/{company}/sessions
/sessions/{session_id}/candidates
/resumes/{resume_id}
```

Those capabilities still exist internally as pipeline stages, but they are no longer exposed as recruiter-facing APIs.

## Request Format

Endpoint:

```http
POST /candidate-ranking
```

Content type:

```text
multipart/form-data
```

Fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `resumes` | file array | yes | One or more resume files, PDF/DOCX |
| `jd` | file | no | JD file, PDF/DOCX/TXT |
| `jd_text` | text | no | Raw pasted JD text |

Use exactly one JD source:

```text
Send either jd or jd_text.
Do not send both.
```

The Swagger form shows `resumes` before `jd` because FastAPI/Pydantic requires required fields before optional fields. This does not affect frontend integration.

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

Windows PowerShell:

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

If `DATABASE_URL` is empty, the API still processes files, but saved-session retrieval will not work.

### 5. Initialize Database

Only needed when using persistence:

```bash
python -m app.core.database
```

### 6. Start API

```bash
uvicorn app.api.main:app --reload
```

### 7. Call the API

With JD file:

```bash
curl -X POST "http://127.0.0.1:8000/candidate-ranking" \
  -F "jd=@JD.pdf" \
  -F "resumes=@Resume1.pdf" \
  -F "resumes=@Resume2.pdf"
```

With pasted JD text:

```bash
curl -X POST "http://127.0.0.1:8000/candidate-ranking" \
  -F "jd_text=AI Engineer role requiring Python, LLMs, RAG, vector databases, and cloud deployment." \
  -F "resumes=@Resume1.pdf" \
  -F "resumes=@Resume2.pdf"
```

Windows PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/candidate-ranking" `
  -F "jd=@JD.pdf" `
  -F "resumes=@Resume1.pdf" `
  -F "resumes=@Resume2.pdf"
```

## Project Structure

### Root Files

| File | Purpose |
|---|---|
| `README.md` | Project documentation, setup, architecture, API usage |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables |
| `.env` | Local secrets/config, not meant for GitHub |
| `.gitignore` | Files ignored by Git |
| `weights.json` | Stage 5 ranking weights |
| `recruitment_platform.db` | Local SQLite database file if used |
| `main.py` | Legacy/simple entry file, not the main FastAPI app |
| `app.py` | Legacy/simple entry file, not the main FastAPI app |

### Application Files

| File | Purpose |
|---|---|
| `app/api/main.py` | FastAPI app, CORS, logging, health route, candidate ranking routes, deprecated pipeline route |
| `app/core/config.py` | Loads `.env` settings with cached `get_settings()` |
| `app/core/database.py` | Creates database engine/session and initializes SQLModel tables |
| `app/models/schemas.py` | Core structured models: `ResumeData`, `JDData`, work experience, education, projects, certifications |
| `app/models/results.py` | Public pipeline response models: filter results, scores, ranked candidates, summaries, knowledge briefs |
| `app/models/db_models.py` | SQLModel persistence tables for sessions, JDs, resumes, scores, rankings, summaries, briefs |
| `app/llm/base.py` | Shared interface all LLM providers implement |
| `app/llm/factory.py` | Chooses the active LLM provider from `.env` |
| `app/llm/groq_provider.py` | Groq structured-completion implementation with rate-limit retry handling |
| `app/llm/openai_provider.py` | OpenAI provider implementation |
| `app/llm/anthropic_provider.py` | Anthropic provider implementation |
| `app/llm/gemini_provider.py` | Gemini provider implementation |
| `app/llm/ollama_provider.py` | Local Ollama provider implementation |
| `app/services/candidate_ranking.py` | Business service behind `/candidate-ranking`; validates uploads and calls the pipeline |
| `app/services/pipeline.py` | End-to-end orchestration across parsing, cleaning, extraction, filtering, scoring, ranking, summaries, briefs, persistence |
| `app/services/parser.py` | Extracts raw text from PDF, DOCX, TXT, MD; uses OCR fallback for weak PDFs |
| `app/services/text_cleaner.py` | Removes Outlook/email headers, forwarded-message blocks, disclaimers, signatures, noisy whitespace |
| `app/services/extractor.py` | LLM extraction for JD/resumes plus post-processing guardrails |
| `app/services/skill_aliases.py` | Deterministic skill normalization and aliases used by hard filter |
| `app/services/hard_filter.py` | Deterministic knockout gate for experience, must-have skills, OR skill groups, certs, strict location |
| `app/services/embedder.py` | Local embedding creation and caching helper |
| `app/services/scorer_semantic.py` | Local semantic scoring using embeddings/cosine similarity |
| `app/services/scorer_llm.py` | LLM-based section scoring for passed candidates |
| `app/services/ranker.py` | Loads `weights.json`, blends scores, computes final rank order |
| `app/services/summariser.py` | LLM summaries for passed candidates and rejected candidates |
| `app/services/knowledge_brief.py` | LLM-generated interview/context briefs for top ranked candidates |
| `app/services/session_repo.py` | Saves and reconstructs full pipeline sessions from the database |
| `tests/test_text_cleaner.py` | Unit tests for email/PDF boilerplate cleaning |
| `tests/test_stages_4_5_6.py` | Stage tests for scoring, ranking, and summarization behavior |
| `scripts/*.py` | Manual/local stage test scripts and experiments |
| `scripts/JD_Resume/*` | Local sample JD/resume files for testing |

## End-to-End Flow

```text
POST /candidate-ranking
        |
        v
app/api/main.py
        |
        v
CandidateRankingService
        |
        v
Pipeline Orchestrator
        |
        v
Stage 1: Parse documents
        |
        v
Text Cleaner
        |
        v
Stage 2: LLM extraction
        |
        v
Stage 2b: Deterministic extraction guardrails
        |
        v
Stage 3: Hard filter
        |
        |-- rejected candidates -> rejection summaries
        |
        v
Stage 4A: Semantic scoring
        |
        v
Stage 4B: LLM scoring
        |
        v
Stage 5: Weighted ranking
        |
        v
Stage 6: Candidate summaries
        |
        v
Stage 7: Knowledge briefs for top 3
        |
        v
Persistence
        |
        v
PipelineResult JSON
```

## Stage Details

### 1. API Upload and Validation

`app/api/main.py` receives multipart data and calls `CandidateRankingService`.

`app/services/candidate_ranking.py`:

- checks that at least one resume was uploaded
- accepts JD as either file or text
- rejects requests where both `jd` and `jd_text` are sent
- validates supported file extensions
- enforces `MAX_RESUME_SIZE_MB`
- converts `jd_text` into an internal text source named `job_description.txt`
- calls `run_pipeline_with_persistence_async()`

LLM calls: `0`

### 2. Document Parsing

`app/services/parser.py` converts files into raw text.

Supported input:

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
OCR fallback with pytesseract when the PDF has too little extractable text
```

Output:

```python
raw_text: str
```

LLM calls: `0`

### 3. Text Cleaning

`app/services/text_cleaner.py` runs after parsing and before LLM extraction.

It removes:

```text
Outlook transport text
EXTERNAL email warnings
From/Sent/To/Cc/Bcc/Subject/Date headers
Original Message and Forwarded Message blocks
quoted reply lines
email-only lines
confidentiality disclaimers
extra blank lines
```

Output:

```python
cleaned_text: str
```

LLM calls: `0`

### 4. JD Extraction

`app/services/extractor.py` calls the active LLM and returns `JDData`.

Main output fields:

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
```

LLM calls:

```text
1 per JD
```

Current output budget:

```python
max_tokens=2500
```

### 5. JD Guardrails

After the LLM extracts the JD, deterministic cleanup fixes common false-rejection cases.

Examples:

```text
AWS + Azure + GCP in must_have_skills
    -> moved to must_have_skill_groups=[["AWS", "Azure", "GCP"]]

REST API + GraphQL in must_have_skills
    -> moved to must_have_skill_groups=[["REST API", "GraphQL"]]

TensorFlow/PyTorch alternatives
    -> moved to must_have_skill_groups=[["TensorFlow", "PyTorch"]]

Large Language Models (LLMs)
    -> normalized to LLM

Vector Databases
    -> normalized to vector database
```

This prevents the system from requiring all acceptable alternatives at once.

LLM calls: `0`

### 6. Resume Extraction

`app/services/extractor.py` calls the active LLM once per resume and returns `ResumeData`.

Main output fields:

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
```

LLM calls:

```text
1 per resume
```

Current output budget:

```python
max_tokens=3500
```

Resume extraction retries when the provider fails or when extraction looks suspiciously incomplete.

### 7. Resume Experience Guardrail

The extractor now uses explicit professional-summary experience as a floor.

Example:

```text
"Generative AI Engineer with 3+ years of hands-on experience..."
```

If the LLM sums dated jobs as `28 months`, the guardrail adjusts:

```text
28 months -> 36 months
```

This prevents false rejection when the resume clearly states a total experience value.

LLM calls: `0`

### 8. Skill Normalization

`app/services/skill_aliases.py` normalizes skill variants before hard filtering.

Examples:

```text
Large Language Models (LLMs) -> llm
LLMs -> llm
Vector Databases -> vector database
REST/GraphQL APIs -> rest api
K8s -> kubernetes
ReactJS -> react
Postgres -> postgresql
Gen AI -> generative ai
```

`app/services/hard_filter.py` also supports category matches:

```text
JD says vector database
Resume says Pinecone, Weaviate, FAISS, Chroma, Qdrant, Milvus, or pgvector
Result: match

JD says LLM
Resume says LangChain, RAG, prompt engineering, fine-tuning, transformers, GPT, Llama, OpenAI
Result: match

JD says REST API
Resume says FastAPI, Flask, Django, Spring, Express.js, or NestJS
Result: match
```

LLM calls: `0`

### 9. Hard Filter

`app/services/hard_filter.py` is a deterministic knockout gate.

It checks:

```text
minimum experience
must-have skills
must-have skill groups
required certifications
strict location only when location_strict=True
education presence as informational only
```

Output:

```python
passed_resumes: list[ResumeData]
filter_results: list[FilterResult]
```

Rejected candidates stay in the final response with reasons.

LLM calls for filter logic: `0`

### 10. Rejection Summaries

`app/services/summariser.py` creates a human-readable summary for each rejected candidate that was successfully extracted.

Output added to rejected candidates:

```text
rejection_summary
is_close_miss
```

LLM calls:

```text
1 per hard-filter rejected candidate
```

### 11. Semantic Scoring

`app/services/scorer_semantic.py` runs for passed candidates when:

```env
SIMILARITY_MODE=semantic
```

or:

```env
SIMILARITY_MODE=both
```

It uses local embeddings from `sentence-transformers/all-MiniLM-L6-v2`.

Score sections:

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

Groq tokens used: `0`

### 12. LLM Scoring

`app/services/scorer_llm.py` runs for passed candidates when:

```env
SIMILARITY_MODE=llm
```

or:

```env
SIMILARITY_MODE=both
```

Output:

```python
CandidateScores(method="llm")
```

LLM calls:

```text
1 per passed candidate
```

Current output budget:

```python
max_tokens=4096
```

### 13. Weighted Ranking

`app/services/ranker.py` loads `weights.json` and computes final candidate order.

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

When `SIMILARITY_MODE=both`, semantic and LLM section scores are blended.

LLM calls: `0`

### 14. Candidate Summaries

`app/services/summariser.py` creates strengths, gaps, recommendation, and recruiter notes for ranked candidates.

LLM calls:

```text
1 per ranked candidate
```

Current output budget:

```python
max_tokens=4096
```

### 15. Knowledge Briefs

`app/services/knowledge_brief.py` creates interview/context briefs for the top ranked candidates only.

Limit:

```text
top 3 candidates
```

LLM calls:

```text
1 per top ranked candidate, maximum 3
```

Current output budget:

```python
max_tokens=4096
```

### 16. Persistence

`app/services/session_repo.py` saves a completed run when `DATABASE_URL` is configured.

Saved tables:

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

Retrieve saved data with:

```http
GET /candidate-ranking/{session_id}
GET /candidate-ranking/{session_id}/candidate/{resume_id}
```

## Response Contract

`POST /candidate-ranking` returns `PipelineResult`:

```json
{
  "jd_title": "AI Engineer",
  "total_resumes_processed": 4,
  "total_passed_filter": 2,
  "total_rejected": 2,
  "rejected_candidates": [
    {
      "resume_id": "uuid",
      "candidate_name": "Candidate Name",
      "passed": false,
      "reject_reasons": ["Experience too low: candidate has 24 months, JD requires minimum 36 months"],
      "rejection_summary": "Candidate is below the required experience threshold...",
      "is_close_miss": false,
      "checks": {
        "min_experience": false,
        "must_have_skills": true,
        "must_have_skill_groups": true,
        "required_certifications": true,
        "location_match": true
      }
    }
  ],
  "ranked_candidates": [
    {
      "rank": 1,
      "resume_id": "uuid",
      "candidate_name": "Candidate Name",
      "overall_score": 0.86,
      "section_scores": {
        "must_have_skills": 1.0,
        "good_to_have_skills": 0.8,
        "domain_knowledge": 0.8,
        "certifications": 0.5,
        "total_experience": 1.0,
        "role_match": 0.8,
        "semantic_boost": 0.82,
        "project_relevance": 0.75,
        "summary_score": 0.8
      },
      "weights_used": {
        "must_have_skills": 0.4,
        "good_to_have_skills": 0.1,
        "domain_knowledge": 0.1,
        "certifications": 0.05,
        "total_experience": 0.15,
        "role_match": 0.05,
        "semantic_similarity_boost": 0.08,
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
P = candidates that passed the hard filter
R = extracted candidates rejected by the hard filter
K = min(3, P)
```

With default `SIMILARITY_MODE=both`:

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
total_llm_calls = 1 + N + R + 2P + K
```

Examples:

```text
4 resumes, 2 pass, 2 rejected:
1 + 4 + 2 + 4 + 2 = 13 LLM calls

4 resumes, 4 rejected:
1 + 4 + 4 + 0 + 0 = 9 LLM calls
```

Retries can increase the actual number of calls.

## Token Usage

The app does not currently calculate exact tokens before every request. Approximation:

```text
prompt_tokens ~= characters / 4
request_tokens ~= prompt_tokens + max_tokens
```

Current output reservations:

| Stage | LLM calls | max_tokens |
|---|---:|---:|
| JD extraction | 1 per JD | 2500 |
| Resume extraction | 1 per resume | 3500 |
| LLM scoring | 1 per passed candidate | 4096 |
| Rejection summary | 1 per rejected candidate | 4096 |
| Candidate summary | 1 per ranked candidate | 4096 |
| Knowledge brief | 1 per top candidate, max 3 | 4096 |

Groq rate-limit errors:

```text
413 Request too large
```

The single request is too large for the current Groq tokens-per-minute tier.

```text
429 Rate limit reached
```

The rolling token window is already partially used.

The Groq provider reads messages like `try again in 32.79s`, waits, and retries.

## Async and Rate Limits

`POST /candidate-ranking` uses async orchestration.

Parsing can run concurrently. LLM calls are protected by:

```python
asyncio.Semaphore(1)
```

This keeps Groq LLM calls one at a time to reduce TPM failures. Local semantic scoring does not consume Groq tokens.

## Run Tests

Focused cleaner tests:

```bash
python -m pytest tests/test_text_cleaner.py
```

All tests:

```bash
python -m pytest
```

Some older script-based tests under `scripts/` are manual stage experiments and may not match the latest public API exactly.

## Troubleshooting

### `/` returns 404

Use:

```text
/docs
/health
/candidate-ranking
```

### Swagger does not show file upload for JD

Restart Uvicorn completely. Also make sure the current code has `jd` typed as an optional `UploadFile` with `File(...)`.

### Swagger shows resumes above JD

This is expected. Required fields appear before optional fields. It does not affect frontend upload behavior.

### Swagger still shows Stage 1/2 or History endpoints

Restart Uvicorn fully. The public app should only show health, candidate ranking, and deprecated pipeline routes.

### Groq 413

Cause:

```text
system prompt + document text + reserved max_tokens exceeds the tier limit
```

Fix options:

```text
remove long email chains
clean large PDFs
reduce output token budgets
wait and retry later
upgrade Groq tier
switch provider
```

### Groq 429

Cause:

```text
rolling tokens-per-minute quota is already used
```

Current mitigation:

```text
LLM calls are serialized
Groq retry-after time is parsed and respected
```

### `tenacity.RetryError`

This means a retried operation failed after all attempts. The real cause is usually printed above it in logs, commonly Groq `413`, Groq `429`, invalid structured JSON, or provider failure.

### OCR does not work

For scanned PDFs, install the Tesseract executable on the machine. The Python package alone is not enough.

## Development Notes

Important rules:

```text
Keep public API focused on /candidate-ranking
Do not re-expose Stage 1/2 endpoints
Keep hard filter deterministic
Keep skill aliases auditable and hand-curated
Keep LLM provider access centralized in app/llm/factory.py
Do not casually change ranking weights or formulas
```

Recommended integration endpoints:

```text
POST /candidate-ranking
GET /candidate-ranking/{session_id}
GET /candidate-ranking/{session_id}/candidate/{resume_id}
```
