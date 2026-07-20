# Recruitment Intelligence Platform — Pre-Call Setup Module

A backend-first pipeline that takes a bulk set of resumes and a single job description and produces **ranked, scored, and summarized candidate outputs** for recruiters — cutting down manual resume screening before a recruiter's first call with a candidate.

This README tracks what's built, what's still to come, and *why* each design decision was made — so the architecture stays legible as the project grows.

---

## 1. Architecture Overview

```
                         ┌─────────────────────┐
                         │   Bulk Resumes (PDF/ │
                         │   DOCX) + Job Desc   │
                         └──────────┬──────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │ Stage 1 — Document Parsing     │  ✅ DONE
                    │ PyMuPDF + OCR fallback / docx  │
                    └──────────────┬─────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Stage 2 — LLM Extraction       │  ✅ DONE
                    │ Raw text → structured          │
                    │ ResumeData / JDData             │
                    └──────────────┬─────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Skill Normalization Layer      │  ✅ DONE
                    │ Alias table + parenthetical    │
                    │ abbreviation extraction         │
                    └──────────────┬─────────────────┘
                                   ▼
                    ┌───────────────────────────────┐
                    │ Stage 3 — Hard Filter          │  ✅ DONE
                    │ Deterministic knockout gate    │
                    └──────────────┬─────────────────┘
                          pass │        │ reject
                               ▼        ▼
              ┌─────────────────────┐  (recruiter sees
              │ Stage 4A — Semantic │   rejection reason,
              │ Scoring (embeddings)│   candidate excluded
              └──────────┬───────────┘  from ranking)
                        ▼
              ┌─────────────────────┐
              │ Stage 4B — LLM      │   ✅ DONE
              │ Section Scoring      │
              └──────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │ Stage 5 — Ranker    │   ✅ DONE
              │ Weighted formula     │
              └──────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │ Stage 6 — Summariser│   ✅ DONE
              │ Recruiter-facing fit │
              │ summary               │
              └──────────┬───────────┘
                        ▼
        ┌─────────────────────────────────────┐
        │  Pipeline Orchestration              │  ✅ DONE
        │  pipeline.py + API endpoint          │
        └─────────────────────────────────────┘
                        ▼
        ┌─────────────────────────────────────┐
        │  DB Persistence Layer                │  🔧 DESIGNED
        │  (db_models.py schema ready,         │   NOT WIRED
        │   needs connection to pipeline)      │
        └─────────────────────────────────────┘

        FUTURE ENHANCEMENTS (deferred):
        • Knowledge Brief Engine (JD summary + resume digest)
        • Q&A Generation Engine (interview questions, red flags)
```

**Design principle carried through every stage:** each stage returns a consistent, well-defined output shape so the next stage doesn't need to know *how* the previous one produced it. This is why Stage 4A (embeddings) and Stage 4B (LLM scoring) are designed to return the same `CandidateScores` shape even though their internal methods are completely different — the Ranker (Stage 5) stays agnostic to which scoring path fed it.

---

## 2. What's Built So Far

### Stage 1 — Document Parsing ✅
Extracts raw text from uploaded resumes and job descriptions.
- **PyMuPDF** for PDF text extraction, with an **OCR fallback** for scanned/image-based PDFs
- **python-docx** for `.docx` files
- **Why:** Resumes arrive in inconsistent formats (text-based PDFs, scanned PDFs, Word docs). A single parser can't handle all of them reliably, so parsing is format-aware from the start rather than assuming clean text input.

### Stage 2 — LLM Extraction ✅
Converts raw, unstructured resume/JD text into structured data.
- **Groq (`llama-3.3-70b-versatile`)** as the LLM provider
- Structured output types: `ResumeData` and `JDData` (defined in `schemas.py`)
- **Why Groq:** fast inference speed matters for a bulk-processing pipeline where dozens of resumes may need extraction in one batch; using a hosted, fast-inference LLM keeps per-resume latency low without needing to self-host a model.
- **Why structured extraction over keyword scraping:** resumes are free text with huge structural variance (headers, bullet styles, date formats). An LLM can normalize this into a consistent schema in one pass; regex/keyword scraping cannot generalize across resume formats.

### Skill Normalization Layer ✅
A dedicated module (`skill_aliases.py`) that canonicalizes skill strings before any comparison happens.
- **Alias table:** hand-curated, deterministic mappings for known synonyms (e.g. `AWS` / `Amazon Web Services`, `K8s` / `Kubernetes`, `JS` / `JavaScript`)
- **Parenthetical abbreviation extraction:** a general regex-based fix for the very common resume pattern `"Full Term (ABBR)"` (e.g. `"Object-Oriented Programming (OOP)"` → also matchable as `"oop"`)
- **Why deterministic, not fuzzy/LLM-based:** semantic or fuzzy matching belongs in Stage 4 (embeddings), which is designed to handle "close but not identical" similarity. Stage 3 is a hard knockout gate — its logic must stay fully deterministic and auditable.

### Stage 3 — Hard Filter ✅
A deterministic knockout gate that rejects clearly unqualified candidates before any expensive scoring happens.
- **Checks:** minimum experience, must-have skills, must-have skill *groups* (OR-logic sets), required certifications, optional strict location matching
- **Why hard-reject before scoring:** running semantic/LLM scoring (Stage 4) on every resume is comparatively expensive. Filtering out clear non-matches first saves cost and keeps the ranked output focused.

### Stage 4A — Semantic Scoring ✅
Embedding-based similarity scoring using local model.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (free, no API calls)
- **Computes 9 component scores:** must_have_skills, good_to_have_skills, domain_knowledge, certifications, total_experience, role_match, project_relevance, semantic_boost, summary_score
- **Why local embeddings:** no external API cost, deterministic, fast for bulk processing

### Stage 4B — LLM Scoring ✅
LLM-based section scoring with structured output.
- **Temperature=0.0** for determinism
- **Structured response schema:** `LLMScoringResponse` with 8 score fields
- **Fallback:** neutral scores (0.5) on failure with error logging

### Stage 5 — Ranker ✅
Weighted ranking using README-specified formula.
- **Scoring formula:**
  ```
  final_score = (
    0.40 × must_have_skills +
    0.10 × good_to_have_skills +
    0.10 × domain_knowledge +
    0.15 × total_experience +
    0.05 × role_match +
    0.08 × semantic_boost +
    0.05 × certifications +
    0.05 × project_relevance +
    0.02 × summary_score
  )
  ```
- **Deterministic tiebreaker:** higher must_have_skills score wins
- **Returns:** detailed `section_scores` breakdown for transparency

### Stage 6 — Summariser ✅
Grounded LLM-generated recruiter summaries.
- **Inputs:** ResumeData, JDData, RankedCandidate (not free-form LLM invention)
- **Output:** strengths, gaps, recommendation
- **Grounding constraint:** Every claim traceable to structured data

### Pipeline Orchestration ✅
`pipeline.py` wires all stages into one callable function.
- **Input:** list of (resume_bytes, filename) + (jd_bytes, filename)
- **Output:** `PipelineResult` with ranked_candidates, rejected_candidates
- **API:** `POST /pipeline/run` endpoint in `main.py`
- **Error isolation:** one bad resume doesn't crash the batch

---

## 3. Tech Stack Summary

| Component | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF + OCR fallback | Handles both text-based and scanned PDFs |
| DOCX parsing | python-docx | Standard, reliable `.docx` extraction |
| LLM extraction | Groq / `llama-3.3-70b-versatile` | Fast inference for bulk processing |
| Data validation | Pydantic v2 schemas | Enforces consistent structured shape |
| Semantic scoring | `sentence-transformers/all-MiniLM-L6-v2` | Free, local, no API limits |
| Vector store | ChromaDB | Embedded, zero-infra |
| Logging | `loguru` | Structured, readable pipeline logs |
| API layer | FastAPI | Async-friendly, auto-generated docs |

---

## 4. API Endpoints

### POST /pipeline/run
Process multiple resumes against a job description.

**Request (multipart/form-data):**
- `resumes`: one or more PDF/DOCX files
- `jd`: single PDF/DOCX job description

**Response:**
```json
{
  "jd_title": "Software Engineer – Full Stack Developer (MERN)",
  "total_resumes_processed": 3,
  "total_passed_filter": 2,
  "total_rejected": 1,
  "ranked_candidates": [
    {
      "rank": 1,
      "candidate_name": "Mohit Singh",
      "overall_score": 0.6559,
      "section_scores": {
        "must_have_skills": 0.75,
        "good_to_have_skills": 0.45,
        "domain_knowledge": 0.58,
        ...
      },
      "fit_summary": "Based on the overall score of 0.66..."
    }
  ],
  "rejected_candidates": [
    {
      "candidate_name": "resume2.pdf",
      "passed": false,
      "reject_reasons": ["Missing required skills: python, react"]
    }
  ]
}
```

---

## 5. What's Left to Build

| Component | Status | Priority |
|-----------|--------|----------|
| DB Persistence wiring | Schema ready in `db_models.py`, needs connection | Medium |
| Session-oriented API (`/sessions`) | Not started | Medium |
| Background job processing | Not started | Low (for large batches) |
| Knowledge Brief Engine | Not started | Future enhancement |
| Q&A Generation Engine | Not started | Future enhancement |
| Frontend | Deferred by design | Future |

---

## 6. Testing

### Automated Tests
- `tests/test_stages_4_5_6.py` — 17 tests for scoring, ranking, summary stages
- `scripts/test_stage12.py` — Stage 1 & 2 parsing/extraction
- `scripts/test_stage3.py` — Hard filter tests
- `scripts/test_scoring.py` — Full pipeline smoke test

### Run Tests
```bash
python -m pytest tests/test_stages_4_5_6.py -v
```

---

## 7. Design Principles

- **Backend-first, frontend deferred** — validate the pipeline logic before UI
- **Each stage owns a single transformation** with consistent output shapes
- **Deterministic gates stay deterministic** — Stage 3 uses exact matching only
- **Test against real data** — skill-normalization bug caught via real JD/resume pair
- **Judgment calls documented** — policy decisions explicitly commented