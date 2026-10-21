# StudyMate

Local PDF study assistant with persistent SQLite storage and optional Gemini answers.

## Start (Python 3.11+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env to add your own Google API key; optional for PDF retrieval.
.\.venv\Scripts\python -m streamlit run app.py --server.address 127.0.0.1
```

For semantic embeddings, install `requirements-semantic.txt`, then enable Semantic search. The first use downloads `all-MiniLM-L6-v2`; embeddings persist in SQLite and indexes stay cached in memory. Without the optional model, or if loading fails, keyword search remains available. No API key is required for offline retrieval. Tesseract must be installed separately for scanned pages; missing OCR produces explicit page warnings.

## Implemented

- Multiple PDFs, all pages, page-aware chunks; no silent 5,000-word truncation. Explicit limits: 25 MB per file and 1,000 pages per PDF.
- SHA-256 content cache for PDF extraction, versioned persistent embeddings, bounded in-memory indexes, and one-hour successful-answer cache.
- SQLite `documents`, `vectors`, `messages`, `responses`, and `requests` tables, WAL and atomic rate reservations. Database is created automatically under `data/`; override `STUDYMATE_DATA_DIR` for persistent hosting storage.
- Global eight-call and per-workspace five-call rolling 60-second limits across sessions and restarts on one database. Every attempted provider call consumes a slot. No background retry loops; one bounded retry for transient failures. SDK retries disabled and timeout 20 seconds per call.
- Missing credentials, provider errors, rate limits, and invalid citation labels yield clearly marked retrieved excerpts. General chat reports unavailability rather than fabricating an offline response.
- Relevant passages retain document and page metadata; UI source expanders and source PDF downloads let users check answers. Model citation labels are validated, but semantic correctness is not guaranteed.
- Persistent chat per selected document set; export, clear history, and delete documents. Deleting a document also clears histories and answer caches to remove retained source text.

## Limits and privacy

This is a single-user local installation, not a multi-tenant public service. Every visitor to the same installation shares its library and history. Workspace IDs identify document selections, not authenticated users. Bind to localhost. Public hosting requires authentication, user-scoped ownership, storage quotas, retention policy, and deployment-specific abuse protection. Stored PDFs and chat are plaintext local data; back up and protect the data directory. Relevant excerpts are sent to Google when a key is configured. Never commit `.env` or `data/`.

Relevance thresholds are heuristic, not accuracy scores. Keyword fallback can miss paraphrases. SQLite supports one host; multiple replicas require a shared database/rate limiter. No measured user-impact or accuracy claims are made.

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover persistence, extraction caching, page attribution, full-document chunking, retrieval, no-match behavior, concurrent rate limiting, provider fallback, retries, citation validation, and deletion. They use synthetic material and mocked provider calls, not live Gemini.

## Study modes and verification

Flashcards and Practice quiz generate up to five topic-specific items from retrieved passages when Gemini is configured. Without Gemini, the app shows source excerpts instead. Cited pages can be previewed directly.

Run `python evaluate.py` for the four-case synthetic retrieval smoke check. The implementation passes 16 automated tests, including an offline Streamlit question-and-history flow. Live Gemini responses, optional model downloads, and Tesseract OCR have not been exercised in this environment.
