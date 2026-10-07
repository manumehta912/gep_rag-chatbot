# World Bank Global Economic Prospects (June 2026) - Advanced Document RAG

Handles a 200-page structured report with hierarchical, section-aware indexing and hybrid retrieval.

## What is different from a baseline RAG
| Stage | Technique |
|---|---|
| Extraction | `pypdf` for correct two-column reading order; cleaning of running headers, hyphenation, ligatures, drop-cap glyphs, chart axis noise |
| Structure | PDF bookmarks (chapters, regions) + font-based heading detection with `pdfplumber` (section / subsection / run-in topic, BOX captions) |
| Chunking | Chunks never cross a section boundary; ~1200 chars, sentence-aligned, 150-char overlap inside a section |
| Metadata | chapter, region, section, subsection, topic, box, PDF page range, report page, figure/table ids |
| Embedding | "breadcrumb + text" is embedded, so each chunk knows where it lives (Gemini `gemini-embedding-001`) |
| Retrieval | Dense (Gemini) + BM25 keyword search fused with Reciprocal Rank Fusion; metadata boosts for explicit "Figure 3.4 / Box 1.1 / Chapter 3 / region" mentions; optional chapter filter |
| Precision mode | Optional Gemini reranking of the top candidates |
| Context management | Neighbour expansion within the same section, de-duplication, document order, size budget, breadcrumb headers, page citations |
| Conversation | Follow-up questions are rewritten into standalone search queries |

## Setup
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # paste your GEMINI_API_KEY
python ingest.py --no-embed                          # optional: inspect chunking only (no API calls), see index/chunks.json
python ingest.py                                     # full build (~740 chunks; throttled, resumable)
uvicorn app:app --reload                             # http://127.0.0.1:8000
```
Embedding on the free tier takes roughly 10-15 minutes. If it stops (quota), just run `python ingest.py` again; it resumes.

## Validate
Put the provided sample questions in `questions.txt` (optionally `question | expected,pdf,pages`), then:
```bash
python evaluate.py            # add --rerank to test precision mode
```

## Known limitations
- Values that exist only inside charts are not in the text layer (chart internals are dropped as noise).
- Pages 185-186 (annex divider/tables) have no extractable text.
- Box text and main text are interleaved on some pages, so a few chunks may carry a box label for neighbouring body text.
