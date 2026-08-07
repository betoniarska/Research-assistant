# RAG Research Assistant

A research assistant for querying and synthesizing insights across multiple academic papers. Upload PDFs, ask questions in natural language, and get answers grounded in source material with balanced representation across papers enforced at the retrieval level.

**Status: August 2026: backend working (query + synthesize), frontend and loop agent in progress**

---

## Pipeline

```
uploads/                  <- user-uploaded PDFs (via /ingest endpoint)
storage/
  index.faiss             <- FAISS vector index (persisted)
  chunks.json             <- chunk metadata: text, paper_id, section, page, year

src/
  ingest.py               orchestrates PDF → GROBID → chunks → FAISS
  parser.py               GROBID XML parsing, section extraction, chunking, page tracking
  query.py                retrieval pipeline: FAISS recall → reranking → balanced selection → LLM
  synthesize.py           per-paper structured extraction → Python aggregation → narrative synthesis
  schemas.py              shared Pydantic models for all endpoints
  prompt.py               prompt construction + OpenAI call for /query
  loader.py               GROBID HTTP client (PDF → TEI XML)
  services/
    embedding_service.py  sentence-transformers wrapper
    reranker.py           cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
  vector_store/
    faiss_store.py        FAISS index management, persistence, per-paper search

api.py                    FastAPI entry point — /ingest, /query, /papers, /synthesize
```

Each module does one designated job and can be examined independently.

---

## 2 Modes:

### Query (`/query`): traditional RAG, fixed pipeline

Retrieval runs globally across all indexed papers in two stages:

1. **Recall:** FAISS cosine similarity (`IndexFlatIP`, L2-normalized), retrieving candidates *per paper independently* via `search_balanced`. This is the core retrieval design decision: rather than searching the full index and letting one paper crowd out others (which happens naturally when a paper is longer or more topically dominant), each paper contributes its own top-k candidates before any cross-paper ranking.

2. **Reranking:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) scores each candidate against the query for precision. Candidates scoring below a threshold are dropped before final selection, preventing irrelevant papers from filling slots just because balanced retrieval allocated them.

3. **Selection:** slot allocation across papers ensures no single paper dominates the final context, with a configurable per-paper cap.

4. **Generation:** retrieved chunks are passed to an LLM with source attribution (paper, section, scores). Conversation history is maintained across turns so follow-up questions work naturally without re-retrieval.

### Synthesize (`/synthesize`): structured cross-paper analysis

Designed for a different task than `/query`: rather than answering a question, it evaluates a claim across the full corpus and returns structured per-paper stances with aggregated counts.

The key design principle here is that aggregation happens in Python, not in the LLM. The LLM classifies one paper at a time against a fixed schema; counts and sorting are computed from those classifications. This avoids a class of failure common in RAG synthesis where the model is asked to both classify and summarize, producing inconsistent counts or hallucinated figures.

Synthesize is work-in-progress at the moment

**Three-stage structure:**

1. **Per-paper retrieval:** for each paper independently, retrieve the chunks most relevant to the claim (scoped FAISS + rerank). Unlike `/query`, there is no cross-paper competition, every paper gets its own dedicated retrieval pass, and the agent decides when evidence is sufficient.

2. **Structured extraction:** one LLM call (1st of 2 prompts in synthesize) per paper with `temperature=0`, producing:
   ```
   stance: supports | contradicts | mixed | not_addressed
   method_summary: str
   key_finding: str
   reasoning: str
   page_refs: list[str]
   ```

3. **Aggregation + narrative:** stance counts are computed in Python (`Counter`), papers sorted chronologically, then a final LLM call (2nd prompt) writes a narrative paragraph around the computed counts. The model is explicitly told the counts and instructed not to recompute them.

**Known limitation:** `/synthesize` makes N+1 LLM calls for N papers (N extractions + 1 narrative). Run concurrently with `asyncio.gather`, but latency scales with corpus size.

---

## Key design decisions

**Per-paper retrieval isolation:** searching globally and then trying to balance results after the fact doesn't work well when papers vary significantly in length or topical overlap. E.g. BERT chunks will legitimately score higher on transformer-related queries (than Attention Is All You Need) because BERT discusses transformers extensively, while attention introduces the architecture itself. `search_balanced` solves this by running independent FAISS searches per paper, so each contributes equally to the candidate pool regardless of embedding similarity dominance.

**Cross-encoder reranking:** bi-encoder embeddings (used for FAISS recall) are fast but imprecise. The cross-encoder sees the query and chunk together, which catches relevance that embedding similarity misses. Used in both `/query` (global reranking) and `/synthesize` (per-paper reranking).

**Score gating:** reranked chunks scoring below a threshold are excluded from context even if balanced retrieval allocated them a slot. Without this, a paper that is genuinely irrelevant to a query still contributes chunks just to satisfy the balance constraint.

**Python aggregation:**  LLM shouldn't be asked to count or rank across structured outputs it generated itself. Compute counts, sort orders, and aggregations in code --> give the LLM the results to narrate.

**GROBID for parsing:** academic PDFs are structurally complex (multi-column, citations, figures). GROBID's TEI XML output preserves section structure, which is used to attach section-level metadata to every chunk. This makes source attribution ("Introduction, p. 3") possible and improves retrieval by prepending section titles to chunk text.

---

## Known limitations

**Publication year extraction is unreliable:** GROBID extracts the date from PDF metadata, which for arXiv papers apparently is often the most recent revision date, not the original publication year. E.g. "Attention Is All You Need" (2017) was extracted as 2023 from a revised arXiv PDF. This blocks diachronic comparison (tracking how claims evolved over time), which is a planned feature. Planned fix: Semantic Scholar API lookup at ingest time, falling back to GROBID's date or a loop agent functionality to have an LLM pick out the correct publication date (roadmap).

**Page-level citations are unverified:** `<pb/>` (page break) tags in GROBID's TEI XML are not consistently present across PDFs. The `page_refs` field in synthesis output may be empty or incorrect depending on how the source PDF was structured. The LLM has shown a tendency to fabricate section titles as page references when real page data is absent. This is somewhat mitigated by explicit prompt instructions, not fully solved.

**Stance classification degrades on ambiguous claims:** `/synthesize` forces a binary stance per paper. For vague or multi-interpretable claims, the LLM silently picks one reading and classifies confidently against it, which produces plausible-looking but wrong results. Partially addressed by an `ambiguity_note` field in the schema. A loop agent (see Roadmap) would address this more fundamentally by allowing the model to signal uncertainty and search further before committing.

**Title extraction:** GROBID sometimes merges license/attribution boilerplate into the title field when it appears directly above the paper title on the first page. Addressed by stripping known patterns via regex, but new publisher formats will need additions to the strip list.

---

## Roadmap

- [ ] **Loop agent for `/synthesize`:** replace fixed per-paper retrieval with an agent loop where the LLM decides when it has sufficient evidence before classifying stance. Addresses the core weakness of fixed retrieval: if a claim is only partially covered by the top-k chunks, the agent can reformulate its search query rather than classifying on incomplete evidence. Also allows year extraction from paper text rather than metadata.
- [ ] **Semantic Scholar year lookup:** unblocks diachronic comparison
- [ ] **Diachronic comparison:** track how stance on a claim shifts across publication years within the corpus (very interesting comparisons)
- [ ] **Methodological comparison:** dedicated synthesis view using `method_summary` fields already extracted per paper
- [ ] **React frontend:** chat interface, paper sidebar, source attribution panel, synthesis results view
- [ ] **Docker Compose:** GROBID + FastAPI + React + nginx
- [ ] **VPS deployment:** persistent FAISS volumes, nginx reverse proxy

---

## Stack

| Layer | Technology |
|---|---|
| PDF parsing | GROBID (Docker, TEI XML) |
| Embeddings | `sentence-transformers` |
| Vector store | FAISS (`IndexFlatIP`, cosine similarity) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | OpenAI API (`gpt-4.1-mini`) |
| Backend | FastAPI + uvicorn |
| Frontend | React (planned) |
| Hosting | VPS + Docker Compose + nginx (planned) |

Some of these may be subject to change at a later date (atleast the LLM)

---

## Running locally

GROBID must be running as a Docker service before ingestion:

```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.0
```

Start the API:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Interactive API docs available at `http://localhost:8000/docs`.

Ingest PDFs via `POST /ingest` (multipart file upload), then query via `POST /query` or evaluate a claim across the corpus via `POST /synthesize`.
