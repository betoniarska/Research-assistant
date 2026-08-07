# api.py

import os
import hashlib
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.ingest import ingest_pdf
from src.query import query_rag
from src.synthesize import run_synthesis
from src.vector_store.faiss_store import FAISSStore
from src.schemas import (
    QueryRequest, QueryResponse,
    PaperMeta,
    SynthesizeRequest, SynthesizeResponse,
)


# ── app setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

store = FAISSStore()

if store.index_exists():
    store.load()
    print("Loaded existing index.")
else:
    print("No index found. Ingest PDFs to get started.")


# ── /ingest ──────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=PaperMeta)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    paper_id = hashlib.md5(save_path.encode()).hexdigest()

    if store.chunks:
        existing_ids = {c["paper_id"] for c in store.chunks}
        if paper_id in existing_ids:
            raise HTTPException(status_code=409, detail=f"'{file.filename}' is already indexed.")

    try:
        ingest_pdf(save_path, store)
        store.save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    chunk_count = sum(1 for c in store.chunks if c["paper_id"] == paper_id)
    title = next(c["paper_title"] for c in store.chunks if c["paper_id"] == paper_id)
    year = next((c["publication_year"] for c in store.chunks if c["paper_id"] == paper_id), None)

    return PaperMeta(paper_id=paper_id, paper_title=title, publication_year=year, chunk_count=chunk_count)


# ── /query ───────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    if not store.index_exists():
        raise HTTPException(status_code=400, detail="No papers indexed yet. Upload a PDF first.")

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer, sources = query_rag(body.question, store, body.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    seen_sources = set()
    clean_sources = []
    for s in sources:
        key = (s["paper_title"], s["section"])
        if key in seen_sources:
            continue
        seen_sources.add(key)
        clean_sources.append({
            "paper_title": s["paper_title"],
            "section": s["section"],
            "score": round(s["score"], 3),
            "rerank_score": round(s["rerank_score"], 3),
        })

    return QueryResponse(answer=answer, sources=clean_sources)


# ── /papers ──────────────────────────────────────────────────────────────────

@app.get("/papers", response_model=list[PaperMeta])
async def list_papers():
    if not store.chunks:
        return []

    seen = {}
    for chunk in store.chunks:
        pid = chunk["paper_id"]
        if pid not in seen:
            seen[pid] = {
                "paper_title": chunk["paper_title"],
                "publication_year": chunk.get("publication_year"),
                "count": 0,
            }
        seen[pid]["count"] += 1

    return [
        PaperMeta(
            paper_id=pid,
            paper_title=meta["paper_title"],
            publication_year=meta["publication_year"],
            chunk_count=meta["count"],
        )
        for pid, meta in seen.items()
    ]


# ── /synthesize ──────────────────────────────────────────────────────────────

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(body: SynthesizeRequest):
    if not store.index_exists():
        raise HTTPException(status_code=400, detail="No papers indexed yet. Upload PDFs first.")

    if not body.claim.strip():
        raise HTTPException(status_code=400, detail="Claim cannot be empty.")

    try:
        synthesis, stance_counts, per_paper, chronological = await run_synthesis(
            body.claim, store, body.paper_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    return SynthesizeResponse(
        claim=body.claim,
        synthesis=synthesis,
        stance_counts=stance_counts,
        per_paper=per_paper,
        chronological=chronological,
    )