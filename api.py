# FastAPI backend entrypoint

import os
import hashlib
import shutil
 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from src.ingest import ingest_pdf
from src.query import query_rag
from src.vector_store.faiss_store import FAISSStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
DATA_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

store = FAISSStore()

if store.index_exists():
    store.load()
    print("Loaded existing index.")
else:
    print("No index found. Ingest PDFs to get started.")

class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]
 
class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
 
class PaperMeta(BaseModel):
    paper_id: str
    paper_title: str
    chunk_count: int


# Endpoints

@app.post("/ingest", response_model=PaperMeta)
async def ingest(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest and index
    ingest_pdf(file_path, store)
    store.save()

    # Get paper metadata for response
    paper_id = hashlib.md5(file_path.encode()).hexdigest()
    paper_title = os.path.splitext(file.filename)[0]
    chunk_count = sum(1 for c in store.chunks if c["paper_id"] == paper_id)

    title = next(c["paper_title"] for c in store.chunks if c["paper_id"] == paper_id)

    return PaperMeta(paper_id=paper_id, paper_title=paper_title, chunk_count=chunk_count)

@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):

    if not store.index_exists():
        raise HTTPException(status_code=400, detail="No papers indexed yet. Upload a PDF first.")
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    

    # the query 

    try:
        answer, sources = query_rag(body.question, store, body.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    

    clean_sources = [
        {
            "paper_title": s["paper_title"],
            "section": s["section"],
            "score": round(s["score"], 3),
            "rerank_score": round(s["rerank_score"], 3),
        }
        for s in sources
    ]
 
    return QueryResponse(answer=answer, sources=clean_sources)


@app.get("/papers", response_model=list[PaperMeta])
async def list_papers():
    """
    Return metadata for all currently indexed papers.
    """
    if not store.chunks:
        return []
 
    seen = {}
    for chunk in store.chunks:
        pid = chunk["paper_id"]
        if pid not in seen:
            seen[pid] = {"paper_title": chunk["paper_title"], "count": 0}
        seen[pid]["count"] += 1
 
    return [
        PaperMeta(paper_id=pid, paper_title=meta["paper_title"], chunk_count=meta["count"])
        for pid, meta in seen.items()
    ]