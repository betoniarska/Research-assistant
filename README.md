RAG Research Assistant

A local research assistant that lets you query academic papers in natural language. Upload PDFs, ask questions, and get answers with reference to
the source material.

How it works

Ingestion
PDFs are parsed with GROBID (running as a Docker service) which extracts structured sections from academic papers. Sections are chunked with overlap, 
embedded, and stored in a FAISS vector index alongside metadata (paper title, section, chunk ID).

Ingestion runs once at startup. Already-indexed papers are skipped.

Retrieval
Queries go through two stages:

Recall: FAISS cosine similarity search, retrieving candidates per paper independently to ensure balanced representation across sources
Precision: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) scores each candidate against the query for relevance

Final context is assembled with a per-paper slot limit so no single paper dominates the answer.

Generation
Retrieved chunks are passed to an LLM with a prompt that instructs it to distinguish between sources and stay grounded in context. 
Conversation history is maintained across turns so follow-up questions work naturally.



Roadmap (currently)

 FastAPI backend with /ingest, /query, /papers endpoints
 React frontend with chat interface and source panel
 User PDF uploads
 Docker Compose setup (GROBID + backend + frontend)
 VPS deployment with nginx and persistent volumes (FAISS)
 Query rewriting with history for better retrieval on follow-up questions (?)
