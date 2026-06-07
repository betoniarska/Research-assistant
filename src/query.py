# query.py

from src.prompt import ask_llm
from src.services.reranker import Reranker

reranker = Reranker()

def query_rag(question, store):

    store.load()

    # recall stage (FAISS)
    candidates = store.search(question, k=30)

    # precision stage (cross-encoder)
    reranked = reranker.rerank(question, candidates)

    # final selection from reranked results (deduplication + top-k)

    seen_papers = set()
    final = []

    for r in reranked:
        if len(final) >= 10:
            break
        if r["paper_id"] in seen_papers:
            continue
        seen_papers.add(r["paper_id"])
        final.append(r)

    return ask_llm(question, final)