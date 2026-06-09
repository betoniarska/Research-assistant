# query.py

from src.prompt import ask_llm
from src.services.reranker import Reranker
from collections import defaultdict

reranker = Reranker()

def query_rag(question, store):

    store.load()

    # recall stage (FAISS)
    candidates = store.search_balanced(question, k_per_paper=20)

    # precision stage (cross-encoder)
    reranked = reranker.rerank(question, candidates)

    # final selection from reranked results (deduplication + top-k)

    # balance results across papers to get a more diverse set of sources
    final = select_balanced(reranked, top_k=10, per_paper=5)

    return ask_llm(question, final)




def select_balanced(reranked, top_k=10, per_paper=5):

    """
    Distribute slots evenly across papers.
    per_paper: max chunks allowed per paper.
    Falls back to filling remaining slots if one paper is exhausted.
    """

    buckets = defaultdict(list)
    for r in reranked:
        buckets[r["paper_id"]].append(r)

    final = []
    round_idx = 0

    while len(final) < top_k:
        added_this_round = 0
        for paper_id, chunks in buckets.items():
            if len(final) >= top_k:
                break
            if round_idx < len(chunks) and round_idx < per_paper:
                final.append(chunks[round_idx])
                added_this_round += 1
        if added_this_round == 0:
            break  # all buckets exhausted
        round_idx += 1

    return final