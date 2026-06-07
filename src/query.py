from src.prompt import ask_llm


def query_rag(question, store):

    store.load()

    #results = store.search(question, k=10)

    results = store.search(question, k=30)

    # group by paper
    papers = {}

    for r in results:
        papers.setdefault(r["paper_id"], []).append(r)

    # take top 5 from each paper
    final = []

    for paper in papers.values():
        final.extend(paper[:5])



    return ask_llm(question, final)