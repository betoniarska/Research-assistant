# query.py

from src.prompt import ask_llm


def query_rag(question, store):

    # Load index and metadata, then search for relevant chunks

    store.load()

    results = store.search(question, k=30)

    # Filter results by a similarity threshold and limit to top 10
    final = [r for r in results if r["score"] > 0.13][:10]

    return ask_llm(question, final)
