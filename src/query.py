from src.prompt import ask_llm


def query_rag(question, store):

    store.load()

    results = store.search(question, k=10)

    answer = ask_llm(question, results)

    return answer, results