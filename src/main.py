# main.py (CLI entry point)

import os

from src.ingest import ingest_pdf
from src.query import query_rag
from src.vector_store.faiss_store import FAISSStore


DATA_DIR = "data"


if __name__ == "__main__":

    store = FAISSStore()

    # Ingest every pdf

    if store.index_exists():
        store.load()
    else:

        for file in os.listdir(DATA_DIR):

            if file.endswith(".pdf"):

                path = os.path.join(DATA_DIR, file)

                ingest_pdf(path, store)

        store.save()

    history = []

    # interactive loop to ask multiple questions while keeping the conversation history
    while True:

        query = input("\nQuestion (or 'exit'): ").strip()

        if query.lower() == "exit":
            break


        answer, sources = query_rag(query, store, history)
        print(f"\nAnswer:\n{answer}")
        
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


