import os

from src.ingest import ingest_pdf
from src.query import query_rag
from src.vector_store.faiss_store import FAISSStore


DATA_DIR = "data"


if __name__ == "__main__":

    store = FAISSStore()

    # Ingest every pdf

    for file in os.listdir(DATA_DIR):

        if file.endswith(".pdf"):

            path = os.path.join(DATA_DIR, file)

            ingest_pdf(path, store)

    store.save()

    # Ask question

    query = input("\nQuestion: ")

    answer, sources = query_rag(query, store)

    print("\nAnswer:\n")
    print(answer)