# create embeddings for chunks using sentence transformers
import faiss
#from sentence_transformers import SentenceTransformer
import numpy as np
from src.services.embedding_service import embedding_service

def create_index(chunks):

    texts = [c["text"] for c in chunks]

    embeddings = embedding_service.encode(texts)

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)

    return index


def search(query, index, chunks, k=5):

    query_vec = embedding_service.encode([query])

    query_vec = np.array(
        query_vec
    ).astype("float32")

    distances, indices = index.search(
        query_vec,
        k
    )

    return [chunks[i] for i in indices[0]]