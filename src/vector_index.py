# create embeddings for chunks using sentence transformers
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def create_index(chunks):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]

    embeddings = np.array(model.encode(texts, show_progress_bar=True)).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Total vectors:", index.ntotal)

    return index, model


def search(query, model, index, chunks, k=5):

    query_vec = model.encode([query]).astype("float32")
    
    distances, indices = index.search(query_vec, k)
    
    results = []
    for i in indices[0]:
        results.append(chunks[i])
    
    return results