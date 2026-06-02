import faiss
import numpy as np
import json
import os

from src.services.embedding_service import embedding_service


class FAISSStore:

    def __init__(
        self,
        index_path="storage/index.faiss",
        meta_path="storage/chunks.json"
    ):
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = None
        self.chunks = None



    def build(self, chunks):


        # Create FAISS index from scratch


        texts = [c["text"] for c in chunks]

        embeddings = embedding_service.encode(texts)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        # Flat index = brute-force exact search
        self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        self.chunks = chunks

        return self.index
    
    def save(self):
        
        # Persist FAISS + metadata

        os.makedirs("storage", exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w") as f:
            json.dump(self.chunks, f)


    def load(self):

        # Load FAISS + metadata from disk

        print("📦 Loading FAISS index from disk...")


        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "r") as f:
            self.chunks = json.load(f)

        print("✅ FAISS loaded into memory")

        return self.index, self.chunks
    

    def search(self, query, k=5):

        query_vec = embedding_service.encode([query])

        query_vec = np.array(query_vec).astype("float32")

        distances, indices = self.index.search(query_vec, k)

        # deduplicate results (in case of identical sections) while preserving order
        seen = set()
        results = []

        for i in indices[0]:

            chunk = self.chunks[i]

            if chunk["id"] in seen:
                continue

            seen.add(chunk["id"])
            results.append(chunk)

        return results