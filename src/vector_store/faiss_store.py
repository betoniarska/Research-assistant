# FAISS-based vector store for paper chunks with metadata management and balanced retrieval.

from collections import defaultdict
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



    def add(self, chunks):

        # Check if paper is already indexed to avoid duplicates (based on paper_id)

        if self.chunks:
            existing_ids = {c["paper_id"] for c in self.chunks}
            if chunks[0]["paper_id"] in existing_ids:
                print(f"Already indexed: {chunks[0]['paper_title']}, skipping.")
                return

        texts = [c["text"] for c in chunks]

        embeddings = embedding_service.encode(texts)

        embeddings = np.array(
            embeddings
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        if self.index is None:

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.chunks = []

        self.index.add(embeddings)

        self.chunks.extend(chunks)
        
    def save(self):
        
        # Persist FAISS + metadata

        os.makedirs("storage", exist_ok=True)

        faiss.write_index(self.index, self.index_path)

        with open(self.meta_path, "w") as f:
            json.dump(self.chunks, f)


    def load(self):

        # Load FAISS + metadata from disk


        self.index = faiss.read_index(self.index_path)

        with open(self.meta_path, "r") as f:
            self.chunks = json.load(f)


        return self.index, self.chunks
    

    def search(self, query, k=15):

        query_vec = embedding_service.encode([query])

        query_vec = np.array(query_vec).astype("float32")

        faiss.normalize_L2(query_vec) # normalize for cosine similarity aswell

        scores, indices = self.index.search(query_vec, k)

        # deduplicate results (in case of identical sections) while preserving order
        seen = set()
        results = []

        # print scores with section titles for debugging
        for score, idx in zip(scores[0], indices[0]):
            print(f"{score:.4f} | {self.chunks[idx]['section']}")

        # retrieve corresponding chunks for results
        for score, idx in zip(scores[0], indices[0]):

            chunk = self.chunks[idx]

            if chunk["id"] in seen:
                continue

            seen.add(chunk["id"])

            result = chunk.copy()
            result["score"] = float(score)

            results.append(result)

        return results
    

    # Balanced retrieval: get top-k candidates per paper independently to ensure diversity of sources in the recall stage before reranking
    def search_balanced(self, query, k_per_paper=20):
        
        query_vec = np.array(embedding_service.encode([query])).astype("float32")
        faiss.normalize_L2(query_vec)

        # Group chunk indices by paper
        paper_indices = defaultdict(list)
        for idx, chunk in enumerate(self.chunks):
            paper_indices[chunk["paper_id"]].append(idx)

        results = []
        seen = set()

        for paper_id, indices in paper_indices.items():

            # Build a temporary index for this paper's chunks
            sub_chunks = [self.chunks[i] for i in indices]
            sub_texts = [c["text"] for c in sub_chunks]
            sub_vecs = np.array(embedding_service.encode(sub_texts)).astype("float32")
            faiss.normalize_L2(sub_vecs)

            sub_index = faiss.IndexFlatIP(sub_vecs.shape[1])
            sub_index.add(sub_vecs)

            k = min(k_per_paper, len(sub_chunks))
            scores, local_indices = sub_index.search(query_vec, k)

            for score, local_idx in zip(scores[0], local_indices[0]):
                chunk = sub_chunks[local_idx]
                if chunk["id"] not in seen:
                    seen.add(chunk["id"])
                    r = chunk.copy()
                    r["score"] = float(score)
                    results.append(r)

        return results
    
    def index_exists(self):
        return os.path.exists(self.index_path) and os.path.exists(self.meta_path)