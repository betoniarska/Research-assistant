from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, chunks):
        pairs = [(query, c["text"]) for c in chunks]

        scores = self.model.predict(pairs)

        for c, s in zip(chunks, scores):
            c["rerank_score"] = float(s)

        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)