from sentence_transformers import SentenceTransformer

class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=False
        )

embedding_service = EmbeddingService()