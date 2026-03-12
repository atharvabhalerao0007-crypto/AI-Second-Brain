from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=True)