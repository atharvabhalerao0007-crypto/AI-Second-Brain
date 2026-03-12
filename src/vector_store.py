import faiss
import numpy as np
import pickle
from pathlib import Path

class VectorStore:
    def __init__(self, dim: int, store_path: str):
        self.dim = dim
        self.store_path = Path(store_path)
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

        if self.store_path.exists():
            self.load_store()

    def add_vectors(self, embeddings: np.ndarray, texts: list):
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.save_store()

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        distances, indices = self.index.search(query_vector, top_k)
        results = [self.texts[i] for i in indices[0]]
        return results

    def save_store(self):
        with open(self.store_path, "wb") as f:
            pickle.dump({"index": self.index, "texts": self.texts}, f)

    def load_store(self):
        with open(self.store_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.texts = data["texts"]