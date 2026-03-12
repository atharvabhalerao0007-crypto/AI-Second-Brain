import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    def __init__(self, dim: int, store_path: str):
        self.dim = dim
        self.store_path = Path(store_path)

        self.index_file = self.store_path / "faiss.index"
        self.text_file = self.store_path / "texts.pkl"

        self.store_path.mkdir(parents=True, exist_ok=True)

        if self.index_file.exists() and self.text_file.exists():
            self.load_store()
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def add_vectors(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")

        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        self.index.add(embeddings)
        self.texts.extend(texts)

        self.save_store()

    def search(self, query_vector, top_k: int = 5):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array(query_vector).astype("float32")

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])

        return results

    def save_store(self):
        faiss.write_index(self.index, str(self.index_file))

        with open(self.text_file, "wb") as f:
            pickle.dump(self.texts, f)

    def load_store(self):
        self.index = faiss.read_index(str(self.index_file))

        with open(self.text_file, "rb") as f:
            self.texts = pickle.load(f)