from src.retriever import Retriever
import numpy as np

class SemanticSearch:
    def __init__(self, vector_store):
        self.retriever = Retriever(vector_store)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Return top-k semantically similar chunks.
        """
        results = self.retriever.retrieve(np.array([query_embedding]), top_k)
        return results