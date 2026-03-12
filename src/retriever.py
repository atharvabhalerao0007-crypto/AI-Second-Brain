# src/retriever.py
import numpy as np

class Retriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5):
        """
        Return top-k most relevant documents from vector store
        """
        return self.vector_store.search(query_vector, top_k)