import numpy as np
import faiss
import pickle
from config.settings import VECTOR_DB_PATH

# Dummy initial embeddings (empty for now)
dim = 384  # Match embedding dimension from your embeddings model
index = faiss.IndexFlatL2(dim)
texts = []

# Save initial vector store
store_path = VECTOR_DB_PATH + "/store.pkl"
with open(store_path, "wb") as f:
    pickle.dump({"index": index, "texts": texts}, f)

print(f"Initialized empty vector store at {store_path}")