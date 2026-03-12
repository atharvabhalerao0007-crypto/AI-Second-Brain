import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Documents folder
DOCUMENT_PATH = os.path.join(BASE_DIR, "data", "documents")

# Vector database folder
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_store")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM model
LLM_MODEL = "llama-3.1-8b-instant"

# Groq / LLM API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")