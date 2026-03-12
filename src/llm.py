# src/llm.py

from groq import Groq
from config.settings import LLM_MODEL, GROQ_API_KEY
from sentence_transformers import SentenceTransformer


class LLMWrapper:

    def __init__(self, api_key=None, model=None):

        self.api_key = api_key or GROQ_API_KEY
        self.model = model or LLM_MODEL

        # Groq client
        self.client = Groq(api_key=self.api_key)

        # Embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")


    def get_embedding(self, text):
        """
        Generate embedding using sentence-transformers
        """
        embedding = self.embedder.encode(text)
        return embedding.tolist()


    def generate_text(self, prompt):
        """
        Generate response from Groq Llama3
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return completion.choices[0].message.content