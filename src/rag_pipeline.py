# src/rag_pipeline.py

import numpy as np


class RAGPipeline:

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm


    def answer_question(self, question, top_k=3):

        # Convert question to embedding
        question_embedding = self.llm.get_embedding(question)
        question_embedding = np.array([question_embedding])

        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(question_embedding, top_k=top_k)

        # Combine context
        context = "\n".join(retrieved_chunks)

        prompt = f"""
You are an intelligent document assistant.

Your task is to answer questions using ONLY the provided context.

Instructions:
- Provide a clear and complete explanation.
- Use bullet points if helpful.
- Do not invent information.
- If the answer cannot be found in the context, say:
  "The document does not contain enough information to answer this."

Context:
{context}

Question:
{question}

Helpful Answer:
"""

        answer = self.llm.generate_text(prompt)

        return answer, retrieved_chunks