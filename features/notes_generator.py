class NotesGenerator:
    """
    Generate concise notes from given text chunks or context.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_notes(self, context: str) -> str:
        """
        Uses LLM to summarize or create structured notes.
        """
        prompt = f"Create concise study notes from the following text:\n{context}"
        notes = self.llm.generate_answer("Generate notes", context=prompt)
        return notes