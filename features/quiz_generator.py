class QuizGenerator:
    """
    Generate quiz questions and answers from context.
    """

    def __init__(self, llm):
        self.llm = llm

    def generate_quiz(self, context: str, num_questions: int = 5) -> str:
        """
        Returns a quiz in Q&A format.
        """
        prompt = f"Generate {num_questions} multiple-choice questions (with answers) from the following text:\n{context}"
        quiz = self.llm.generate_answer("Generate quiz", context=prompt)
        return quiz