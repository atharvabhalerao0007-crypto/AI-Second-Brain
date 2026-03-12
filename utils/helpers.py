# utils/helpers.py
def clean_text(text: str) -> str:
    """
    Basic text cleaning: strip spaces, remove newlines
    """
    return " ".join(text.split())