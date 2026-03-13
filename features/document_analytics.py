import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def get_document_stats(text):

    doc = nlp(text)

    words = [token.text.lower() for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    word_count = len(words)
    sentence_count = len(sentences)

    keywords = Counter(words).most_common(10)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Approximate pages (500 words per page)
    pages = max(1, word_count // 500)

    return {
        "pages": pages,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "keywords": keywords,
        "entities": entities
    }