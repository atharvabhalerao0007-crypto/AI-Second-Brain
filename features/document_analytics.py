import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def get_document_stats(pages, chunks):

    text = " ".join(pages)

    words = text.split()

    word_count = len(words)

    reading_time = round(word_count / 200)  # avg reading speed

    doc = nlp(text)

    entities = [ent.text for ent in doc.ents]

    entity_count = len(set(entities))

    keywords = [token.text.lower() for token in doc
                if token.is_alpha and not token.is_stop]

    top_keywords = Counter(keywords).most_common(10)

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "entities": entity_count,
        "reading_time": reading_time,
        "top_keywords": top_keywords
    }