try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def get_document_stats(text):
    stats = {}

    stats["characters"] = len(text)
    stats["words"] = len(text.split())

    if nlp:
        doc = nlp(text)
        stats["sentences"] = len(list(doc.sents))
        stats["entities"] = len(doc.ents)
    else:
        stats["sentences"] = text.count(".")
        stats["entities"] = 0

    return stats