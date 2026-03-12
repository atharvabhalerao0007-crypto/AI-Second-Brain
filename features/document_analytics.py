import spacy
from collections import Counter

try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def get_document_stats(text):

    doc = nlp(text)

    word_count = len([token for token in doc if not token.is_punct])
    sentence_count = len(list(doc.sents))

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    keywords = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.text) > 3
    ]

    top_keywords = Counter(keywords).most_common(10)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "entities": entities[:10],
        "keywords": top_keywords,
    }