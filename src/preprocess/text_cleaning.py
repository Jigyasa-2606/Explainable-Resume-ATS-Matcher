import re
from typing import Set

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep domain-relevant short tokens that default stopword lists may drop.
DOMAIN_KEEP_WORDS: Set[str] = {
    "sql",
    "ml",
    "ai",
    "aws",
    "gcp",
    "azure",
    "nlp",
    "cv",
    "etl",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) - DOMAIN_KEEP_WORDS


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    if not text:
        return ""
    tokens = [tok for tok in text.split() if tok not in STOP_WORDS]
    return " ".join(tokens)


def clean_text(text: str) -> str:
    normalized = normalize_text(text)
    return remove_stopwords(normalized)
