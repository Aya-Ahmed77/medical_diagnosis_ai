"""
Text cleaning / normalization used by BOTH training-time dataset
preparation and inference-time prediction, so the two never drift apart.

Design choice (documented per project requirement 6): we do NOT blindly
strip stopwords. Words like "no", "not", "without" change medical
meaning ("no fever" vs "fever"), so a conservative custom stopword list
is used instead of a generic NLTK/spaCy list, and negation words are
explicitly preserved.
"""
import re
import string
from typing import List

# Deliberately conservative -- purely functional filler words that do not
# carry medical meaning. Negation terms and quantifiers are excluded on
# purpose.
_SAFE_STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "it", "as", "at", "by", "from",
}

_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation if p not in "-"})


def clean_text(raw_text: str) -> str:
    """Lowercase, strip HTML leftovers/punctuation (keeping hyphens, since
    terms like 'shortness-of-breath' style phrasing and hyphenated medical
    terms are meaningful), collapse whitespace."""
    if not raw_text:
        return ""
    text = raw_text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # stray HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer applied after clean_text. A regex-based
    tokenizer is used (not full NLTK) to keep the pipeline dependency-light
    and perfectly reproducible between training and inference."""
    if not text:
        return []
    return text.split()


def remove_safe_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in _SAFE_STOPWORDS]


def normalize_symptom_phrase(phrase: str) -> str:
    """Normalize a single symptom/cause phrase (as scraped, e.g. a <li>
    item) into a clean lowercase string with stopwords removed but
    medically meaningful phrasing preserved."""
    cleaned = clean_text(phrase)
    tokens = remove_safe_stopwords(tokenize(cleaned))
    return " ".join(tokens)


def build_model_input_text(symptoms: List[str], extra_text: str = "") -> str:
    """Combine a list of symptom phrases (+ optional free text, e.g. what a
    user typed) into a single normalized string used as model input.
    This is the SAME function used at training time (from scraped/synthetic
    symptom lists) and at inference time (from user free text split into
    clauses), which is what keeps train/inference representations aligned.
    """
    parts = [normalize_symptom_phrase(s) for s in symptoms if s and s.strip()]
    if extra_text:
        parts.append(normalize_symptom_phrase(extra_text))
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    return " . ".join(deduped)


def split_free_text_into_clauses(free_text: str) -> List[str]:
    """Split a user's free-text symptom description into clause-like
    fragments on common separators (commas, 'and', semicolons, periods),
    so it can be fed through the same normalization as scraped symptom
    lists."""
    if not free_text:
        return []
    raw_parts = re.split(r",|;|\.|\band\b", free_text.lower())
    return [p.strip() for p in raw_parts if p.strip()]
