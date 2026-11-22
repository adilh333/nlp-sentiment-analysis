"""Extractive summary and key-issue phrases.

No hosted model and no API key. The summary is the two intake sentences
closest to the document centroid under TF-IDF. Key issues are the highest
scoring RAKE-style phrases. A complaints handler can read both in a few
seconds and still has to open the case.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "have", "has", "had",
    "was", "were", "are", "been", "being", "but", "not", "you", "your", "she",
    "her", "his", "him", "they", "their", "them", "our", "out", "about", "into",
    "over", "after", "before", "because", "which", "what", "when", "where",
    "who", "how", "its", "it's", "also", "than", "then", "there", "their",
    "would", "could", "should", "did", "does", "doing", "complains", "complain",
    "complaint", "said", "says", "told", "mr", "mrs", "ms", "miss",
}


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    return [part for part in parts if len(part) > 40]


def _tokens(text: str) -> list[str]:
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in _STOP]


def extractive_summary(text: str, n_sentences: int = 2) -> str:
    rows = sentences(text)
    if not rows:
        compact = " ".join(text.split())
        return compact[:400]
    if len(rows) <= n_sentences:
        return " ".join(rows)
    docs = [_tokens(row) for row in rows]
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n = len(docs)
    weights: list[dict[str, float]] = []
    centroid: Counter[str] = Counter()
    for doc in docs:
        tf = Counter(doc)
        length = max(len(doc), 1)
        vec = {}
        for token, count in tf.items():
            idf = math.log((1 + n) / (1 + df[token])) + 1
            vec[token] = (count / length) * idf
        weights.append(vec)
        centroid.update(vec)
    for token in centroid:
        centroid[token] /= n

    def score(vec: dict[str, float]) -> float:
        return sum(value * centroid[token] for token, value in vec.items())

    ranked = sorted(range(len(rows)), key=lambda i: score(weights[i]), reverse=True)
    chosen = sorted(ranked[:n_sentences])
    return " ".join(rows[i] for i in chosen)


def key_issues(text: str, k: int = 5) -> list[str]:
    """RAKE: candidate phrases split on stopwords, scored by word degree/frequency."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    phrases: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _STOP or len(token) < 3:
            if current:
                phrases.append(current)
                current = []
            continue
        current.append(token)
    if current:
        phrases.append(current)
    freq: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in phrases:
        unique = set(phrase)
        for token in unique:
            freq[token] += 1
            degree[token] += len(phrase)
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for phrase in phrases:
        if len(phrase) > 4:
            continue
        surface = " ".join(phrase)
        if surface in seen or len(surface) < 6:
            continue
        seen.add(surface)
        score = sum((degree[token] / freq[token]) for token in phrase)
        scored.append((score, surface))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [phrase for _, phrase in scored[:k]]
