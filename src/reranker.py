import math
from typing import List, Dict, Any


def _tokenize(text: str) -> List[str]:
    return [part.lower() for part in text.replace("\n", " ").split() if part.strip()]


def rerank_chunks(question: str, chunks: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    q_terms = set(_tokenize(question))
    scored = []
    for chunk in chunks:
        text = (chunk.get("text") or "").lower()
        tokens = _tokenize(text)
        exact_hits = sum(1 for term in q_terms if term in text)
        keyword_overlap = exact_hits / max(len(q_terms), 1)
        word_count = len(tokens)
        score = keyword_overlap * 2.0 + (exact_hits * 0.4) + (0.1 if chunk.get("section") else 0.0)
        if word_count:
            score += min(1.0, len(q_terms) / max(word_count, 1)) * 0.5
        scored.append((score, chunk))

    reranked = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)]
    return reranked[:limit]
