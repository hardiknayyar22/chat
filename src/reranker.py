import math
import re
from typing import List, Dict, Any


SECTION_KEYWORDS = (
    "objective",
    "eligibility",
    "process",
    "workflow",
    "applicability",
    "scope",
    "requirements",
    "procedure",
    "criteria",
    "purpose",
    "definition",
)


def _tokenize(text: str) -> List[str]:
    return [part.lower() for part in re.sub(r"[^a-z0-9\s]", " ", (text or "")).split() if part.strip()]


def _is_noise_chunk(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return True
    if "table of contents" in lowered:
        return True
    if "©" in lowered and "policy" in lowered and len(lowered.split()) < 80:
        return True
    if "version change history" in lowered:
        return True
    if "about coforge" in lowered:
        return True
    if "document title" in lowered:
        return True
    return False


def rerank_chunks(question: str, chunks: List[Dict[str, Any]], limit: int = 6) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    q_terms = set(_tokenize(question))
    q_lower = (question or "").lower()
    scored = []
    for chunk in chunks:
        text = chunk.get("text") or ""
        lowered = text.lower()
        tokens = _tokenize(text)
        is_noise = _is_noise_chunk(text)

        if is_noise:
            score = -999.0
        else:
            exact_hits = sum(1 for term in q_terms if term in lowered)
            keyword_overlap = exact_hits / max(len(q_terms), 1)
            section_hits = sum(1 for keyword in SECTION_KEYWORDS if keyword in lowered and keyword in q_lower)
            phrase_boost = 1.5 if any(phrase in lowered for phrase in ["1.0 objective", "2.0 eligibility", "3.0 process", "objective.", "eligibility."]) else 0.0

            score = keyword_overlap * 3.0 + (exact_hits * 0.5) + section_hits * 1.5 + phrase_boost
            if chunk.get("section"):
                score += 0.2
            if tokens:
                score += min(1.0, len(q_terms) / max(len(tokens), 1)) * 0.5

        scored.append((score, chunk))

    reranked = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)]
    return reranked[:limit]
