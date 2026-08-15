"""
Decides whether a user's query names a specific policy.

This is the one piece of logic that makes the "answer only from that
policy" requirement work. It is deliberately simple: fuzzy string
matching against the list of known policy names pulled straight out
of the vector store's metadata, so it self-updates whenever you
re-run ingest.py with new documents. No extra LLM call needed.
"""

from difflib import SequenceMatcher
from typing import List, Optional

import config


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_known_policy_names(vectorstore) -> List[str]:
    """Pull the distinct policy_name values out of the vector store's
    stored document metadata."""
    names = set()
    # FAISS (via LangChain) keeps documents in an internal docstore dict.
    for doc in vectorstore.docstore._dict.values():
        name = doc.metadata.get("policy_name")
        if name:
            names.add(name)
    return sorted(names)


def detect_policy(query: str, known_policy_names: List[str]) -> Optional[str]:
    """Return the best-matching policy name if the query clearly names one,
    else None (meaning: search across all policies).

    Matching strategy, cheapest first:
      1. Exact substring match (case-insensitive) - handles the common case
         where the user types the policy name verbatim, e.g.
         "what does the leave policy say about sandwich leave".
      2. Fuzzy match on the query against each policy name, for typos /
         partial names, e.g. "leave rules" -> "Leave Policy".
    """
    if not query or not known_policy_names:
        return None

    query_lower = query.lower()

    # 1. Substring match
    for name in known_policy_names:
        if name.lower() in query_lower:
            return name

    # 2. Fuzzy match - score the query against a sliding window comparison
    best_name, best_score = None, 0.0
    for name in known_policy_names:
        score = _similarity(query_lower, name.lower())
        # Also check word overlap, since SequenceMatcher penalizes
        # different-length strings heavily (e.g. short query vs long name).
        name_words = set(name.lower().split())
        query_words = set(query_lower.split())
        overlap = len(name_words & query_words) / max(len(name_words), 1)
        combined_score = max(score, overlap)

        if combined_score > best_score:
            best_name, best_score = name, combined_score

    if best_score >= config.POLICY_MATCH_THRESHOLD:
        return best_name

    return None
