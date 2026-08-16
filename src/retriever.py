import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

import config
from src.embeddings import embed_texts


ALL_POLICIES_KEY = "__all__"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def bm25_score(query: str, text: str, idf: Dict[str, float], avgdl: float, doc_len: int) -> float:
    score = 0.0
    q_terms = Counter(tokenize(query))
    text_terms = Counter(tokenize(text))
    for term, qf in q_terms.items():
        if term not in text_terms:
            continue
        tf = text_terms[term]
        score += (idf.get(term, 1.0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))) * qf
    return score


k1 = 1.5
b = 0.75


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    corpus = [chunk.get("text", "") for chunk in chunks]
    df: Dict[str, int] = {}
    for text in corpus:
        seen = set()
        for word in set(tokenize(text)):
            if word not in seen:
                df[word] = df.get(word, 0) + 1
                seen.add(word)
    total_docs = max(len(corpus), 1)
    idf = {word: math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1.0) for word, freq in df.items()}
    avgdl = sum(len(tokenize(text)) for text in corpus) / total_docs if corpus else 1.0
    return {"idf": idf, "avgdl": avgdl, "corpus": corpus}


def build_bm25_indexes(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexes = {ALL_POLICIES_KEY: build_bm25_index(chunks)}
    policies = sorted({chunk.get("policy_name") for chunk in chunks if chunk.get("policy_name")})
    for policy_name in policies:
        policy_chunks = [chunk for chunk in chunks if chunk.get("policy_name") == policy_name]
        indexes[policy_name] = build_bm25_index(policy_chunks)
    return indexes


def load_chunks() -> List[Dict[str, Any]]:
    chunks_path = config.CHUNKS_PATH
    if not chunks_path.exists():
        return []
    with chunks_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload.get("chunks", [])
    return payload


def load_bm25_index() -> Dict[str, Any]:
    if not config.BM25_PATH.exists():
        return {"idf": {}, "avgdl": 1.0, "corpus": []}
    with config.BM25_PATH.open("rb") as handle:
        raw = handle.read()
    try:
        return pickle.loads(raw)
    except Exception:
        return json.loads(raw.decode("utf-8"))


def load_faiss_index() -> Optional[faiss.Index]:
    if not config.FAISS_INDEX_PATH.exists():
        return None
    return faiss.read_index(str(config.FAISS_INDEX_PATH))


def load_index_bundle() -> Dict[str, Any]:
    chunks = load_chunks()
    if not chunks:
        empty_bm25 = {"idf": {}, "avgdl": 1.0, "corpus": []}
        return {"chunks": [], "bm25": empty_bm25, "bm25_indexes": {ALL_POLICIES_KEY: empty_bm25}, "faiss": None}
    bm25 = load_bm25_index()
    return {"chunks": chunks, "bm25": bm25, "bm25_indexes": build_bm25_indexes(chunks), "faiss": load_faiss_index()}


def _boost_section_relevance(text: str, query: str) -> float:
    lowered_query = query.lower()
    lowered_text = text.lower()
    priority_terms = [
        "objective",
        "eligibility",
        "process",
        "applicability",
        "scope",
        "requirements",
        "procedure",
        "criteria",
    ]
    score = 0.0
    for term in priority_terms:
        if term in lowered_query and term in lowered_text:
            score += 0.75
    return score


def hybrid_retrieve(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 10,
    policy_name: Optional[str] = None,
    bm25_index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    filtered = chunks if not policy_name else [chunk for chunk in chunks if chunk.get("policy_name") == policy_name]
    if not filtered:
        return []

    bm25_index = bm25_index or build_bm25_index(filtered)
    scored = []
    for chunk in filtered:
        text = chunk.get("text", "")
        bm25 = bm25_score(query, text, bm25_index["idf"], bm25_index["avgdl"], len(tokenize(text)))
        token_overlap = sum(1 for token in set(tokenize(query)) if token in set(tokenize(text)))
        semantic_score = 1.0 if token_overlap else 0.0
        section_boost = _boost_section_relevance(text, query)
        total = bm25 + semantic_score * 2.0 + token_overlap * 0.25 + section_boost
        scored.append((total, chunk))

    ranked = [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)]
    return ranked[:top_k]
