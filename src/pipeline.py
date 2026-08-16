import logging
import re
import time
from typing import Any, Dict, List, Optional

from src.conversation import detect_follow_up, update_context
from src.generator import generate_answer
from src.reranker import rerank_chunks
from src.router import route_query
from src.retriever import ALL_POLICIES_KEY, hybrid_retrieve


RETRIEVAL_TOP_K = 12
CONTEXT_CHUNK_LIMIT = 4
MAX_CONTEXT_CHARS_PER_CHUNK = 1800
MAX_CONTEXT_TOTAL_CHARS = 5000
SECTION_PRIORITY_KEYWORDS = (
    "objective",
    "eligibility",
    "process",
    "scope",
    "applicability",
    "requirements",
    "procedure",
    "policy",
)
logger = logging.getLogger(__name__)


def _section_priority_score(chunk: Dict[str, Any]) -> int:
    text = (chunk.get("text") or "").lower()
    section = (chunk.get("section") or "").lower()
    combined = f"{section} {text}"
    return sum(1 for keyword in SECTION_PRIORITY_KEYWORDS if keyword in combined)


def _is_noise_chunk(chunk: Dict[str, Any]) -> bool:
    text = (chunk.get("text") or "").lower()
    if not text:
        return True
    if "table of contents" in text:
        return True
    if "version change history" in text:
        return True
    if "about coforge" in text:
        return True
    if "©" in text and "policy" in text and len(text.split()) < 80:
        return True
    return False


def _compact_context_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_CHUNK) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    collected = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            collected.append(candidate)
            continue
        if current:
            return current.strip()
        return sentence[: max_chars - 3].rstrip() + "..."

    return current.strip() if current else stripped[: max_chars - 3].rstrip() + "..."


def build_context(chunks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    reranked = rerank_chunks(question, chunks, limit=CONTEXT_CHUNK_LIMIT)
    priority_sorted = sorted(
        reranked,
        key=lambda item: (
            0 if _is_noise_chunk(item) else 1,
            _section_priority_score(item),
            item.get("page", 0),
        ),
        reverse=True,
    )

    deduped = []
    seen = set()
    char_budget = 0
    for chunk in priority_sorted:
        key = (chunk.get("policy_name"), chunk.get("page"), chunk.get("section"), chunk.get("text", "")[:120])
        if key in seen:
            continue
        seen.add(key)
        compacted = dict(chunk)
        compacted["text"] = _compact_context_text(chunk.get("text", ""), MAX_CONTEXT_CHARS_PER_CHUNK)
        compacted["_context_chars"] = len(compacted["text"])
        if deduped and char_budget + compacted["_context_chars"] > MAX_CONTEXT_TOTAL_CHARS:
            break
        deduped.append(compacted)
        char_budget += compacted["_context_chars"]
    return deduped


def answer_query(
    question: str,
    chunks: List[Dict[str, Any]],
    registry: Optional[Dict[str, Dict[str, Any]]] = None,
    conversation_state: Optional[Dict[str, Any]] = None,
    bm25_indexes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    registry = registry or {}
    state = dict(conversation_state or {})
    route = route_query(question, registry, state)
    route_time = time.time() - t0

    retrieval_time = 0.0
    rerank_time = 0.0
    llm_time = 0.0

    policy_name = route.get("policy_id")
    if route.get("route") == "specific_policy":
        candidate_chunks = [chunk for chunk in chunks if chunk.get("policy_name") == policy_name]
    else:
        candidate_chunks = list(chunks)
    bm25_index = (bm25_indexes or {}).get(policy_name or ALL_POLICIES_KEY)

    t1 = time.time()
    retrieved = hybrid_retrieve(
        question,
        candidate_chunks,
        top_k=RETRIEVAL_TOP_K,
        policy_name=policy_name,
        bm25_index=bm25_index,
    )
    retrieval_time = time.time() - t1

    t2 = time.time()
    context = build_context(retrieved, question)
    rerank_time = time.time() - t2

    if not context:
        answer = "I couldn't find information about this in the available Coforge India policy documents."
        llm_time = 0.0
    else:
        t3 = time.time()
        answer = generate_answer(question, context, state)
        llm_time = time.time() - t3

    logger.info(
        "Query timing | route=%s policy=%s retrieval=%.3fs rerank=%.3fs llm=%.3fs total=%.3fs context_chunks=%s",
        route.get("route"),
        policy_name,
        retrieval_time,
        rerank_time,
        llm_time,
        time.time() - t0,
        len(context),
    )

    state = update_context(state, question, answer, active_policy=policy_name or state.get("active_policy"), sources=context)
    route["route"] = route.get("route", "all_policies")
    route["timings"] = {
        "routing_time": round(route_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "reranking_time": round(rerank_time, 3),
        "llm_time": round(llm_time, 3),
        "total_response_time": round(time.time() - t0, 3),
    }
    return {"answer": answer, "sources": context, "route": route, "conversation_state": state}
