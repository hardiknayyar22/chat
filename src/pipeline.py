import time
from typing import Any, Dict, List, Optional

from src.conversation import detect_follow_up, update_context
from src.generator import generate_answer
from src.reranker import rerank_chunks
from src.router import route_query
from src.retriever import ALL_POLICIES_KEY, hybrid_retrieve


RETRIEVAL_TOP_K = 8
CONTEXT_CHUNK_LIMIT = 4


def build_context(chunks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    reranked = rerank_chunks(question, chunks, limit=CONTEXT_CHUNK_LIMIT)
    deduped = []
    seen = set()
    for chunk in reranked:
        key = (chunk.get("policy_name"), chunk.get("page"), chunk.get("section"), chunk.get("text", "")[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
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
