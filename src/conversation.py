import re
from typing import Any, Dict, List, Optional


FOLLOW_UP_PATTERNS = [
    "what about",
    "how about",
    "what happens next",
    "how do i do that",
    "what are the exceptions",
    "and what about",
    "does this apply",
    "what about reporting",
    "how can an employee report",
    "does it apply",
    "what are the conditions",
]


def detect_follow_up(query: str, conversation_state: Optional[Dict[str, Any]] = None) -> bool:
    if not query:
        return False
    state = conversation_state or {}
    last_question = (state.get("last_question") or "").lower()
    recent_topic = (state.get("recent_topic") or "").lower()
    normalized = query.lower()
    if any(pattern in normalized for pattern in FOLLOW_UP_PATTERNS):
        return True
    if len(normalized.split()) <= 8 and any(term in normalized for term in ["it", "that", "this", "they", "them"]):
        return True
    if last_question and recent_topic and recent_topic in normalized:
        return True
    return False


def update_context(conversation_state: Optional[Dict[str, Any]], question: str, answer: str, active_policy: Optional[str] = None, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    state = dict(conversation_state or {})
    state["last_question"] = question
    state["last_answer"] = answer
    if active_policy:
        state["active_policy"] = active_policy
    if sources:
        state["last_sources"] = sources
    state["recent_topic"] = question[:160]
    return state


def get_follow_up_policy(conversation_state: Optional[Dict[str, Any]]) -> Optional[str]:
    if not conversation_state:
        return None
    return conversation_state.get("active_policy")
