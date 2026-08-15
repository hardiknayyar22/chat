import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def alias_matches(query: str, aliases: List[str]) -> Optional[str]:
    normalized_query = normalize_text(query)
    for alias in aliases:
        if not alias:
            continue
        normalized_alias = normalize_text(alias)
        if normalized_alias in normalized_query or normalized_query in normalized_alias:
            return alias
    return None


def route_query(query: str, registry: Optional[Dict[str, Dict[str, Any]]] = None, conversation_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    registry = registry or {}
    state = conversation_state or {}
    normalized_query = normalize_text(query)

    if not normalized_query:
        return {"route": "all_policies", "policy_id": None, "is_follow_up": False, "confidence": 0.0}

    for policy_name, info in registry.items():
        aliases = [policy_name] + list(info.get("aliases", []))
        explicit = alias_matches(normalized_query, aliases)
        if explicit:
            return {"route": "specific_policy", "policy_id": policy_name, "is_follow_up": False, "confidence": 0.95}

    active_policy = state.get("active_policy")
    last_question = (state.get("last_question") or "").lower()
    if active_policy and len(normalized_query.split()) <= 12:
        pronoun_like = any(word in normalized_query for word in ["it", "that", "this", "they", "them", "what about", "how do i", "how can", "does this"])
        if pronoun_like or normalized_query in last_question:
            return {"route": "specific_policy", "policy_id": active_policy, "is_follow_up": True, "confidence": 0.8}

    return {"route": "all_policies", "policy_id": None, "is_follow_up": False, "confidence": 0.9}
