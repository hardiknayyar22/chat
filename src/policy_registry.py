import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_policy_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[_-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.title()


def canonicalize_policy_name(filename: str) -> str:
    stem = Path(filename).stem
    normalized = stem.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.title()


def build_policy_registry(policy_directory: Path, alias_map: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
    alias_map = alias_map or {}
    registry: Dict[str, Dict[str, Any]] = {}
    if not policy_directory.exists():
        return registry

    for pdf_path in sorted(policy_directory.glob("*.pdf")):
        canonical = canonicalize_policy_name(pdf_path.name)
        aliases = alias_map.get(canonical, [])
        alias_list = [a.strip() for a in aliases if a.strip()]
        alias_list.extend([canonical, pdf_path.stem])

        if "PIP" in canonical.upper() or "PERFORMANCE IMPROVEMENT PLAN" in canonical.upper():
            alias_list.extend(["PIP", "Performance Improvement Plan", "Performance Improvement Plan Policy"])

        if "LEAVE" in canonical.upper() and "POLICY" in canonical.upper():
            alias_list.extend([
                "Leave Policy",
                "HR Leave Policy",
                "India Leave Policy",
                "Leave and Holiday Policy",
                "Comp Off Policy",
                "LWP Policy",
            ])

        alias_list = sorted(set(alias_list), key=lambda x: (x.lower() != canonical.lower(), x.lower()))
        registry[canonical] = {
            "canonical_name": canonical,
            "filename": pdf_path.name,
            "aliases": alias_list,
            "normalized_name": normalize_policy_name(canonical),
        }
    return registry


def match_policy_name(query: str, registry: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not query or not registry:
        return None

    normalized_query = normalize_policy_name(query).lower()
    for policy_name, info in registry.items():
        candidate_names = [policy_name] + info.get("aliases", [])
        for candidate in candidate_names:
            cand = normalize_policy_name(candidate).lower()
            if cand in normalized_query or normalized_query in cand:
                return policy_name
    return None
