from src.router import route_query


REGISTRY = {
    "Human Rights Policy": {"aliases": ["Human Rights", "HR Policy"]},
    "India Leave Policy": {"aliases": ["Leave Policy", "Annual Leave Policy"]},
}


def test_explicit_policy_name_routes_to_specific_policy():
    result = route_query("What does the Human Rights Policy say about harassment?", REGISTRY)
    assert result["route"] == "specific_policy"
    assert result["policy_id"] == "Human Rights Policy"


def test_no_policy_name_routes_to_all_policies():
    result = route_query("What does the policy say about employee privacy?", REGISTRY)
    assert result["route"] == "all_policies"
    assert result["policy_id"] is None


def test_obvious_follow_up_uses_previous_policy():
    state = {"active_policy": "Human Rights Policy", "last_question": "What does the Human Rights Policy say about harassment?"}
    result = route_query("What about reporting it?", REGISTRY, conversation_state=state)
    assert result["route"] == "specific_policy"
    assert result["policy_id"] == "Human Rights Policy"


def test_unknown_policy_falls_back_safely():
    result = route_query("What is the capital of France?", REGISTRY)
    assert result["route"] == "all_policies"
    assert result["policy_id"] is None
