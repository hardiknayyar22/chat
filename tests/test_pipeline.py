from src.pipeline import answer_query
from src.pipeline import build_context


CHUNKS = [
    {
        "chunk_id": "c1",
        "policy_name": "Human Rights Policy",
        "source_file": "human-rights-policy.pdf",
        "page": 4,
        "section": "Harassment-Free Workplace",
        "text": "Harassment is unwelcome conduct, and employees may report it to HR or the manager.",
    },
    {
        "chunk_id": "c2",
        "policy_name": "Human Rights Policy",
        "source_file": "human-rights-policy.pdf",
        "page": 5,
        "section": "Reporting Mechanism",
        "text": "Employees may also raise concerns through the grievance committee or through the ethics hotline.",
    },
]


def test_basic_end_to_end_pipeline_behavior(monkeypatch):
    def fake_generator(question, context, conversation_state=None):
        return "The Human Rights Policy says harassment includes unwelcome conduct and it can be reported to HR or the grievance committee."

    monkeypatch.setattr("src.pipeline.generate_answer", fake_generator)

    result = answer_query(
        "What does the Human Rights Policy say about harassment?",
        chunks=CHUNKS,
        registry={"Human Rights Policy": {"aliases": ["Human Rights", "HR Policy"]}},
        conversation_state={},
    )

    assert result["answer"]
    assert result["sources"]
    assert result["route"]["policy_id"] == "Human Rights Policy"


def test_unsupported_question_is_safely_rejected(monkeypatch):
    def fake_generator(question, context, conversation_state=None):
        return "I couldn't find information about this in the available Coforge India policy documents."

    monkeypatch.setattr("src.pipeline.generate_answer", fake_generator)

    result = answer_query(
        "What is the capital of France?",
        chunks=CHUNKS,
        registry={"Human Rights Policy": {"aliases": ["Human Rights", "HR Policy"]}},
        conversation_state={},
    )

    assert "couldn't find information" in result["answer"].lower()


def test_context_is_limited_before_prompting():
    chunks = [
        {
            "chunk_id": f"c{i}",
            "policy_name": "Human Rights Policy",
            "source_file": "human-rights-policy.pdf",
            "page": i,
            "section": "Reporting",
            "text": f"Employees can report harassment concern {i} to HR or their manager.",
        }
        for i in range(10)
    ]

    context = build_context(chunks, "How can employees report harassment?")

    assert len(context) <= 4
