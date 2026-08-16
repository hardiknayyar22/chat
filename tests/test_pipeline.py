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


def test_objective_keyword_chunks_are_preferred():
    chunks = [
        {
            "chunk_id": "header",
            "policy_name": "Performance Improvement Plan (Pip) Policy I 2025",
            "section": "Performance Improvement Plan (PIP) Policy - 2025",
            "text": "Performance Improvement Plan (PIP) Policy - 2025 Table of Contents 1.0 Objective 2.0 Eligibility 3.0 Process Workflow",
        },
        {
            "chunk_id": "objective",
            "policy_name": "Performance Improvement Plan (Pip) Policy I 2025",
            "section": "Performance Improvement Plan (PIP) Policy - 2025",
            "text": "1.0 Objective. The aim of this policy is to address and manage employee underperformance. It is initiated when written or verbal feedback has not led to expected improvement in performance.",
        },
    ]

    context = build_context(chunks, "Tell me about the objectives of the PIP policy")

    assert context and "objective" in (context[0].get("text") or "").lower()
    assert "table of contents" not in (context[0].get("text") or "").lower()


def test_real_pip_sections_are_kept_in_context():
    chunks = [
        {
            "chunk_id": "toc",
            "policy_name": "Performance Improvement Plan (Pip) Policy I 2025",
            "section": "Performance Improvement Plan (PIP) Policy - 2025",
            "text": "Table of Contents 1.0 Objective 2.0 Eligibility 3.0 Process Workflow",
        },
        {
            "chunk_id": "actual_objective",
            "policy_name": "Performance Improvement Plan (Pip) Policy I 2025",
            "section": "Performance Improvement Plan (PIP) Policy - 2025",
            "text": "1.0 Objective. The aim of this policy is to address and manage employee underperformance. It is initiated when written or verbal feedback has not led to expected improvement in performance.",
        },
        {
            "chunk_id": "actual_eligibility",
            "policy_name": "Performance Improvement Plan (Pip) Policy I 2025",
            "section": "Performance Improvement Plan (PIP) Policy - 2025",
            "text": "2.0 Eligibility. This policy is applicable for all regular, full-time and part-time employees working in India at Coforge Limited and its subsidiaries.",
        },
    ]

    context = build_context(chunks, "tell me about the objectives and eligibility of the pip policy")

    assert any("1.0 objective" in (item.get("text") or "").lower() for item in context)
    assert any("2.0 eligibility" in (item.get("text") or "").lower() for item in context)
