from types import SimpleNamespace

from src.generator import extract_text_from_gemini_response, remove_source_section


def test_extract_text_from_gemini_response_ignores_non_text_parts():
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="First response text."),
                        SimpleNamespace(text=None),
                        SimpleNamespace(thought_signature="sig-123"),
                    ]
                )
            )
        ]
    )

    assert extract_text_from_gemini_response(response) == "First response text."


def test_remove_source_section_from_answer_text():
    answer = """Employees can report harassment to HR or their manager.

Sources:
- Human Rights Policy | Section: Reporting | Page: 4
"""

    assert remove_source_section(answer) == "Employees can report harassment to HR or their manager."


def test_remove_markdown_source_section_from_answer_text():
    answer = """Employees should follow the approval process before booking travel.

## References
Travel Policy, page 3
"""

    assert remove_source_section(answer) == "Employees should follow the approval process before booking travel."
