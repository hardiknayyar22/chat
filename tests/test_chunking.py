from src.chunking import chunk_policy_document


def test_chunks_are_generated_and_metadata_retained():
    pages = [
        {
            "page": 1,
            "text": "Human Rights Policy\n\nHarassment-Free Workplace\n\nDefinition of harassment. This includes unwelcome conduct.\n\nReporting process. Employees can report concerns to HR or a manager."
        },
        {
            "page": 2,
            "text": "Policy Scope\n\nThis policy applies to all employees including contractors.\n\nExceptions may apply for interns."
        },
    ]

    chunks = chunk_policy_document(
        policy_name="Human Rights Policy",
        source_file="human-rights-policy.pdf",
        pages=pages,
        chunk_size=80,
        overlap=10,
    )

    assert len(chunks) >= 2
    assert all(chunk["policy_name"] == "Human Rights Policy" for chunk in chunks)
    assert all(chunk["source_file"] == "human-rights-policy.pdf" for chunk in chunks)
    assert all("text" in chunk and chunk["text"] for chunk in chunks)
    assert any(chunk["page"] == 1 for chunk in chunks)
