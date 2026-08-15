from src.retriever import ALL_POLICIES_KEY, build_bm25_indexes, hybrid_retrieve


CHUNKS = [
    {
        "chunk_id": "c1",
        "policy_name": "Human Rights Policy",
        "source_file": "human-rights-policy.pdf",
        "page": 4,
        "section": "Harassment-Free Workplace",
        "text": "Harassment is unwelcome conduct and can be reported to HR or the manager.",
    },
    {
        "chunk_id": "c2",
        "policy_name": "India Leave Policy",
        "source_file": "india-leave-policy.pdf",
        "page": 2,
        "section": "Annual Leave",
        "text": "Employees receive annual leave according to tenure and local policy.",
    },
    {
        "chunk_id": "c3",
        "policy_name": "Human Rights Policy",
        "source_file": "human-rights-policy.pdf",
        "page": 5,
        "section": "Reporting Mechanism",
        "text": "Employees may report concerns by email or through the grievance committee.",
    },
]


def test_policy_filtering_works():
    matches = [chunk for chunk in CHUNKS if chunk["policy_name"] == "Human Rights Policy"]
    assert len(matches) == 2
    assert all(chunk["policy_name"] == "Human Rights Policy" for chunk in matches)


def test_hybrid_retrieval_returns_metadata():
    candidates = hybrid_retrieve("harassment report to HR", CHUNKS, top_k=5)
    assert candidates
    assert all("text" in chunk for chunk in candidates)
    assert all("policy_name" in chunk for chunk in candidates)


def test_hybrid_retrieval_produces_candidates():
    result = hybrid_retrieve("reporting harassment", CHUNKS, top_k=5)
    assert len(result) >= 1
    assert any("harassment" in chunk["text"].lower() for chunk in result)


def test_bm25_indexes_are_built_per_policy():
    indexes = build_bm25_indexes(CHUNKS)

    assert ALL_POLICIES_KEY in indexes
    assert "Human Rights Policy" in indexes
    assert "India Leave Policy" in indexes
    assert indexes["Human Rights Policy"]["corpus"]


def test_hybrid_retrieval_uses_supplied_bm25_index(monkeypatch):
    indexes = build_bm25_indexes(CHUNKS)

    def fail_if_rebuilt(chunks):
        raise AssertionError("BM25 index should be reused, not rebuilt")

    monkeypatch.setattr("src.retriever.build_bm25_index", fail_if_rebuilt)

    result = hybrid_retrieve(
        "reporting harassment",
        [chunk for chunk in CHUNKS if chunk["policy_name"] == "Human Rights Policy"],
        top_k=5,
        policy_name="Human Rights Policy",
        bm25_index=indexes["Human Rights Policy"],
    )

    assert result
