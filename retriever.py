"""
Thin wrapper around the FAISS vector store: load it once, and expose a
single retrieve() function that optionally filters results down to one
policy's chunks.
"""

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: it's our own local index
    )


def retrieve(
    vectorstore: FAISS,
    query: str,
    policy_name: Optional[str] = None,
    k: Optional[int] = None,
):
    """Return the top-k relevant chunks for the query.

    If policy_name is given, restrict the search to chunks whose
    metadata.policy_name matches exactly - this is what makes
    "search only that policy" actually happen.
    """
    k = k or config.TOP_K

    if policy_name:
        results = vectorstore.similarity_search(
            query,
            k=k,
            filter={"policy_name": policy_name},
        )
    else:
        results = vectorstore.similarity_search(query, k=k)

    return results
