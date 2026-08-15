"""
Ingestion pipeline.

Run this once (and again whenever policy docs change):

    python ingest.py

What it does:
  1. Reads every PDF / DOCX in data/policies/
  2. Splits each into overlapping text chunks
  3. Tags every chunk with metadata: policy_name + source_file
  4. Embeds the chunks and persists a FAISS index to vectorstore/

The "policy_name" is derived from the filename, so name your files
after the policy, e.g.:

    data/policies/leave-policy.pdf          -> "Leave Policy"
    data/policies/work-from-home-policy.pdf -> "Work From Home Policy"
    data/policies/pf-and-gratuity-policy.pdf -> "Pf And Gratuity Policy"
"""

import sys

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config


def filename_to_policy_name(filename: str) -> str:
    """'leave-policy.pdf' -> 'Leave Policy'"""
    stem = filename.rsplit(".", 1)[0]
    words = stem.replace("_", "-").split("-")
    return " ".join(w.capitalize() for w in words)


def load_documents():
    """Load every supported file in data/policies/ into LangChain Documents,
    stamping each with its policy name."""
    docs = []

    if not config.POLICY_DOCS_DIR.exists():
        print(f"No policy folder found at {config.POLICY_DOCS_DIR}")
        return docs

    files = sorted(config.POLICY_DOCS_DIR.iterdir())
    if not files:
        print(f"No files found in {config.POLICY_DOCS_DIR}. "
              f"Add the downloaded India HR policy PDFs there first.")
        return docs

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        policy_name = filename_to_policy_name(file_path.name)
        print(f"Loading '{file_path.name}' -> policy '{policy_name}'")

        for doc in loader.load():
            doc.metadata["policy_name"] = policy_name
            doc.metadata["source_file"] = file_path.name
            docs.append(doc)

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_and_persist_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(config.VECTORSTORE_DIR))
    print(f"Saved FAISS index with {len(chunks)} chunks -> {config.VECTORSTORE_DIR}")


def main():
    docs = load_documents()
    if not docs:
        print("Nothing to ingest. Aborting.")
        sys.exit(1)

    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} document(s), split into {len(chunks)} chunks.")

    policy_names = sorted({c.metadata["policy_name"] for c in chunks})
    print("Policies detected:")
    for name in policy_names:
        print(f"  - {name}")

    build_and_persist_index(chunks)


if __name__ == "__main__":
    main()
