import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import fitz  # PyMuPDF
import numpy as np

import config
from src.chunking import chunk_policy_document
from src.embeddings import embed_texts


BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = config.POLICY_DOCS_DIR
INDEX_DIR = config.INDEX_DIR


def discover_pdfs() -> List[Path]:
    if not POLICY_DIR.exists():
        return []
    return sorted(POLICY_DIR.glob("*.pdf"))


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text")
        cleaned = text.replace("\x00", "").replace("\ufeff", "")
        if cleaned.strip():
            pages.append({"page": page_number + 1, "text": cleaned})
    doc.close()
    return pages


def build_faiss_index(chunks: List[Dict[str, Any]]) -> None:
    texts = [chunk.get("text", "") for chunk in chunks if chunk.get("text")]
    if not texts:
        raise ValueError("No text chunks were generated during ingestion.")

    embeddings = embed_texts(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))


def build_indexes() -> Dict[str, Any]:
    pdfs = discover_pdfs()
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {POLICY_DIR}. Add India HR policy PDFs there and rebuild the index.")

    chunks: List[Dict[str, Any]] = []
    for pdf_path in pdfs:
        policy_name = pdf_path.stem.replace("-", " ").replace("_", " ").title()
        pages = extract_pdf_pages(pdf_path)
        page_chunks = chunk_policy_document(
            policy_name,
            pdf_path.name,
            pages,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )
        chunks.extend(page_chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    bm25_payload = {
        "total": len(chunks),
        "policies": sorted({chunk["policy_name"] for chunk in chunks}),
    }
    (INDEX_DIR / "bm25.pkl").write_bytes(json.dumps(bm25_payload, ensure_ascii=False).encode("utf-8"))

    build_faiss_index(chunks)

    return {
        "pdf_count": len(pdfs),
        "chunk_count": len(chunks),
        "policies": sorted({chunk["policy_name"] for chunk in chunks}),
    }


def main() -> None:
    try:
        result = build_indexes()
        print(f"Indexed {result['pdf_count']} PDFs into {result['chunk_count']} chunks.")
        for policy in result["policies"]:
            print(f"- {policy}")
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
