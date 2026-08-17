"""Application configuration for the Coforge India HR Policy Assistant POC."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    try:
        raw = ENV_PATH.read_text(encoding="utf-8-sig")
        if raw.startswith("\ufeff"):
            ENV_PATH.write_text(raw.lstrip("\ufeff"), encoding="utf-8")
    except Exception:
        pass

load_dotenv(ENV_PATH, override=False)

POLICY_DOCS_DIR = BASE_DIR / "data" / "policies"
INDEX_DIR = BASE_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RETRIEVAL_TOP_K = 15
RERANK_TOP_N = 6

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_RETRY_COUNT = int(os.getenv("GEMINI_RETRY_COUNT", "2"))
GEMINI_BACKOFF_SECONDS = float(os.getenv("GEMINI_BACKOFF_SECONDS", "1.0"))
GEMINI_BACKOFF_MAX_SECONDS = float(os.getenv("GEMINI_BACKOFF_MAX_SECONDS", "8.0"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

POLICY_MATCH_THRESHOLD = 0.55
