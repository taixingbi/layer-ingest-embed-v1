"""
Configuration for RAG ingest (Qdrant URL, API key, collection and data paths).
Loads .env if present; values can be overridden by environment variables.
All settings below must be set in the environment (no built-in defaults).
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _require_str(name: str) -> str:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        raise RuntimeError(
            f"Missing or empty environment variable: {name}. "
            "Set it in .env or export it in the shell."
        )
    return str(v).strip()


def _require_int(name: str) -> int:
    raw = _require_str(name)
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(f"Environment variable {name} must be an integer, got {raw!r}") from e


# Qdrant
QDRANT_URL = _require_str("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or ""

# Embedding (local v1/embeddings API)
EMBEDDING_URL = _require_str("EMBEDDING_URL")
EMBEDDING_MODEL = _require_str("EMBEDDING_MODEL")
EMBEDDING_INTERNAL_KEY = _require_str("EMBEDDING_INTERNAL_KEY")

# CLI defaults (still required via .env)
COLLECTION_NAME = _require_str("COLLECTION_NAME")
DATA_DIR = _require_str("DATA_DIR")

VECTOR_SIZE = _require_int("VECTOR_SIZE")
BATCH_SIZE = _require_int("BATCH_SIZE")
