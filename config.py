"""
Configuration for RAG ingest (Qdrant URL, API key, collection and data paths).
Loads .env if present; values can be overridden by environment variables.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.86.173:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Embedding (local v1/embeddings API)
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://192.168.86.173:8001")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Defaults for CLI (--collection, --data-dir)
COLLECTION_NAME = "rag_dev"
DATA_DIR = "./data"

# Vector and batching (BAAI/bge-m3 outputs 1024)
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1024"))
BATCH_SIZE = 20

# Keys to copy from record into payload when present
PAYLOAD_META_KEYS = ("metadata", "contact", "source")
