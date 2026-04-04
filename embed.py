"""Generate embeddings via local v1/embeddings API."""
import uuid

import httpx

from config import EMBEDDING_INTERNAL_KEY, EMBEDDING_MODEL, EMBEDDING_URL, VECTOR_SIZE


def _embed_headers() -> dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
        "X-Session-Id": str(uuid.uuid4()),
    }
    if EMBEDDING_INTERNAL_KEY:
        h["X-Internal-Key"] = EMBEDDING_INTERNAL_KEY
    return h


def embed_text(text: str) -> list[float]:
    """Embed a single text."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via local v1/embeddings API. Reuses HTTP client; batches if API accepts it."""
    if not texts:
        return []

    url = f"{EMBEDDING_URL.rstrip('/')}/v1/embeddings"
    model = EMBEDDING_MODEL or "BAAI/bge-m3"

    with httpx.Client(timeout=60.0) as client:
        payload = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
        resp = client.post(url, json=payload, headers=_embed_headers())
        if resp.status_code == 200:
            data = resp.json()
            embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda x: x["index"])]
        else:
            embeddings = []
            for text in texts:
                r = client.post(
                    url, json={"model": model, "input": text}, headers=_embed_headers()
                )
                r.raise_for_status()
                emb = r.json()["data"][0]["embedding"]
                embeddings.append(emb)

    for emb in embeddings:
        if len(emb) != VECTOR_SIZE:
            raise ValueError(
                f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}. "
                "Set VECTOR_SIZE in .env to match your embedding model."
            )
    return embeddings
