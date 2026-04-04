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


def _order_embeddings(items: list[dict], n: int) -> list[list[float]]:
    """Restore embedding order from API response (O(n) when indices are present)."""
    if len(items) != n:
        return [e["embedding"] for e in sorted(items, key=lambda x: x["index"])]
    try:
        ordered: list[list[float] | None] = [None] * n
        for e in items:
            ordered[e["index"]] = e["embedding"]
        if any(x is None for x in ordered):
            raise ValueError
        return ordered  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError, IndexError):
        return [e["embedding"] for e in sorted(items, key=lambda x: x["index"])]


def _embed_texts_with_client(texts: list[str], client: httpx.Client) -> list[list[float]]:
    url = f"{EMBEDDING_URL.rstrip('/')}/v1/embeddings"
    model = EMBEDDING_MODEL or "BAAI/bge-m3"
    payload = {"model": model, "input": texts if len(texts) > 1 else texts[0]}
    resp = client.post(url, json=payload, headers=_embed_headers())
    if resp.status_code == 200:
        data = resp.json()
        embeddings = _order_embeddings(data["data"], len(texts))
    else:
        embeddings = []
        for text in texts:
            r = client.post(url, json={"model": model, "input": text}, headers=_embed_headers())
            r.raise_for_status()
            embeddings.append(r.json()["data"][0]["embedding"])

    for emb in embeddings:
        if len(emb) != VECTOR_SIZE:
            raise ValueError(
                f"Embedding dim {len(emb)} != VECTOR_SIZE {VECTOR_SIZE}. "
                "Set VECTOR_SIZE in .env to match your embedding model."
            )
    return embeddings


def embed_text(text: str, *, client: httpx.Client | None = None) -> list[float]:
    """Embed a single text."""
    return embed_texts([text], client=client)[0]


def embed_texts(texts: list[str], *, client: httpx.Client | None = None) -> list[list[float]]:
    """Embed texts via local v1/embeddings API. Pass ``client`` to reuse connections across calls."""
    if not texts:
        return []
    if client is not None:
        return _embed_texts_with_client(texts, client)
    with httpx.Client(timeout=60.0) as c:
        return _embed_texts_with_client(texts, c)
