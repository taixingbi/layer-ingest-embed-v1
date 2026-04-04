"""Text for embedding from chunk records."""

MAX_PAYLOAD_TEXT = 2000


def truncate_for_payload(raw: str) -> str:
    """Truncate text stored on Qdrant payload."""
    return raw[:MAX_PAYLOAD_TEXT] if len(raw) > MAX_PAYLOAD_TEXT else raw


def chunk_text(record: dict) -> str:
    """Return embeddable / stored text (truncated for payload)."""
    return truncate_for_payload(embedding_text(record))


def embedding_text(record: dict) -> str:
    """Full text used for embedding (must match chunk schema or fall back to flatten)."""
    if isinstance(record, dict):
        t = record.get("text")
        if isinstance(t, str) and t.strip():
            return t.strip()
    return _flatten_fallback(record)


def _flatten_fallback(obj) -> str:
    """JSONL / legacy dicts without top-level ``text``."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        skip = ("metadata", "embedding")
        parts = []
        for k, v in obj.items():
            if k in skip:
                continue
            if isinstance(v, (dict, list)):
                parts.append(f"{k}: {_flatten_fallback(v)}")
            elif v is not None:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_flatten_fallback(x) for x in obj)
    return str(obj)
