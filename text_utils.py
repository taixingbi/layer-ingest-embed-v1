"""Text extraction from records for embedding and payload."""

SKIP_KEYS = ("metadata", "schema_version", "generated_on", "source")


def record_to_text(record: dict) -> str:
    """Flatten record to searchable text."""
    return _flatten(record)


def _flatten(obj, skip_keys=SKIP_KEYS):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "q" in obj and "a" in obj:
            return f"Q: {obj['q']}\nA: {obj['a']}"
        parts = []
        for k, v in obj.items():
            if k in skip_keys:
                continue
            if isinstance(v, (dict, list)):
                parts.append(f"{k}: {_flatten(v, skip_keys)}")
            elif v is not None:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(_flatten(x, skip_keys) for x in obj)
    return str(obj)
