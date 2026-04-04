"""Build Qdrant payload from chunk records (id, text, category, tags, metadata)."""
from text_utils import chunk_text


def build_payload(
    record: dict,
    *,
    source_file: str = "",
    record_type: str = "",
) -> dict:
    """Payload matches data pattern + ingest fields (source_file, record_type)."""
    payload: dict = {
        "text": chunk_text(record),
        "source_file": source_file,
        "record_type": record_type,
    }
    for key in ("id", "category", "tags"):
        val = record.get(key)
        if val is not None:
            payload[key] = val
    meta = record.get("metadata")
    if meta is not None:
        payload["metadata"] = meta
    return payload
