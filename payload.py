"""Build Qdrant payload from chunk records (id, text, category, tags, metadata)."""
from text_utils import chunk_text, truncate_for_payload


def build_payload(
    record: dict,
    *,
    source_file: str = "",
    record_type: str = "",
    text_for_payload: str | None = None,
) -> dict:
    """Payload matches data pattern + ingest fields (source_file, record_type)."""
    text_stored = (
        truncate_for_payload(text_for_payload)
        if text_for_payload is not None
        else chunk_text(record)
    )
    payload: dict = {
        "text": text_stored,
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
