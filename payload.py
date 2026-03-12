"""Build searchable payload from records."""
from config import PAYLOAD_META_KEYS
from text_utils import record_to_text


def build_payload(
    record: dict,
    *,
    source_file: str = "",
    record_type: str = "",
) -> dict:
    """Extract searchable metadata for Qdrant payload."""
    features = record.get("wit_features", [])
    en_feature = next(
        (f for f in features if f.get("language") == "en"),
        features[0] if features else {},
    )

    payload = {
        "text": record_to_text(record)[:2000],
        "page_title": (
            en_feature.get("page_title")
            or record.get("q")
            or (record.get("profile") or {}).get("name")
            or ""
        ),
        "page_url": en_feature.get("page_url", ""),
        "caption": en_feature.get("caption_attribution_description", ""),
        "description": en_feature.get("context_page_description", ""),
        "source_file": source_file,
        "record_type": record_type,
    }

    if "q" in record and "a" in record:
        payload["q"] = record["q"]
        payload["a"] = record["a"]

    for key in PAYLOAD_META_KEYS:
        val = record.get(key)
        if val is None and key == "contact":
            val = (record.get("profile") or {}).get("contact")
        if val is not None:
            payload[key] = val

    return payload
