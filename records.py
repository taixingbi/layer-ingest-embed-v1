"""Stream (source_file, points) from JSON files. Embeds on-the-fly when missing."""
import hashlib
import json
from pathlib import Path

from qdrant_client.http.models import PointStruct

from config import VECTOR_SIZE
from embed import embed_text
from payload import build_payload
from text_utils import record_to_text


def _parse_json_file(path: Path) -> list[dict]:
    """Parse JSON or JSONL file into list of records."""
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    # Try JSONL first
    records = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records = []
            break

    if not records:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ("qa_pairs", "items", "records", "data"):
                if isinstance(data.get(key), list):
                    records = data[key]
                    break
            else:
                records = [data]

    return records


def iter_records(data_dir: str):
    """Yield (source_file, [PointStruct, ...]) for each *.json file."""
    paths = sorted(Path(data_dir).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON files in: {data_dir}")

    for path in paths:
        print(f"  Reading {path.name} …")
        try:
            records = _parse_json_file(path)
        except json.JSONDecodeError:
            print(f"    ⚠ Skipping {path.name}: invalid JSON")
            continue

        source_file = path.name
        record_type = path.stem
        points = []
        for i, record in enumerate(records):
            embedding = record.get("embedding")
            if not embedding or len(embedding) != VECTOR_SIZE:
                try:
                    embedding = embed_text(record_to_text(record))
                except Exception as e:
                    print(f"    ⚠ Skipping: {e}")
                    continue
            point_id = int(hashlib.md5(f"{source_file}_{i}".encode()).hexdigest()[:16], 16)
            payload = build_payload(
                record, source_file=source_file, record_type=record_type
            )
            points.append(
                PointStruct(id=point_id, vector=embedding, payload=payload)
            )
        if points:
            yield source_file, points
