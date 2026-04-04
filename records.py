"""Stream (source_file, points) from JSON files. Embeds on-the-fly when missing."""
import hashlib
import json
from pathlib import Path

import httpx
from qdrant_client.http.models import PointStruct

from config import VECTOR_SIZE
from embed import embed_text, embed_texts
from payload import build_payload
from text_utils import embedding_text

# Embedding API batch size (independent of Qdrant upsert BATCH_SIZE).
_EMBED_BATCH = 32


def _try_parse_jsonl(path: Path) -> list[dict] | None:
    """Stream-parse JSONL; return None if the file is not line-delimited JSON."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return None
    return records


def _parse_json_file(path: Path) -> list[dict]:
    """Parse JSON or JSONL file into list of records."""
    as_jsonl = _try_parse_jsonl(path)
    if as_jsonl is not None:
        return as_jsonl

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("qa_pairs", "items", "records", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def _embed_texts_resilient(
    texts: list[str], client: httpx.Client
) -> list[list[float] | None]:
    """Batch embed; fall back per-text so one bad row does not drop a whole batch."""
    try:
        return embed_texts(texts, client=client)
    except Exception:
        out: list[list[float] | None] = []
        for t in texts:
            try:
                out.append(embed_text(t, client=client))
            except Exception:
                out.append(None)
        return out


def iter_records(data_dir: str):
    """Yield (source_file, [PointStruct, ...]) for each *.json / *.jsonl file."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Create it, add *.json or *.jsonl files, or pass --data-dir."
        )
    paths = sorted(
        set(root.glob("*.json")) | set(root.glob("*.jsonl")),
        key=lambda p: p.name.lower(),
    )
    if not paths:
        raise FileNotFoundError(
            f"No *.json or *.jsonl files in: {data_dir}"
        )

    for path in paths:
        print(f"  Reading {path.name} …")
        try:
            records = _parse_json_file(path)
        except json.JSONDecodeError:
            print(f"    ⚠ Skipping {path.name}: invalid JSON")
            continue

        source_file = path.name
        record_type = path.stem
        prepared: list[tuple[dict, str, list[float] | None]] = []
        for record in records:
            text = embedding_text(record)
            emb = record.get("embedding")
            if isinstance(emb, list) and len(emb) == VECTOR_SIZE:
                prepared.append((record, text, emb))
            else:
                prepared.append((record, text, None))

        to_embed_pos: list[int] = []
        to_embed_texts: list[str] = []
        for pos, (_r, text, e) in enumerate(prepared):
            if e is None:
                to_embed_pos.append(pos)
                to_embed_texts.append(text)

        with httpx.Client(timeout=60.0) as http_client:
            for start in range(0, len(to_embed_texts), _EMBED_BATCH):
                batch_t = to_embed_texts[start : start + _EMBED_BATCH]
                batch_embs = _embed_texts_resilient(batch_t, http_client)
                for j, row_emb in enumerate(batch_embs):
                    if row_emb is None:
                        continue
                    pos = to_embed_pos[start + j]
                    rec, text, _ = prepared[pos]
                    prepared[pos] = (rec, text, row_emb)

        points = []
        for i, (record, text, embedding) in enumerate(prepared):
            if embedding is None:
                print(f"    ⚠ Skipping row {i}: no embedding")
                continue
            stable = str(record.get("id") or f"{source_file}_{i}")
            point_id = int(hashlib.md5(stable.encode()).hexdigest()[:16], 16)
            payload = build_payload(
                record,
                source_file=source_file,
                record_type=record_type,
                text_for_payload=text,
            )
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
        if points:
            yield source_file, points
