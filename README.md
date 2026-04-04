# RAG Ingest

Ingest JSON data into Qdrant. Supports JSON and JSONL, embeds on-the-fly via local v1/embeddings API when records lack embeddings.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `.env` with **all** of the following (no defaults in code; startup fails if any required variable is missing or empty):

| Variable                | Description                                      |
|-------------------------|--------------------------------------------------|
| `QDRANT_URL`            | Qdrant URL                                       |
| `QDRANT_API_KEY`        | Optional; omit or leave empty for local Qdrant   |
| `EMBEDDING_URL`         | Embedding API base URL                           |
| `EMBEDDING_MODEL`       | e.g. `BAAI/bge-m3`                               |
| `EMBEDDING_INTERNAL_KEY`| Sent as `X-Internal-Key`                         |
| `COLLECTION_NAME`       | Default collection (CLI `--collection` overrides) |
| `DATA_DIR`              | Default data directory (CLI `--data-dir` overrides) |
| `VECTOR_SIZE`           | Embedding dimension (integer)                    |
| `BATCH_SIZE`            | Qdrant upsert batch size (integer)               |

Each embedding HTTP call sends fresh random UUIDs for `X-Request-Id` and `X-Session-Id`.

## Usage

```bash
python main.py --data-dir ./data --collection taixing_knowledge --env dev
```

## Data pattern

Each `*.json` file is an array of chunks (or JSONL with one chunk per line). Required fields:

```json
{
  "id": "stable_unique_id",
  "text": "the actual chunk text that gets embedded",
  "category": "...",
  "tags": ["...", "..."],
  "metadata": {
    "source": "...",
    "date": "...",
    "author": "..."
  }
}
```

`metadata` may include extra keys (e.g. `keywords`, `priority`, `search_keywords`). Qdrant payload adds `source_file` and `record_type` (from the filename).