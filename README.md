# RAG Ingest

Ingest JSON data into Qdrant. Supports JSON and JSONL, embeds on-the-fly via local v1/embeddings API when records lack embeddings.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `.env`:

| Variable         | Description                    |
|------------------|--------------------------------|
| `QDRANT_URL`     | Qdrant URL (default: local)    |
| `QDRANT_API_KEY` | API key for Qdrant Cloud      |
| `EMBEDDING_URL`         | Embedding API base URL                    |
| `EMBEDDING_MODEL`       | Model name (default: BAAI/bge-m3)         |
| `EMBEDDING_INTERNAL_KEY`| Sent as `X-Internal-Key` if set (often required) |
| `VECTOR_SIZE`           | Embedding dim (default: 1024)             |

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