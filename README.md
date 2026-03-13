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
| `EMBEDDING_URL`  | Local embedding API (default: :8001) |
| `EMBEDDING_MODEL`| Model name (default: BAAI/bge-m3)   |
| `VECTOR_SIZE`    | Embedding dim (default: 1024)  |

## Usage

```bash
python main.py --data-dir ./data --collection taixing_knowledge --env dev
```
