"""
RAG Ingest: JSON data → Qdrant
==============================
Reads *.json from a data directory, embeds records on-the-fly (or uses pre-computed
embeddings), and upserts into a Qdrant collection.
"""
import argparse

from dotenv import load_dotenv

# Load --env before config so .env.{env} overrides
_load_parser = argparse.ArgumentParser()
_load_parser.add_argument("--env")
_load_args, _ = _load_parser.parse_known_args()
load_dotenv()
if _load_args.env:
    load_dotenv(f".env.{_load_args.env}")

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from config import COLLECTION_NAME, DATA_DIR, QDRANT_API_KEY, QDRANT_URL
from ingest import ensure_collection, ingest


def run(*, data_dir: str, collection_name: str) -> None:
    print("Connecting to Qdrant …")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        check_compatibility=False,
    )

    try:
        ensure_collection(client, collection_name=collection_name)
    except UnexpectedResponse as e:
        if e.status_code == 403:
            print("Error: 403 Forbidden — Qdrant rejected the request.")
            print("  • Set QDRANT_API_KEY in .env for Qdrant Cloud.")
            raise SystemExit(1) from e
        raise

    print(f"\nIngesting from '{data_dir}' …")
    ingest(client, data_dir, collection_name=collection_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest JSON into Qdrant with embeddings.")
    parser.add_argument("--data-dir", default=DATA_DIR, help="Directory with *.json files")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--env", help="Environment (loads .env.{env}, appends to collection name)")
    args, _ = parser.parse_known_args()
    collection_name = args.collection
    if args.env:
        collection_name = f"{args.collection}_{args.env}"
    run(data_dir=args.data_dir, collection_name=collection_name)
