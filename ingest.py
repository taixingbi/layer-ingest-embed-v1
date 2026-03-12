"""
Qdrant collection setup and batch ingestion.
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from config import BATCH_SIZE, COLLECTION_NAME, VECTOR_SIZE
from records import iter_records


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    *,
    recreate: bool = False,
) -> None:
    """Create the collection if it does not exist. Use recreate=True to delete and recreate."""
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        if recreate:
            print(f"Deleting collection '{collection_name}' …")
            client.delete_collection(collection_name)
            print("  ✓ Deleted.")
        else:
            print(f"Collection '{collection_name}' already exists — skipping creation.")
            return
    print(f"Creating collection '{collection_name}' …")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )
    print("  ✓ Collection created.")


def ingest(client: QdrantClient, data_dir: str, collection_name: str = COLLECTION_NAME) -> None:
    """Stream records from data_dir and upsert in batches."""
    batch: list[PointStruct] = []
    total = 0

    for point_id, embedding, payload in iter_records(data_dir):
        batch.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            )
        )
        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)
            print(f"  Upserted {total} points …")
            batch.clear()

    if batch:
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    print(f"\n✅ Done. {total} points ingested into '{collection_name}'.")
