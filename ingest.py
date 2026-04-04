"""
Qdrant collection setup and batch ingestion.
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, VectorParams

from config import BATCH_SIZE, COLLECTION_NAME, VECTOR_SIZE
from records import iter_records


def _collection_exists(client: QdrantClient, name: str) -> bool:
    exists = getattr(client, "collection_exists", None)
    if exists is not None:
        return bool(exists(name))
    return name in {c.name for c in client.get_collections().collections}


def ensure_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Create collection if it does not exist."""
    if _collection_exists(client, collection_name):
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
    """Stream records from data_dir. Overwrites points with matching source_file."""
    total = 0
    for source_file, points in iter_records(data_dir):
        if not points:
            continue
        # Delete existing points from this source_file
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            ),
        )
        # Upsert new points
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i : i + BATCH_SIZE]
            client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)
        print(f"  {source_file}: {len(points)} points")
    print(f"\n✅ Done. {total} points ingested into '{collection_name}'.")
