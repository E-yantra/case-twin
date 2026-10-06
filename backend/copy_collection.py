"""Copy the twin-case collection between Qdrant servers without re-embedding.

Points are streamed with their vectors and payloads through the HTTP API, so it
works across Qdrant versions (unlike snapshots, which may not restore on an
older server). The target collection is created with the source's vector config
and payload index; existing point IDs are overwritten, so re-running is safe.

    python copy_collection.py --source http://localhost:6333 --target http://<server>:6333
"""

from __future__ import annotations

import argparse

from qdrant_client import QdrantClient, models


def copy_collection(source: QdrantClient, target: QdrantClient, name: str, target_name: str, batch: int = 64) -> int:
    info = source.get_collection(name)
    if not target.collection_exists(target_name):
        target.create_collection(collection_name=target_name, vectors_config=info.config.params.vectors)
        for field, schema in (info.payload_schema or {}).items():
            target.create_payload_index(target_name, field_name=field, field_schema=schema.data_type)
    copied, offset = 0, None
    while True:
        points, offset = source.scroll(name, limit=batch, offset=offset, with_payload=True, with_vectors=True)
        if points:
            target.upsert(target_name, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
            copied += len(points)
            print(f"Copied {copied}/{info.points_count}", flush=True)
        if offset is None:
            break
    if target.get_collection(target_name).points_count != info.points_count:
        raise RuntimeError("Point count mismatch after copy")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--collection", default="multicare_cases")
    parser.add_argument("--target-collection", default=None)
    parser.add_argument("--source-api-key", default=None)
    parser.add_argument("--target-api-key", default=None)
    args = parser.parse_args()
    source = QdrantClient(url=args.source, api_key=args.source_api_key)
    target = QdrantClient(url=args.target, api_key=args.target_api_key)
    copied = copy_collection(source, target, args.collection, args.target_collection or args.collection)
    print(f"Done: {copied} points")


if __name__ == "__main__":
    main()
