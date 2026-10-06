"""Resumable, local-only indexing of canonical MultiCaRe twin-case manifest records.

Each Qdrant point carries two named vectors:
  image — MedSigLIP embedding of the case image (1152-d)
  text  — Qwen3 embedding of the structured case report (4096-d)
so search can match on what the image looks like, what the case says, or both.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx
from PIL import Image
from qdrant_client import QdrantClient, models

from embedding_service import generate_embedding
from local_ai import text_embeddings
from manifest import ASSET_ROOT, MANIFEST_PATH, MANIFEST_SCHEMA_VERSION, case_document

BATCH = 16


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(data.get("records"), list):
        raise ValueError(f"{path} is not a {MANIFEST_SCHEMA_VERSION} manifest")
    return data


def payload_for(record: dict) -> dict:
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "collection": record["collection"],
            "profile": record["profile"], "primary_asset_id": record["primary_image"]["asset_id"],
            "related_asset_ids": [image["asset_id"] for image in record.get("related_images", [])],
            "primary_image": record["primary_image"], "related_images": record.get("related_images", []),
            "raw_narrative": record.get("raw_narrative", ""), "raw_abstract": record.get("raw_abstract", ""),
            "enrichment_status": record.get("enrichment_status")}


def ensure_collection(client: QdrantClient, name: str, image_size: int, text_size: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(collection_name=name, vectors_config={
        "image": models.VectorParams(size=image_size, distance=models.Distance.COSINE),
        "text": models.VectorParams(size=text_size, distance=models.Distance.COSINE),
    })
    client.create_payload_index(name, field_name="collection", field_schema=models.PayloadSchemaType.KEYWORD)


def index_manifest(manifest_path: Path = MANIFEST_PATH, collection: str | None = None,
                   limit: int | None = None, passes: int = 3) -> tuple[int, int]:
    manifest = load_manifest(manifest_path)
    records = manifest["records"][:limit] if limit else manifest["records"]
    if not records:
        raise ValueError("Manifest has no records")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"), api_key=os.getenv("QDRANT_API_KEY") or None)
    collection = collection or os.getenv("COLLECTION_NAME", "multicare_cases")
    indexed = 0
    pending = records
    for attempt in range(1, passes + 1):
        failed: list[dict] = []
        for start in range(0, len(pending), BATCH):
            batch = pending[start:start + BATCH]
            if client.collection_exists(collection):
                existing = {str(point.id) for point in client.retrieve(collection, ids=[r["point_id"] for r in batch],
                                                                         with_payload=False, with_vectors=False)}
                batch = [r for r in batch if r["point_id"] not in existing]
            points = []
            for record in batch:
                try:
                    text_vector = text_embeddings([case_document(record["profile"], record.get("raw_narrative", ""))])[0]
                    image_path = ASSET_ROOT / record["primary_image"]["asset_id"]
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Manifest asset is missing: {image_path}")
                    with Image.open(image_path) as image:
                        image_vector = generate_embedding(image)
                except (ValueError, httpx.HTTPError) as exc:
                    # Transient embedding-server faults: retry this record in the next pass.
                    print(f"Deferred {record['point_id']}: {exc}", flush=True)
                    failed.append(record)
                    continue
                ensure_collection(client, collection, len(image_vector), len(text_vector))
                points.append(models.PointStruct(id=record["point_id"], vector={"image": image_vector, "text": text_vector},
                                                 payload=payload_for(record)))
            if points:
                client.upsert(collection_name=collection, points=points)
                indexed += len(points)
            print(f"Pass {attempt}: {min(start + BATCH, len(pending))}/{len(pending)} checked, {indexed} indexed", flush=True)
        if not failed:
            break
        pending = failed
    else:
        raise RuntimeError(f"{len(failed)} records could not be embedded after {passes} passes")
    total = client.get_collection(collection).points_count
    return indexed, total - indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    indexed, skipped = index_manifest(args.manifest, args.collection, args.limit)
    print(f"Indexed {indexed}; resumed {skipped} existing points.")


if __name__ == "__main__":
    main()
