"""Offline integrity checks for a prepared canonical MultiCaRe dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from collections_config import COLLECTIONS
from manifest import MANIFEST_SCHEMA_VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(manifest_path: Path, source: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unexpected manifest schema version")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest contains no records")
    ids = set()
    assets = manifest_path.parent / "assets"
    for record in records:
        point_id = record.get("point_id")
        primary = record.get("primary_image", {})
        profile = record.get("profile", {})
        if not point_id or point_id in ids:
            raise ValueError(f"Missing or non-stable duplicate point_id: {point_id}")
        ids.add(point_id)
        if profile.get("profile_id") != f"{record.get('article_id')}:{primary.get('asset_id')}":
            raise ValueError(f"Profile identity mismatch for {point_id}")
        spec = COLLECTIONS.get(record.get("collection"))
        if not spec:
            raise ValueError(f"Unknown collection {record.get('collection')!r} for {point_id}")
        if (primary.get("image_type"), primary.get("image_subtype")) != (spec["image_type"], spec["image_subtype"]):
            raise ValueError(f"Image type does not match collection {record['collection']} for {point_id}")
        for image in [primary, *record.get("related_images", [])]:
            asset = assets / image.get("asset_id", "")
            if not asset.is_file():
                raise FileNotFoundError(f"Missing copied asset: {asset}")
            if image.get("sha256") and sha256(asset) != image["sha256"]:
                raise ValueError(f"Checksum mismatch: {asset}")
    stats = manifest.get("stats", {})
    if stats.get("record_count") != len(records):
        raise ValueError(f"Manifest stats says {stats.get('record_count')} records but contains {len(records)}")
    report = {"records": len(records), "unique_point_ids": len(ids), "assets_checked": len(list(assets.iterdir())),
              "by_collection": stats.get("by_collection"), "unavailable_extractions": stats.get("unavailable_extractions")}
    if source:
        import sys
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_pipeline"))
        from prepare_multicare import article_id, select_primaries
        captions = pd.read_csv(source / "captions_and_labels.csv", low_memory=False)
        captions["article_id"] = captions["patient_id"].map(article_id)
        expected = select_primaries(captions[captions["article_id"] != ""], list(stats["by_collection"]),
                                    stats["per_collection_articles"], stats["images_per_patient"],
                                    stats.get("collection_articles"))
        if len(expected) != len(records) + stats.get("missing_source_images", 0) + stats.get("zero_shot_rejected", 0):
            raise ValueError(f"Selection rule gives {len(expected)} images, manifest has {len(records)}")
        report["expected_source_images"] = len(expected)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("dataset/manifest.json"))
    parser.add_argument("--source", type=Path, default=None, help="Also re-run the deterministic selection and compare counts")
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, args.source), indent=2))


if __name__ == "__main__":
    main()
