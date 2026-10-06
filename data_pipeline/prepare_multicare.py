"""Create the reproducible, derived MultiCaRe twin-case dataset.

The source tree is read-only. This command writes only ``dataset/``:
assets (content-addressed image copies), manifest.json, and enrichment-cache.

The library is split into collections that match MedSigLIP's training domains
(chest X-ray, chest CT, dermatology, fundus, H&E histopathology). Each
collection takes a deterministic sample of articles, so the same command always
produces the same slice. Every patient's case text is turned into a structured
case report by Gemma 4 using the same schema as the live intake
(``backend/extraction.py``). Re-runs reuse cached extractions and assets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from collections_config import COLLECTIONS, quality_labels  # noqa: E402
from local_ai import medsiglip_classify  # noqa: E402
from extraction import article_prompt, extract_profile  # noqa: E402
from manifest import MANIFEST_SCHEMA_VERSION, empty_profile, merge_profile  # noqa: E402

SOURCE_DEFAULT = ROOT / "medical_datasets" / "whole_multicare_dataset"
DATASET_DEFAULT = ROOT / "dataset"
MAX_RELATED_IMAGES = 8
EXTRACTION_VERSION = "gemma4-schema-v2"
# Which OpenAI-compatible server structures the library (set from the CLI).
# None/None = Gemma 4 on the shared gateway; cached profiles are reused whichever
# model produced them, and each records its model in ``extracted_by``.
EXTRACT_MODEL: str | None = None
EXTRACT_BASE_URL: str | None = None


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def as_list(value: Any) -> list:
    value = jsonable(value)
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value.replace("'", '"'))
        except ValueError:
            return [value]
    return value if isinstance(value, list) else []


def source_image_path(source: Path, filename: str) -> Path:
    # MultiCaRe's documented nesting is PMC1/PMC10/<filename>.
    filename = Path(str(filename)).name
    return source / filename[:4] / filename[:5] / filename


def article_id(value: Any) -> str:
    match = re.search(r"PMC\d+", str(value))
    return match.group(0) if match else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def copy_asset(source_file: Path, assets_dir: Path) -> tuple[str, int, str]:
    checksum = sha256_file(source_file)
    suffix = source_file.suffix.lower()
    asset_id = f"{checksum}{suffix}" if suffix else checksum
    destination = assets_dir / asset_id
    if destination.exists():
        if sha256_file(destination) != checksum:
            raise RuntimeError(f"Existing derived asset failed integrity check: {destination}")
    else:
        # A distinct temporary filename makes concurrent workers safe even
        # when two source rows reference the same content-addressed asset.
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            shutil.copy2(source_file, temporary)
            if sha256_file(temporary) != checksum:
                raise RuntimeError(f"Copy integrity check failed: {source_file}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return asset_id, source_file.stat().st_size, checksum


def enrich_patient(patient: str, abstract: str, narrative: str, cache_dir: Path) -> tuple[dict, str, str | None]:
    """Structured extraction for one patient, cached by source hash. Returns (profile, status, model)."""
    cache_file = cache_dir / f"{patient}.json"
    source_hash = hashlib.sha256(f"{EXTRACTION_VERSION}\n{abstract}\n{narrative}".encode()).hexdigest()
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("source_hash") == source_hash and cached.get("status") == "complete":
            # Caches written before model tagging all came from Gemma 4.
            return cached["profile"], "complete", cached.get("extracted_by", "gemma-4")
    model = None
    try:
        profile, step = extract_profile(article_prompt(abstract, narrative[:12000]), attempts=3,
                                        model=EXTRACT_MODEL, base_url=EXTRACT_BASE_URL, timeout=900.0)
        status, error, model = "complete", None, step["model"]
    except Exception as exc:  # noqa: BLE001 - one failed article must not stop the batch
        # The source narrative is retained in every record. An empty structured
        # profile is safer than a hallucinated substitute.
        profile, status, error = {}, "unavailable", str(exc)
    temporary = cache_file.with_name(f".{cache_file.name}.{uuid.uuid4().hex}.part")
    try:
        temporary.write_text(json.dumps({"source_hash": source_hash, "profile": profile, "status": status,
                                         "error": error, "extracted_by": model}, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        temporary.replace(cache_file)
    finally:
        temporary.unlink(missing_ok=True)
    return profile, status, model


def zero_shot_check(source_file: Path, collection: str, cache_dir: Path) -> dict:
    """MedSigLIP zero-shot check that an image really belongs to its collection (cached)."""
    cache_file = cache_dir / "quality" / f"{source_file.stem}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    from PIL import Image
    with Image.open(source_file) as image:
        ranked = medsiglip_classify(image, quality_labels())
    expected = COLLECTIONS[collection]["zero_shot"]
    result = {"collection": collection, "top_label": ranked[0]["label"], "keep": ranked[0]["label"] == expected,
              "scores": ranked[:3]}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result), encoding="utf-8")
    return result


def image_info(row: pd.Series, source: Path, assets_dir: Path) -> dict | None:
    source_file = source_image_path(source, str(row["file"]))
    if not source_file.is_file():
        return None
    asset_id, size, checksum = copy_asset(source_file, assets_dir)
    return {
        "asset_id": asset_id,
        "source_filename": Path(str(row["file"])).name,
        "caption": str(row.get("caption") or ""),
        "license": str(row.get("license") or ""),
        "image_type": str(row.get("image_type") or ""),
        "image_subtype": str(row.get("image_subtype") or ""),
        "radiology_region": jsonable(row.get("radiology_region")),
        "radiology_view": jsonable(row.get("radiology_view")),
        "labels": {"ml": as_list(row.get("ml_labels_for_supervised_classification")),
                   "gt": as_list(row.get("gt_labels_for_semisupervised_classification"))},
        "bytes": size,
        "sha256": checksum,
    }


def records_for_patient(patient: str, primaries: pd.DataFrame, patient_images: pd.DataFrame, source: Path,
                        assets_dir: Path, cache_dir: Path, metadata: dict, abstracts: dict,
                        cases: dict) -> tuple[list[dict], int, str, list[dict]]:
    """Prepare one patient independently so it can be safely run by a worker."""
    article = article_id(patient)
    case = cases.get(patient, {})
    narrative = str(case.get("case_text") or "")
    # Cheap MedSigLIP check first: a patient with no valid image never reaches the
    # library, so it must not cost an LLM extraction.
    kept_rows, rejected, missing = [], [], 0
    for _, row in primaries.sort_values("file_id").iterrows():
        source_file = source_image_path(source, str(row["file"]))
        if not source_file.is_file():
            missing += 1
            continue
        quality = zero_shot_check(source_file, row["collection"], cache_dir)
        if quality["keep"]:
            kept_rows.append((row, quality))
        else:
            rejected.append({"file": source_file.name, "collection": row["collection"], "top_label": quality["top_label"]})
    if not kept_rows:
        return [], missing, "skipped-no-valid-image", rejected
    patient_profile, status, extracted_by = enrich_patient(patient, abstracts.get(article, ""), narrative, cache_dir)
    primary_ids = set(primaries["file_id"].astype(str))
    related: list[dict] = []
    for _, row in patient_images.sort_values("file_id").iterrows():
        if str(row["file_id"]) in primary_ids or len(related) >= MAX_RELATED_IMAGES:
            continue
        info = image_info(row, source, assets_dir)
        if info:
            related.append(info)
        else:
            missing += 1
    records: list[dict] = []
    for row, quality in kept_rows:
        primary = image_info(row, source, assets_dir)
        if not primary:
            missing += 1
            continue
        spec = COLLECTIONS[row["collection"]]
        base = empty_profile(article_id=article, image_id=primary["asset_id"], provenance=metadata.get(article, {}),
                             modality=spec["modality"], body_region=spec["body_region"])
        profile = merge_profile(base, patient_profile)
        # Image-level facts come from the dataset, not from the model.
        profile["case_id"] = patient
        profile["study"].update({"collection": row["collection"], "modality": spec["modality"],
                                 "body_region": spec["body_region"], "caption": primary["caption"],
                                 "view_position": primary["radiology_view"], "radiology_region": primary["radiology_region"],
                                 "image_type": primary["image_type"], "image_subtype": primary["image_subtype"],
                                 "storage_path": primary["asset_id"]})
        profile["patient"]["age_years"] = profile["patient"]["age_years"] if profile["patient"]["age_years"] is not None else case.get("age")
        if not profile["patient"]["sex"] and str(case.get("gender", "")).lower() in {"male", "female"}:
            profile["patient"]["sex"] = str(case["gender"]).lower()
        profile["provenance"]["license"] = primary["license"] or profile["provenance"].get("license")
        profile["tags"].update({"ml_labels": primary["labels"]["ml"], "gt_labels": primary["labels"]["gt"]})
        records.append({"point_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"multicare:{patient}:{row['file_id']}")),
                        "article_id": article, "patient_id": patient, "collection": row["collection"],
                        "source_file_id": str(row["file_id"]), "primary_image": primary, "related_images": related,
                        "profile": profile, "raw_abstract": abstracts.get(article, ""), "raw_narrative": narrative,
                        "enrichment_status": status, "extracted_by": extracted_by, "zero_shot": quality})
    return records, missing, status, rejected


def select_primaries(captions: pd.DataFrame, collections: list[str], per_collection: int,
                     images_per_patient: int, overrides: dict[str, int] | None = None) -> pd.DataFrame:
    """Deterministic sample: articles ranked by hash, at most N images per patient.

    Hash ranking means a larger N always contains the smaller sample, so growing a
    collection reuses every cached extraction.
    """
    selected = []
    for name in collections:
        count = (overrides or {}).get(name, per_collection)
        spec = COLLECTIONS[name]
        mask = (captions["image_type"] == spec["image_type"]) & (captions["image_subtype"] == spec["image_subtype"])
        if spec.get("radiology_region"):
            mask &= captions["radiology_region"] == spec["radiology_region"]
        rows = captions[mask].copy()
        rows["collection"] = name
        articles = sorted(rows["article_id"].unique(), key=stable_rank)[:count]
        rows = rows[rows["article_id"].isin(set(articles))].sort_values("file_id")
        selected.append(rows.groupby("patient_id", sort=False).head(images_per_patient))
    frame = pd.concat(selected, ignore_index=True)
    # An image belongs to the first collection that claims it.
    return frame.drop_duplicates("file_id", keep="first")


def build(source: Path, output: Path, collections: list[str], per_collection: int, images_per_patient: int,
          workers: int, extract_only: bool = False, overrides: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    captions = pd.read_csv(source / "captions_and_labels.csv", low_memory=False)
    captions["article_id"] = captions["patient_id"].map(article_id)
    captions = captions[captions["article_id"] != ""]
    primaries = select_primaries(captions, collections, per_collection, images_per_patient, overrides)
    patients = sorted(primaries["patient_id"].unique())
    wanted_articles = {article_id(patient) for patient in patients}
    images_by_patient = {str(key): rows for key, rows in captions[captions["patient_id"].isin(set(patients))]
                         .groupby("patient_id", sort=False)}
    primaries_by_patient = {str(key): rows for key, rows in primaries.groupby("patient_id", sort=False)}
    meta_frame = pd.read_parquet(source / "metadata.parquet")
    metadata = {str(row["article_id"]): jsonable(row.to_dict()) for _, row in meta_frame.iterrows()
                if str(row["article_id"]) in wanted_articles}
    abstracts = {str(row["article_id"]): str(row.get("abstract") or "") for _, row in
                 pd.read_parquet(source / "abstracts.parquet").iterrows() if str(row["article_id"]) in wanted_articles}
    cases: dict[str, dict] = {}
    for _, row in pd.read_parquet(source / "cases.parquet").iterrows():
        if str(row["article_id"]) in wanted_articles:
            for case in as_list(row.get("cases")):
                if isinstance(case, dict) and case.get("case_id"):
                    cases[str(case["case_id"])] = case
    assets_dir, cache_dir = output / "assets", output / "enrichment-cache"
    assets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(primaries)} images from {len(patients)} patients across {collections}", flush=True)

    records: list[dict] = []
    rejected: list[dict] = []
    missing = unavailable = skipped = 0

    def job(patient: str) -> tuple[list[dict], int, str, list[dict]]:
        if extract_only:
            narrative = str(cases.get(patient, {}).get("case_text") or "")
            return [], 0, enrich_patient(patient, abstracts.get(article_id(patient), ""), narrative, cache_dir)[1], []
        return records_for_patient(patient, primaries_by_patient[patient], images_by_patient[patient], source,
                                   assets_dir, cache_dir, metadata, abstracts, cases)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="multicare") as executor:
        futures = {executor.submit(job, patient): patient for patient in patients}
        for position, future in enumerate(as_completed(futures), start=1):
            patient_records, patient_missing, status, patient_rejected = future.result()
            records.extend(patient_records)
            rejected.extend(patient_rejected)
            missing += patient_missing
            unavailable += status == "unavailable"
            skipped += status == "skipped-no-valid-image"
            if position % 10 == 0 or position == len(patients):
                print(f"Prepared {position}/{len(patients)} patients ({unavailable} extractions unavailable, "
                      f"{skipped} skipped: no valid image)", flush=True)
    # Completion order is intentionally irrelevant: the manifest must remain
    # byte-for-byte stable for a given source and enrichment cache.
    records.sort(key=lambda record: (record["collection"], record["article_id"], record["source_file_id"]))
    stats = {"record_count": len(records), "patients": len({record["patient_id"] for record in records}),
             "articles": len({record["article_id"] for record in records}),
             "by_collection": {name: sum(record["collection"] == name for record in records) for name in collections},
             "missing_source_images": missing, "unavailable_extractions": unavailable,
             "zero_shot_rejected": len(rejected), "patients_skipped_no_valid_image": skipped,
             "extracted_by": {model: sum(record.get("extracted_by") == model for record in records)
                              for model in sorted({str(record.get("extracted_by")) for record in records})},
             "zero_shot_rejected_by_collection": {name: sum(item["collection"] == name for item in rejected) for name in collections},
             "per_collection_articles": per_collection, "collection_articles": overrides or {},
             "images_per_patient": images_per_patient}
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--collections", default=",".join(COLLECTIONS),
                        help=f"Comma-separated subset of: {', '.join(COLLECTIONS)}")
    parser.add_argument("--per-collection", type=int, default=300, help="Articles sampled per collection")
    parser.add_argument("--collection-articles", default="",
                        help="Per-collection article counts overriding --per-collection, e.g. cxr=200,derm=150")
    parser.add_argument("--images-per-patient", type=int, default=2)
    parser.add_argument("--extract-only", action="store_true",
                        help="Only fill the Gemma 4 extraction cache (no images needed, no manifest written).")
    parser.add_argument("--extract-model", default=os.getenv("PREPARE_EXTRACT_MODEL"),
                        help="Model for new extractions (default: GEMMA_MODEL on the gateway).")
    parser.add_argument("--extract-base-url", default=os.getenv("PREPARE_EXTRACT_BASE_URL"),
                        help="Other OpenAI-compatible server for new extractions, e.g. http://<llm-host>:8888/v1")
    parser.add_argument("--workers", type=int, default=int(os.getenv("PREPARE_WORKERS", "2")),
                        help="Concurrent Gemma 4 requests (default: PREPARE_WORKERS or 2). Gemma 4 is a shared service: keep this low.")
    args = parser.parse_args()
    global EXTRACT_MODEL, EXTRACT_BASE_URL
    EXTRACT_MODEL, EXTRACT_BASE_URL = args.extract_model, args.extract_base_url
    collections = [name.strip() for name in args.collections.split(",") if name.strip()]
    overrides = {key.strip(): int(value) for key, value in
                 (item.split("=", 1) for item in args.collection_articles.split(",") if item.strip())}
    unknown = (set(collections) | set(overrides)) - set(COLLECTIONS)
    if unknown:
        parser.error(f"Unknown collections: {sorted(unknown)}")
    if args.workers < 1 or args.per_collection < 1 or args.images_per_patient < 1:
        parser.error("--workers, --per-collection and --images-per-patient must be at least 1")
    source, output = args.source.resolve(), args.output.resolve()
    required = [source / name for name in ("captions_and_labels.csv", "metadata.parquet", "abstracts.parquet", "cases.parquet")]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("MultiCaRe source tables are missing; source is never created or modified by this command")
    records, stats = build(source, output, collections, args.per_collection, args.images_per_patient, args.workers,
                           args.extract_only, overrides)
    if args.extract_only:
        print(json.dumps(stats))
        return
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "dataset_name": "MultiCaRe twin cases",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": {"root": str(source), "collections": {name: COLLECTIONS[name] for name in collections}},
                "stats": stats, "records": records}
    temporary = output / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "manifest.json")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
