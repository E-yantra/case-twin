"""Rebuild CXR evidence from the retained v1 manifest when source tables are absent.

Writes a new manifest and versioned caches; never changes v1 or source assets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from evidence import EXTRACTION_VERSION, FIELD_TERMS, extract_claims, apply_updates  # noqa: E402
from manifest import empty_profile  # noqa: E402

ARTICLE_FIELDS = {"patient.age_years", "patient.sex", "patient.comorbidities",
                  "presentation.chief_complaint", "presentation.symptom_duration", "presentation.pmh",
                  "assessment.diagnosis_primary", "outcome.detail"}
CAPTION_FIELDS = {"assessment.diagnosis_primary", "findings.lungs.consolidation_present",
                  "findings.lungs.opacity_present", "findings.lungs.atelectasis_present", "findings.lungs.edema_present",
                  "findings.pleura.effusion_present", "findings.pleura.pneumothorax_present",
                  "findings.cardiomediastinal.cardiomegaly",
                  "findings.cardiomediastinal.mediastinal_gas_present"}
ARTICLE_EXTRACTION_VERSION = "source-evidence/v2"
CAPTION_EXTRACTION_VERSION = "caption-evidence/v3"


def _cache_file(cache_dir: Path, kind: str, key: str, sources: dict[str, str], allowed: set[str],
                version: str = EXTRACTION_VERSION) -> tuple[Path, str]:
    digest = hashlib.sha256(json.dumps({"version": version, "sources": sources,
                                       "allowed": sorted(allowed)}, sort_keys=True).encode()).hexdigest()
    return cache_dir / f"{kind}-{key}-{digest[:16]}.json", digest


def _write_cache(path: Path, digest: str, updates: list[dict], status: str, warnings: list[str],
                 version: str = EXTRACTION_VERSION) -> dict:
    result = {"source_hash": digest, "version": version,
              "updates": updates, "status": status, "warnings": warnings}
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return result


def _claim_batch(batch: list[tuple[str, dict[str, str], Path, str]], allowed: set[str],
                 version: str = EXTRACTION_VERSION) -> dict[str, dict]:
    sources: dict[str, str] = {}
    source_map: dict[str, tuple[str, str]] = {}
    for position, (key, item_sources, _, _) in enumerate(batch, start=1):
        for source, text in item_sources.items():
            if not text.strip():
                continue
            # Short, explicit source IDs survive model copying more reliably
            # than UUID-prefixed IDs. A single-record retry uses the natural
            # source name so quotes can be attributed without remapping.
            source_id = source if len(batch) == 1 else f"record_{position}_{source}"
            sources[source_id] = text
            source_map[source_id] = (key, source)
    updates, status, warnings = extract_claims(sources, allowed, max_tokens=5000)
    source_fault = any("source id" in warning.lower() or "quote does not occur" in warning.lower()
                       for warning in warnings)
    if (status == "failed" or source_fault) and len(batch) > 1:
        # Recover exact source attribution if the model did not cite the supplied
        # short ID or copied a quote from a neighboring caption.
        return {key: _claim_batch([(key, item_sources, path, digest)], allowed, version)[key]
                for key, item_sources, path, digest in batch}
    output = {}
    for key, item_sources, path, digest in batch:
        selected = [{**item, "source": source_map[item["source"]][1]} for item in updates
                    if item["source"] in source_map and source_map[item["source"]][0] == key]
        record_text = "\n".join(item_sources.values())
        record_warnings = []
        for warning in warnings:
            field_match = re.search(r"field='([^']+)'", warning)
            field = field_match.group(1) if field_match else None
            if field in FIELD_TERMS and re.search(FIELD_TERMS[field], record_text, re.I):
                record_warnings.append(warning)
            elif field == "assessment.diagnosis_primary" or field is None:
                record_warnings.append(warning)
        output[key] = _write_cache(path, digest, selected, status, record_warnings, version)
    return output


def _collect_claims(items: dict[str, dict[str, str]], cache_dir: Path, kind: str,
                    allowed: set[str], batch_size: int, workers: int,
                    retry_failed: bool = False, *, version: str = EXTRACTION_VERSION,
                    previous_versions: tuple[str, ...] = ()) -> dict[str, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    pending = []
    for key, sources in items.items():
        path, digest = _cache_file(cache_dir, kind, key, sources, allowed, version)
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            reusable_status = cached.get("status") == "complete" or (
                cached.get("status") == "failed" and not retry_failed
            )
            if reusable_status and cached.get("source_hash") == digest and not (cached.get("warnings") and retry_failed):
                output[key] = cached
                continue
        else:
            for previous_version in previous_versions:
                previous_path, previous_digest = _cache_file(cache_dir, kind, key, sources, allowed, previous_version)
                if not previous_path.is_file():
                    continue
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                if previous.get("source_hash") != previous_digest:
                    continue
                # Carry only clean, fully validated results across extractor
                # versions. Warning or failure entries are reprocessed below.
                if previous.get("status") == "complete" and not previous.get("warnings"):
                    output[key] = _write_cache(path, digest, previous.get("updates", []), "complete", [], version)
                    break
            if key in output:
                continue
        pending.append((key, sources, path, digest))
    batches = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_claim_batch, batch, allowed, version) for batch in batches]
        for position, future in enumerate(as_completed(futures), start=1):
            output.update(future.result())
            if position % 10 == 0 or position == len(batches):
                print(f"{kind}: {len(output)}/{len(items)} source caches ready", flush=True)
    return output


def rebuild(old_path: Path, new_path: Path, workers: int = 4, retry_failed: bool = False) -> dict:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    if old.get("schema_version") != "multicare-cxr/v1" or len(old.get("records", [])) != 1038:
        raise ValueError("Expected the retained 1,038-record v1 manifest")
    old_records = old["records"]
    cache = new_path.parent / "evidence-cache" / "v2"
    articles = {}
    for record in old_records:
        article = record["article_id"]
        sources = {key: record.get(value, "") or "" for key, value in
                   (("abstract", "raw_abstract"), ("narrative", "raw_narrative"))}
        if article in articles and articles[article] != sources:
            raise ValueError(f"Article text differs across image records: {article}")
        articles[article] = sources

    article_claims = _collect_claims(articles, cache / "article", "article", ARTICLE_FIELDS,
                                     1, workers, retry_failed, version=ARTICLE_EXTRACTION_VERSION)
    captions = {record["point_id"]: {"caption": record["primary_image"].get("caption", "") or ""}
                for record in old_records}
    caption_claims = _collect_claims(captions, cache / "caption", "caption", CAPTION_FIELDS,
                                     5, workers, retry_failed, version=CAPTION_EXTRACTION_VERSION,
                                     previous_versions=(ARTICLE_EXTRACTION_VERSION,))

    def rebuild_record(record: dict) -> dict:
        item = dict(record)
        image = record["primary_image"]
        caption = image.get("caption", "") or ""
        claims = caption_claims[record["point_id"]]
        old_profile = record["profile"]
        profile = empty_profile(article_id=record["article_id"], image_id=image["asset_id"],
                                provenance=old_profile.get("provenance", {}))
        profile["study"].update({"caption": caption, "view_position": image.get("radiology_view"),
                                 "radiology_region": old_profile.get("study", {}).get("radiology_region"),
                                 "storage_path": image["asset_id"]})
        profile["provenance"] = old_profile.get("provenance", profile["provenance"])
        profile["tags"] = old_profile.get("tags", profile["tags"])
        profile = apply_updates(profile, claims["updates"])
        profile["evidence"] = claims["updates"]
        profile["extraction_status"] = claims["status"]
        profile["extraction_warnings"] = claims["warnings"]
        item["profile"] = profile
        item["article_context"] = article_claims[record["article_id"]]
        item["caption_context"] = claims
        item["enrichment_status"] = claims["status"]
        return item

    rebuilt = [rebuild_record(record) for record in old_records]
    old_ids = {record["point_id"] for record in old_records}
    if len(rebuilt) != len(old_records) or {record["point_id"] for record in rebuilt} != old_ids:
        raise ValueError("Not every retained record was accounted for")
    statuses = {status: sum(r["caption_context"]["status"] == status for r in rebuilt)
                for status in {r["caption_context"]["status"] for r in rebuilt}}
    output = {key: value for key, value in old.items() if key != "records"}
    output["schema_version"] = "multicare-cxr/v2"
    output["source_manifest"] = str(old_path)
    output["extraction_version"] = CAPTION_EXTRACTION_VERSION
    output["article_extraction_version"] = ARTICLE_EXTRACTION_VERSION
    output["stats"] = {**old.get("stats", {}), "caption_extraction_statuses": statuses,
                       "article_extraction_failures": sum(c["status"] == "failed" for c in article_claims.values())}
    output["records"] = rebuilt
    new_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = new_path.with_suffix(".part")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(new_path)
    return {"records": len(rebuilt), "articles": len(articles), "caption_statuses": statuses,
            "article_failures": output["stats"]["article_extraction_failures"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "dataset/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/manifest.v2.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry cache entries with failed model extraction status")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(rebuild(args.source_manifest, args.output, args.workers, args.retry_failed), indent=2))
