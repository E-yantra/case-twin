"""Revalidate all v2 cached claims against their retained source text."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from evidence import validate_updates
from manifest import empty_profile
from evidence import apply_updates

ARTICLE_FIELDS = {"patient.age_years", "patient.sex", "patient.comorbidities",
                  "presentation.chief_complaint", "presentation.symptom_duration", "presentation.pmh",
                  "assessment.diagnosis_primary", "outcome.detail"}
CAPTION_FIELDS = {"assessment.diagnosis_primary", "findings.lungs.consolidation_present",
                  "findings.lungs.opacity_present", "findings.lungs.atelectasis_present", "findings.lungs.edema_present",
                  "findings.pleura.effusion_present", "findings.pleura.pneumothorax_present",
                  "findings.cardiomediastinal.cardiomegaly", "findings.cardiomediastinal.mediastinal_gas_present"}


def audit(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "multicare-cxr/v2":
        raise ValueError("Expected a v2 evidence manifest")
    records = data.get("records")
    if not isinstance(records, list) or len(records) != 1038:
        raise ValueError("Expected all 1,038 retained records")
    article_ids = set()
    caption_statuses: dict[str, int] = {}
    article_failures = set()
    rejected_caption_claims = 0
    for record in records:
        article = record["article_id"]
        article_ids.add(article)
        article_context = record.get("article_context", {})
        article_texts = {"abstract": record.get("raw_abstract", "") or "",
                         "narrative": record.get("raw_narrative", "") or ""}
        article_updates = article_context.get("updates", []) if isinstance(article_context, dict) else []
        checked_article, article_warnings = validate_updates(json.dumps({"updates": article_updates}), article_texts, ARTICLE_FIELDS)
        article_status = article_context.get("status", "failed") if isinstance(article_context, dict) else "failed"
        prior_article_warnings = article_context.get("warnings", []) if isinstance(article_context, dict) else []
        if article_status == "failed" or article_warnings:
            article_failures.add(article)
        record["article_context"] = {**(article_context if isinstance(article_context, dict) else {}),
                                     "updates": checked_article,
                                     "warnings": [*prior_article_warnings, *article_warnings]}

        caption = record.get("primary_image", {}).get("caption", "") or ""
        caption_context = record.get("caption_context", {})
        caption_updates = caption_context.get("updates", []) if isinstance(caption_context, dict) else []
        checked_caption, warnings = validate_updates(json.dumps({"updates": caption_updates}), {"caption": caption}, CAPTION_FIELDS)
        rejected_caption_claims += len(caption_updates) - len(checked_caption)
        status = caption_context.get("status", "failed") if isinstance(caption_context, dict) else "failed"
        if status == "failed":
            rejected_caption_claims += len(checked_caption)
            checked_caption = []
        prior_caption_warnings = caption_context.get("warnings", []) if isinstance(caption_context, dict) else []
        if warnings:
            status = "failed"
        caption_statuses[status] = caption_statuses.get(status, 0) + 1
        record["caption_context"] = {**(caption_context if isinstance(caption_context, dict) else {}),
                                     "updates": checked_caption,
                                     "warnings": [*prior_caption_warnings, *warnings]}

        old_profile = record.get("profile", {})
        image = record["primary_image"]
        profile = empty_profile(article_id=article, image_id=image["asset_id"],
                                provenance=old_profile.get("provenance", {}))
        profile["study"].update({"caption": caption, "view_position": image.get("radiology_view"),
                                 "radiology_region": old_profile.get("study", {}).get("radiology_region"),
                                 "storage_path": image["asset_id"]})
        profile["tags"] = old_profile.get("tags", profile["tags"])
        profile = apply_updates(profile, checked_caption)
        profile["evidence"] = checked_caption
        profile["extraction_status"] = status
        profile["extraction_warnings"] = record["caption_context"]["warnings"]
        record["profile"] = profile
        record["enrichment_status"] = status

    if len(article_ids) != data.get("stats", {}).get("retained_articles"):
        raise ValueError("Unique article count does not match retained article statistics")
    data["stats"]["caption_extraction_statuses"] = caption_statuses
    data["stats"]["article_extraction_failures"] = len(article_failures)
    data["stats"]["rejected_caption_claims"] = rejected_caption_claims
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return {"records": len(records), "articles": len(article_ids), "caption_statuses": caption_statuses,
            "article_failures": len(article_failures), "rejected_caption_claims": rejected_caption_claims}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), indent=2))
