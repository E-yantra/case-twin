"""Leave-one-out score calibration for hybrid twin search.

Each signal lives on its own scale: MedSigLIP image cosines sit around 0.3–0.9,
its text-to-image cosines around 0.0–0.15, Qwen3 text cosines around 0.2–0.8 and
reranker logits anywhere from -10 to +5. Adding them raw lets one signal swamp
the others. This script queries the library with its own cases (excluding the
case itself) and records the 5th and 95th percentile of each channel's top-10
scores. Search maps every raw score onto that range, so 0 = typical weak twin
and 1 = typical strong twin on every channel.

Writes dataset/score_calibration.json, which qdrant_service loads on start.
"""

from __future__ import annotations

import argparse
import json
import random

from qdrant_client import models

from local_ai import medsiglip_text_embedding, rerank
from manifest import DATASET_ROOT, case_document, normalize_profile
from qdrant_service import COLLECTION_NAME, _get_client

CALIBRATION_PATH = DATASET_ROOT / "score_calibration.json"


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    client = _get_client()
    points, offset = [], None
    while True:
        batch, offset = client.scroll(COLLECTION_NAME, limit=256, offset=offset, with_payload=True, with_vectors=True)
        points.extend(batch)
        if offset is None:
            break
    random.Random(7).shuffle(points)
    sample = points[:args.samples]
    channels: dict[str, list[float]] = {"image": [], "text": [], "crossmodal": [], "rerank": []}
    for point in sample:
        payload = point.payload or {}
        profile = normalize_profile(payload.get("profile"))
        patient = profile.get("case_id")
        same_collection = models.Filter(
            must=[models.FieldCondition(key="collection", match=models.MatchValue(value=payload.get("collection")))])
        for name, using, vector in (("image", "image", point.vector["image"]), ("text", "text", point.vector["text"])):
            hits = client.query_points(COLLECTION_NAME, query=vector, using=using, limit=args.top + 3, with_payload=True,
                                       query_filter=same_collection if name == "image" else None).points
            hits = [hit for hit in hits if (hit.payload or {}).get("profile", {}).get("case_id") != patient][:args.top]
            channels[name].extend(hit.score for hit in hits)
            if name == "text" and hits:
                documents = [case_document(normalize_profile(hit.payload.get("profile")), hit.payload.get("raw_narrative", ""))
                             for hit in hits]
                channels["rerank"].extend(rerank(case_document(profile), documents))
        findings = "; ".join(profile["findings"].get("imaging_findings") or [])
        if findings:
            text = f"{profile['study'].get('modality') or ''} {findings}"[:300]
            hits = client.query_points(COLLECTION_NAME, query=medsiglip_text_embedding(text), using="image",
                                       limit=args.top, with_payload=False).points
            channels["crossmodal"].extend(hit.score for hit in hits)
    calibration = {name: {"lo": round(percentile(values, 0.05), 4), "hi": round(percentile(values, 0.95), 4),
                          "n": len(values)} for name, values in channels.items() if values}
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps(calibration, indent=2))


if __name__ == "__main__":
    main()
