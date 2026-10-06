"""Hybrid twin-case retrieval over the canonical MultiCaRe manifest payload.

Three signals are combined, and each match reports all of them:
  image  — MedSigLIP image-to-image cosine similarity (what the picture looks like)
  text   — Qwen3 text cosine similarity between structured case reports (what the case says)
  rerank — bge-reranker cross-encoder relevance of the twin report to the query report
When no image is uploaded, MedSigLIP's text tower embeds the query findings into
the image space instead ("cross-modal" search).
"""

from __future__ import annotations

import json
import math
import os
import time

from qdrant_client import QdrantClient, models

from manifest import DATASET_ROOT, case_document, normalize_profile

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multicare_cases")
CANDIDATES_PER_CHANNEL = 40
RERANK_TOP = 20
_client: QdrantClient | None = None
_calibration: dict | None = None


def calibration() -> dict:
    """Per-channel score ranges from calibrate_scores.py (empty -> raw scores are used)."""
    global _calibration
    if _calibration is None:
        path = DATASET_ROOT / "score_calibration.json"
        _calibration = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return _calibration


def calibrated(channel: str, raw: float) -> float:
    """Map a raw score onto 0..1, where 0/1 are the library's typical weak/strong twin."""
    spec = calibration().get(channel)
    if channel == "rerank" and not spec:
        return _sigmoid(raw)
    if not spec or spec["hi"] <= spec["lo"]:
        return raw
    return min(1.0, max(0.0, (raw - spec["lo"]) / (spec["hi"] - spec["lo"])))


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def weights_for(has_image: bool, has_text: bool) -> dict[str, float]:
    """Fusion weights for the channels that are available on this query."""
    if has_image and has_text:
        return {"image": 0.5, "text": 0.3, "rerank": 0.2}
    if has_image:
        return {"image": 1.0}
    return {"crossmodal": 0.15, "text": 0.55, "rerank": 0.3}


def _dot(left: list[float] | None, right: list[float] | None) -> float | None:
    if not left or not right:
        return None
    return sum(a * b for a, b in zip(left, right))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def search_similar(*, image_vector: list[float] | None = None, crossmodal_vector: list[float] | None = None,
                   text_vector: list[float] | None = None, query_document: str = "", collection: str | None = None,
                   limit: int = 10, reranker=None) -> tuple[list[dict], list[dict]]:
    """Return (matches, trace steps). ``reranker(query, docs) -> scores`` is optional."""
    if not (image_vector or crossmodal_vector or text_vector):
        raise ValueError("Search needs an image or a case description")
    client = _get_client()
    query_filter = (models.Filter(must=[models.FieldCondition(key="collection", match=models.MatchValue(value=collection))])
                    if collection else None)
    channels = {"image": image_vector, "crossmodal": crossmodal_vector, "text": text_vector}
    started = time.perf_counter()
    candidate_ids: list = []
    for name, vector in channels.items():
        if not vector:
            continue
        using = "text" if name == "text" else "image"
        for point in client.query_points(collection_name=COLLECTION_NAME, query=vector, using=using,
                                         query_filter=query_filter, limit=CANDIDATES_PER_CHANNEL,
                                         with_payload=False).points:
            if point.id not in candidate_ids:
                candidate_ids.append(point.id)
    if not candidate_ids:
        return [], []
    # Score every candidate on every available channel, not just the one that found it.
    points = client.retrieve(COLLECTION_NAME, ids=candidate_ids, with_payload=True, with_vectors=True)
    weights = weights_for(bool(image_vector), bool(text_vector))
    scored = []
    for point in points:
        vectors = point.vector if isinstance(point.vector, dict) else {}
        scores = {
            "image": _dot(image_vector, vectors.get("image")),
            "crossmodal": _dot(crossmodal_vector, vectors.get("image")),
            "text": _dot(text_vector, vectors.get("text")),
        }
        raw = {key: value for key, value in scores.items() if value is not None}
        scored.append({"point": point, "raw": raw, "scores": {key: calibrated(key, value) for key, value in raw.items()}})
    for item in scored:
        item["fused"] = sum(weights.get(key, 0) * value for key, value in item["scores"].items())
    scored.sort(key=lambda item: item["fused"], reverse=True)
    trace = [{"model": "Qdrant", "task": f"Vector search over {', '.join(k for k, v in channels.items() if v)} "
              f"({len(candidate_ids)} candidates)", "ms": round((time.perf_counter() - started) * 1000)}]

    if reranker and query_document and "rerank" in weights:
        top = scored[:RERANK_TOP]
        started = time.perf_counter()
        try:
            documents = [case_document(normalize_profile((item["point"].payload or {}).get("profile")),
                                       (item["point"].payload or {}).get("raw_narrative", "")) for item in top]
            for item, score in zip(top, reranker(query_document, documents)):
                item["raw"]["rerank"] = score
                item["scores"]["rerank"] = calibrated("rerank", score)
            trace.append({"model": "bge-reranker-v2-m3", "task": f"Rerank top {len(top)} case reports",
                          "ms": round((time.perf_counter() - started) * 1000)})
        except Exception as exc:  # noqa: BLE001 - rerank is an optional refinement
            trace.append({"model": "bge-reranker-v2-m3", "task": f"Rerank skipped: {exc}", "ms": 0})
        for item in scored:
            item["fused"] = sum(weights.get(key, 0) * value for key, value in item["scores"].items())
        # Candidates outside the reranked window stay ranked below it.
        scored.sort(key=lambda item: ("rerank" in item["scores"], item["fused"]), reverse=True)
    # One twin per patient: several images of the same case should not crowd the list.
    unique, seen = [], set()
    for item in scored:
        payload = item["point"].payload or {}
        patient = (payload.get("profile") or {}).get("case_id") or item["point"].id
        if patient not in seen:
            seen.add(patient)
            unique.append(item)
    return [_match(item, weights) for item in unique[:limit]], trace


def _match(item: dict, weights: dict[str, float]) -> dict:
    point, payload = item["point"], item["point"].payload or {}
    profile = normalize_profile(payload.get("profile"))
    assessment, summary, provenance, study = (profile.get("assessment", {}), profile.get("summary", {}),
                                              profile.get("provenance", {}), profile.get("study", {}))
    outcome, management = profile.get("outcome", {}), profile.get("management", {})
    final = max(0.0, item["fused"])
    primary = assessment.get("diagnosis_primary") or summary.get("one_liner") or study.get("caption") or "Historical case"
    return {
        "id": str(point.id), "score": round(final * 100), "diagnosis": str(primary)[:160],
        "scores": {key: round(value, 4) for key, value in item["scores"].items()},
        "raw_scores": {key: round(value, 4) for key, value in item["raw"].items()}, "weights": weights,
        "collection": payload.get("collection"), "modality": study.get("modality"),
        "summary": summary.get("one_liner") or study.get("caption") or "No structured summary available.",
        "facility": "MultiCaRe", "outcome": outcome.get("detail") or "Outcome not reported",
        "outcomeVariant": {"yes": "success", "no": "warning"}.get(str(outcome.get("success")).lower(), "neutral"),
        "conclusion": summary.get("conclusion"), "treatments": management.get("treatments", []),
        "imaging_findings": profile.get("findings", {}).get("imaging_findings", []),
        "asset_id": payload.get("primary_asset_id"), "related_asset_ids": payload.get("related_asset_ids", []),
        "age": profile.get("patient", {}).get("age_years"), "gender": profile.get("patient", {}).get("sex"),
        "pmc_id": provenance.get("pmc_id"), "article_title": provenance.get("article_title"),
        "journal": provenance.get("journal"), "year": provenance.get("year"), "license": provenance.get("license"),
        "source_url": provenance.get("source_url"), "radiology_view": study.get("view_position"),
        "case_text": profile.get("presentation", {}).get("hpi") or payload.get("raw_narrative", ""),
        "raw_payload": {**profile, "primary_asset_id": payload.get("primary_asset_id"),
                        "related_images": payload.get("related_images", [])
                        if isinstance(payload.get("related_images"), list) else []},
    }
