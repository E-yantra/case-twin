"""
CaseTwin FastAPI backend.
POST /extract            — Gemma 4 + MedGemma: unstructured notes and images -> structured case report.
POST /search             — MedSigLIP + Qwen3 + bge-reranker hybrid twin-case retrieval from Qdrant.
POST /compare_insights   — MedGemma 1.5 side-by-side image comparison (+ CXR finding boxes).
POST /chat_twin          — MedGemma answers questions grounded in the twin and current case.
POST /explain_selection  — MedGemma explains a highlighted medical term.
POST /enhance_profile    — MedGemma clinical synthesis of the current case.
GET  /health, /ai_status — health and model availability.

Every AI endpoint returns a ``trace``: which model did which step and how long it took,
so the UI can show the pipeline.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import io
import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import asyncio
import time

from embedding_service import (
    generate_embedding, query_medgemma,
    query_medgemma_localization, query_medgemma_read,
)
from local_ai import (
    GEMMA_MODEL, MEDGEMMA_COMPARISON_MODEL, MEDGEMMA_LOCALIZATION_MODEL, MEDGEMMA_MODEL, medsiglip_classify,
    medsiglip_text_embedding, query_text, rerank, text_embeddings,
)
from collections_config import COLLECTIONS, quality_labels
from extraction import extract_profile, intake_prompt
from qdrant_service import COLLECTION_NAME, search_similar
from manifest import asset_path, case_document, empty_profile, merge_profile, normalize_profile
from document_text import DocumentExtractionError, extract_document_text

app = FastAPI(title="CaseTwin API", version="1.0.0")
logger = logging.getLogger(__name__)

# Allow the Vite dev server (and any localhost port) to call the API
_configured_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080", *_configured_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


MODEL_ROLES = [
    {"id": GEMMA_MODEL, "role": "Turns unstructured notes into the structured case report (JSON schema)"},
    {"id": MEDGEMMA_MODEL, "role": "Reads and compares images, explains terms, synthesises and answers questions about cases"},
    {"id": MEDGEMMA_LOCALIZATION_MODEL, "role": "Localises chest X-ray findings (bounding boxes)"},
    {"id": "MedSigLIP", "role": "Image embeddings for visual search; zero-shot image-type routing; text-to-image search"},
    {"id": "qwen3-embedding-8b", "role": "Text embeddings of case reports for semantic search"},
    {"id": "bge-reranker-v2-m3", "role": "Cross-encoder reranking of the top candidate twins"},
]


@app.get("/ai_status")
async def ai_status():
    """Which local models and stores are reachable right now."""
    import httpx
    from local_ai import GATEWAY_BASE_URL, MEDSIGLIP_BASE_URL, _gateway_headers
    available: set[str] = set()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{GATEWAY_BASE_URL}/models", headers=_gateway_headers())
            available = {model["id"] for model in response.json().get("data", [])}
    except Exception:
        logger.warning("Gateway model list unavailable")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            if (await client.get(f"{MEDSIGLIP_BASE_URL}/health")).status_code == 200:
                available.add("MedSigLIP")
    except Exception:
        logger.warning("MedSigLIP health check failed")
    try:
        from qdrant_service import _get_client
        info = await asyncio.to_thread(_get_client().get_collection, COLLECTION_NAME)
        library = {"collection": COLLECTION_NAME, "points": info.points_count}
    except Exception:
        library = {"collection": COLLECTION_NAME, "points": 0}
    return {"models": [{**model, "available": model["id"] in available} for model in MODEL_ROLES],
            "library": library}


@app.get("/dataset-images/{asset_id}")
def dataset_image(asset_id: str):
    """Serve a derived asset by safe identifier; source paths never leave the backend."""
    try:
        path = asset_path(asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Dataset image not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found")
    return FileResponse(path)


def _dataset_url(request: Request, asset_id: str | None) -> str | None:
    return str(request.url_for("dataset_image", asset_id=asset_id)) if asset_id else None


def _step(model: str, task: str, started: float, **extra) -> dict:
    return {"model": model, "task": task, "ms": round((time.perf_counter() - started) * 1000), **extra}


ROUTING_MARGIN = 1.5


def _route_collection(image: Image.Image) -> tuple[str | None, list[dict]]:
    """MedSigLIP zero-shot: which twin collection does this image belong to?

    Distractor labels (charts, endoscopy, ECG…) let the model say "none of these";
    then the search runs across every collection instead of forcing a wrong one.
    """
    labels = {spec["zero_shot"]: name for name, spec in COLLECTIONS.items()}
    ranked = medsiglip_classify(image, quality_labels())
    scores = [{"collection": labels.get(item["label"]),
               "label": COLLECTIONS[labels[item["label"]]]["label"] if item["label"] in labels else item["label"],
               "score": round(float(item["score"]), 8)} for item in ranked[:4]]
    best_collection = next((item for item in ranked if item["label"] in labels), None)
    # A distractor must clearly win (> ROUTING_MARGIN x) to override a library type;
    # near-ties (a real CXR scoring like "fluoroscopy") still route to the collection.
    if best_collection and ranked[0]["score"] <= ROUTING_MARGIN * best_collection["score"]:
        chosen = labels[best_collection["label"]]
        scores.sort(key=lambda item: item["collection"] != chosen)
        return chosen, scores
    return None, scores


def _crossmodal_text(profile: dict) -> str:
    """Short image-like description for MedSigLIP's text tower (it is trained on captions; max 64 tokens)."""
    study, findings = profile.get("study", {}), profile.get("findings", {})
    parts = [study.get("modality"), study.get("body_region"), "; ".join(findings.get("imaging_findings") or []),
             profile.get("assessment", {}).get("diagnosis_primary")]
    return " ".join(str(part) for part in parts if part)[:400]


@app.get("/collections")
def list_collections():
    return {"collections": [{"id": name, "label": spec["label"], "modality": spec["modality"]}
                            for name, spec in COLLECTIONS.items()]}


@app.post("/search")
async def search(
    request: Request,
    file: Optional[UploadFile] = File(None),
    profile: Optional[str] = Form(None),
    collection: Optional[str] = Form("auto"),
    limit: int = 10
):
    """
    Hybrid twin search. The image (MedSigLIP) and the structured case report
    (Qwen3 embeddings + bge reranker) are both used when available; with only a
    report, MedSigLIP's text tower searches the image space cross-modally.
    """
    trace: list[dict] = []
    image = None
    if file is not None and file.filename:
        if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
            raise HTTPException(status_code=400, detail="Only image files are accepted (jpg, png, webp).")
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    parsed_profile = None
    if profile:
        try:
            parsed_profile = normalize_profile(json.loads(profile))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Profile is not valid JSON: {e}")
    query_document = case_document(parsed_profile) if parsed_profile else ""
    if image is None and len(query_document) < 20:
        raise HTTPException(status_code=400, detail="Upload an image or build a case profile before searching.")

    routing = None
    chosen = collection if collection in COLLECTIONS else None
    image_vector = crossmodal_vector = text_vector = None
    try:
        if image is not None:
            started = time.perf_counter()
            image_vector = await asyncio.to_thread(generate_embedding, image)
            trace.append(_step("MedSigLIP", "Embed uploaded image (1152-d)", started))
            if collection in (None, "", "auto"):
                started = time.perf_counter()
                chosen, scores = await asyncio.to_thread(_route_collection, image)
                routing = {"collection": chosen, "scores": scores}
                trace.append(_step("MedSigLIP", f"Zero-shot image type: {scores[0]['label']}"
                                   + ("" if chosen else " (not a library type; searching all collections)"), started))
        if query_document:
            started = time.perf_counter()
            text_vector = (await asyncio.to_thread(text_embeddings, [query_document], query=True))[0]
            trace.append(_step("qwen3-embedding-8b", "Embed structured case report (4096-d)", started))
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=503, detail=f"A local embedding model is unavailable: {e}")
    if image is None and parsed_profile and _crossmodal_text(parsed_profile):
        # Cross-modal is a minor extra signal: if it fails, search on the report text alone.
        started = time.perf_counter()
        try:
            crossmodal_vector = await asyncio.to_thread(medsiglip_text_embedding, _crossmodal_text(parsed_profile))
            trace.append(_step("MedSigLIP", "Embed findings text into the image space (cross-modal)", started))
        except Exception as e:
            logger.warning("Cross-modal embedding skipped: %s", e)
            trace.append(_step("MedSigLIP", "Cross-modal search skipped (text encoder unavailable)", started))
    if not (image_vector or text_vector or crossmodal_vector):
        raise HTTPException(status_code=503, detail="No embedding could be computed for this case.")

    try:
        matches, search_trace = await asyncio.to_thread(
            search_similar, image_vector=image_vector, crossmodal_vector=crossmodal_vector,
            text_vector=text_vector, query_document=query_document, collection=chosen,
            limit=limit, reranker=rerank,
        )
    except Exception as e:
        logger.exception("Qdrant search failed")
        raise HTTPException(status_code=500, detail=f"Twin search failed: {e}")
    trace.extend(search_trace)

    for match in matches:
        match["image_url"] = _dataset_url(request, match.get("asset_id"))
        match["related_image_urls"] = [_dataset_url(request, asset_id) for asset_id in match.get("related_asset_ids", [])]
        payload = match.get("raw_payload")
        if isinstance(payload, dict):
            # The modal reads the canonical profile, so expose local URLs there too.
            payload.get("study", {})["image_url"] = match["image_url"]
            for related, url in zip(payload.get("related_images", []), match["related_image_urls"]):
                if isinstance(related, dict):
                    related["image_url"] = url
    return {"matches": matches, "count": len(matches), "collection": chosen, "routing": routing, "trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# /compare_insights  – Visual comparison of two independent cases
# ──────────────────────────────────────────────────────────────────────────────
def _model_json(reply: str):
    """Read only the final JSON answer; localization can include reasoning first."""
    if not isinstance(reply, str):
        return None
    final = reply.rsplit("Final Answer:", 1)[-1].strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", final, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        final = fenced.group(1)
    try:
        return json.loads(final)
    except (TypeError, ValueError):
        return None


def _localization_box(reply: str, finding: str) -> list[int] | None:
    result = _model_json(reply)
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        return None
    label = result[0].get("label")
    finding_terms = set(re.findall(r"[a-z]{5,}", finding.lower())) - {
        "large", "small", "sided", "lower", "upper", "right", "left",
        "acute", "chronic", "pleural", "pulmonary", "mediastinal", "lung",
    }
    label_terms = set(re.findall(r"[a-z]{5,}", label.lower())) if isinstance(label, str) else set()
    if not finding_terms or not finding_terms.intersection(label_terms):
        return None
    box = result[0].get("box_2d")
    if not isinstance(box, list) or len(box) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in box):
        return None
    y0, x0, y1, x1 = box
    if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        return None
    if (y1 - y0) < 25 or (x1 - x0) < 25:
        return None
    return [round(value) for value in box]


def _finding_name(value) -> str | None:
    if not isinstance(value, str):
        return None
    finding = " ".join(value.split())
    if not (1 <= len(finding) <= 100) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.'()/-]*", finding):
        return None
    if finding.lower() in {"normal", "none", "no finding", "no abnormality"}:
        return None
    return finding


def _historical_line(text: str) -> str:
    match = re.search(r"Historical:\s*(.*?)(?=Visual comparison:|\Z)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _reported_finding(caption: str) -> tuple[str, str] | None:
    """Recognize a few explicit figure-caption findings, not the case outcome."""
    if not caption:
        return None
    for label, pattern in (
        ("mediastinal gas", r"mediastinal gas|pneumomediastinum|continuous diaphragm sign|Naclerio.s V sign"),
        ("pneumothorax", r"pneumothorax"),
        ("pleural effusion", r"pleural effusion"),
    ):
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if match:
            prefix = caption[max(0, match.start() - 50):match.start()]
            if re.search(r"\b(?:no|without|absence of|resolved|resolution of)\s+(?:\w+\s+){0,2}$", prefix, re.IGNORECASE):
                continue
            return label, pattern
    return None


def _positive_mention(line: str, pattern: str) -> bool:
    """Require a visible, affirmative mention before trusting any model box."""
    if not line or re.search(r"\b(?:caption|report(?:ed)?)\b", line, re.IGNORECASE):
        return False
    for match in re.finditer(pattern, line, flags=re.IGNORECASE):
        before = line[max(0, match.start() - 70):match.start()]
        before = re.split(r"[.;:]", before)[-1]
        after = line[match.end():match.end() + 55]
        if re.search(r"\b(?:no|neither|without|absent|negative for|free of|not)\b(?:\W+\w+){0,6}\W*$", before, re.IGNORECASE):
            continue
        if re.search(r"^\W*(?:is|are)?\s*not\s+(?:visually\s+)?(?:seen|visible|supported|present|identified)", after, re.IGNORECASE):
            continue
        return True
    return False


def _finding_supported(text: str, field: str, finding: str) -> bool:
    label = "Current" if field == "original_box" else "Historical"
    match = re.search(rf"{label}:\s*(.*?)(?=(?:Current|Historical|Visual comparison):|\Z)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return False
    terms = re.findall(r"[a-z]{5,}", finding.lower())
    terms = [term for term in terms if term not in {
        "large", "small", "sided", "lower", "upper", "right", "left",
        "acute", "chronic", "pleural", "pulmonary", "mediastinal", "lung",
    }]
    if not terms:
        return False
    return _positive_mention(match.group(1), r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b")


def _localization_prompt(finding: str) -> str:
    return (
        "Instructions:\nThe following user query will require outputting bounding "
        "boxes. The format of bounding box coordinates is [y0, x0, y1, x1] "
        "where (y0, x0) is the top-left corner and (y1, x1) is the bottom-right "
        "corner. This implies x0 < x1 and y0 < y1. Normalize x and y to [0, "
        "1000], so 15% of the image width has x=150.\n"
        "You MUST output one parseable JSON list of objects enclosed in "
        '```json...``` brackets, for example '
        '```json[{"box_2d":[140,110,340,470],"label":"right clavicle"}]``` '
        "is valid.\nRemember left refers to the patient's left side, near the "
        "heart and on the right side of a frontal radiograph.\n"
        f"Query:\nWhere is the {finding}? Don't give a final answer without "
        'reasoning. Output the final answer in the format "Final Answer: X" '
        'where X is a JSON list of objects with "box_2d" and "label" keys. Answer:'
    )


def _read_prompt(kind: str) -> str:
    """Dictation-style read (finds abnormalities more reliably than a prose findings section).

    MedGemma confuses the patient's left and right on published figures (it tends to
    report the viewer's side), so reads describe position without sides; the CXR
    finding boxes, which MedGemma 1.5 localises correctly, show where a finding is.
    """
    if kind == "chest radiograph":
        return (
            "Describe the key findings visible in this chest radiograph as a short list, the way a radiologist "
            "would dictate them. Do not say left or right; describe position only as upper, middle or lower "
            "zone, and central or peripheral. Mention notable normal findings only if relevant. If a feature is "
            "unclear, say so. Do not give a final diagnosis. Do not mention captions, arrows or marks, chronology, "
            "treatment, prognosis or outcome. Maximum 60 words."
        )
    return (
        f"Describe the key findings visible in this {kind} as a short list, the way a specialist would dictate "
        "them. If a feature is unclear, say so. Do not give a final diagnosis. Do not mention captions, arrows "
        "or marks, chronology, treatment, prognosis or outcome. Maximum 60 words."
    )


def _strip_preamble(text: str) -> str:
    """Drop an opening line such as "Here's a description of the findings:"."""
    lines = [line for line in text.strip().split("\n") if line.strip()]
    while lines and (lines[0].rstrip().endswith(":") or re.match(r"^(?:here(?:'s| is| are)|sure|okay)\b", lines[0], re.I)):
        lines.pop(0)
    return " ".join(" ".join(lines).split())


async def _independent_comparison(current: Image.Image, historical: Image.Image, kind: str) -> tuple[str, str, str]:
    prompt = _read_prompt(kind)
    reads = await asyncio.gather(
        asyncio.to_thread(query_medgemma_read, current, prompt),
        asyncio.to_thread(query_medgemma_read, historical, prompt),
    )
    current_read, historical_read = (_strip_preamble(_clean_reply(read[0].get("generated_text", ""))) for read in reads)
    if not (current_read and historical_read):
        return current_read, historical_read, ""
    comparison = await asyncio.to_thread(
        query_text,
        "Below are two independent reads of images from two different patients. In one or two short "
        "sentences, compare only their visible similarities and differences. Do not mention left or right "
        "sides. Do not suggest that one patient developed the other's findings, and do not mention "
        "treatment, prognosis or outcome.\n\n"
        f"Current: {current_read}\nHistorical: {historical_read}",
        model=MEDGEMMA_COMPARISON_MODEL, max_tokens=160, temperature=0,
    )
    return current_read, historical_read, _strip_preamble(_clean_reply(comparison))


_FINDINGS_SCHEMA = {"type": "object", "properties": {
    "current_finding": {"type": ["string", "null"]}, "historical_finding": {"type": ["string", "null"]}},
    "required": ["current_finding", "historical_finding"]}


def _finding_names_from_reads(current_read: str, historical_read: str) -> dict | None:
    reply = query_text(
        "For each chest X-ray read below, name the single most important abnormal lung or pleural "
        "finding that the read states as present (not negated or uncertain), in 1-4 words, or null if "
        "the read describes no clear abnormality. Return JSON with current_finding and historical_finding."
        f"\n\nCurrent read: {current_read}\nHistorical read: {historical_read}",
        model=MEDGEMMA_COMPARISON_MODEL, max_tokens=80, temperature=0, json_schema=_FINDINGS_SCHEMA,
    )
    return _model_json(reply)


@app.post("/compare_insights")
async def compare_insights(
    original_image: UploadFile = File(...),
    match_diagnosis: str = Form(...),
    match_asset_id: str = Form(None),
    match_payload: str = Form(None),
    match_caption: str = Form(None),
    match_collection: str = Form(None),
):
    """Compare two images without using the historical record as model context.

    MedGemma reads each image in its own call (in parallel), so the second read
    can never copy the first; a text-only call then compares the two reads.
    Finding boxes are produced only for chest X-rays, by MedGemma 1.5, whose
    localization was trained on them.
    """
    # These legacy form fields remain accepted for clients using the existing API.
    _ = match_payload
    is_cxr = match_collection in (None, "", "cxr")
    try:
        contents = await original_image.read()
        current_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read current image") from exc

    if not match_asset_id:
        raise HTTPException(status_code=400, detail="A historical match image is required for visual comparison")
    try:
        local_asset = asset_path(match_asset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Historical match image was not found") from exc
    if not local_asset.is_file():
        raise HTTPException(status_code=404, detail="Historical match image was not found")
    try:
        historical_image = Image.open(local_asset).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Historical match image could not be read") from exc

    kind = "chest radiograph" if is_cxr else COLLECTIONS.get(match_collection, {}).get("label", "medical") + " image"
    started = time.perf_counter()
    try:
        current_read, historical_read, comparison = await _independent_comparison(current_image, historical_image, kind)
    except Exception as exc:
        logger.exception("MedGemma visual comparison failed")
        raise HTTPException(
            status_code=502,
            detail="The local model could not complete the visual comparison. Please retry.",
        ) from exc
    if not (current_read and historical_read and comparison):
        raise HTTPException(
            status_code=502,
            detail="The local model returned no visual comparison. Please retry.",
        )
    text = f"Current: {current_read}\nHistorical: {historical_read}\nVisual comparison: {comparison}"
    if re.search(
        r"\b(?:resolution|resolved|resolving|progression|progressed|follow[- ]up|"
        r"prognos\w*|recover\w*|same patient|before[- ]and[- ]after|outcome)\b",
        text, flags=re.IGNORECASE,
    ):
        raise HTTPException(
            status_code=502,
            detail="The local model returned a comparison that needs review. Please retry.",
        )
    trace = [_step(MEDGEMMA_COMPARISON_MODEL, "Read each image independently (2 parallel calls), then compare the reads",
                   started)]
    if not is_cxr:
        return {"insights_text": text, "original_box": None, "match_box": None, "trace": trace}
    reported = _reported_finding(match_caption or "")
    if reported and not _positive_mention(_historical_line(text), reported[1]):
        return {
            "insights_text": (
                f"AI visual comparison inconclusive: the model did not confirm the historical "
                f"case caption's reported {reported[0]}. Review the source caption and X-ray. "
                "No reliable comparison or suggested boxes are available for this pair."
            ),
            "original_box": None,
            "match_box": None,
            "trace": trace,
        }
    # Finding names come from the two independent reads (text-only), so the
    # localizer is only asked about findings a read actually states as present.
    boxes = {"original_box": None, "match_box": None}
    try:
        findings = await asyncio.to_thread(_finding_names_from_reads, current_read, historical_read)
    except Exception:
        logger.exception("MedGemma finding check was unavailable")
        findings = None
    if isinstance(findings, dict):
        for field, image, finding_key in (
            ("original_box", current_image, "current_finding"),
            ("match_box", historical_image, "historical_finding"),
        ):
            finding = _finding_name(findings.get(finding_key))
            if finding and _finding_supported(text, field, finding):
                try:
                    localized = await asyncio.to_thread(
                        query_medgemma_localization, image,
                        prompt=_localization_prompt(finding),
                    )
                    boxes[field] = _localization_box(localized[0]["generated_text"], finding)
                except Exception:
                    logger.exception("MedGemma localization was unavailable")

    if any(boxes.values()):
        trace.append(_step(MEDGEMMA_LOCALIZATION_MODEL, "Localize visible findings (bounding boxes)", started))
    return {"insights_text": text, **boxes, "trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# /search_hospitals  – You.com RAG integration for dynamic facility routing
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/search_hospitals")
async def search_hospitals(
    diagnosis: str = Form(...),
    location: Optional[str] = Form(None),
    equipment: Optional[str] = Form(None),
    maxTravelTime: Optional[str] = Form(None),
    maxDistance: Optional[str] = Form(None)
):
    """
    Query the You.com RAG API to find relevant top-tier hospitals for the given diagnosis.
    Returns structured data detailing facility names, capabilities, reason for match,
    and approximate locations/coordinates for mapping.
    """
    import os
    import httpx
    import json
    
    ydc_api_key = os.getenv("YDC_API_KEY")
    if not ydc_api_key:
        raise HTTPException(status_code=500, detail="YDC_API_KEY environment variable is missing.")

    loc_context = f" near {location}" if location else " in the United States"
    eq_context = f" YOU MUST ONLY INCLUDE HOSPITALS THAT EXPLICITLY HAVE THE FOLLOWING EQUIPMENT/CAPABILITIES: {equipment}." if equipment else ""
    travel_context = f" The hospital MUST be reachable within a {maxTravelTime} hour travel time from the location." if maxTravelTime else ""
    
    # Use Geopy for real coordinates and OSRM for real routing
    from geopy.geocoders import Nominatim
    import asyncio
    
    # Using a custom user agent as required by Nominatim's Terms of Service
    geolocator = Nominatim(user_agent="casetwin_medical_routing_bot")
    
    user_lat, user_lng = 39.8283, -98.5795 # Default US Center
    search_location_str = location or 'United States'
    
    if location:
        # Check if location is coordinates (e.g., "28.5383, -81.3792")
        is_coords = False
        if ',' in location:
            parts = location.split(',')
            try:
                user_lat = float(parts[0].strip())
                user_lng = float(parts[1].strip())
                is_coords = True
            except ValueError:
                pass
                
        if is_coords:
            def reverse_loc(lat, lng):
                return geolocator.reverse(f"{lat}, {lng}", timeout=5)
            try:
                rev_data = await asyncio.to_thread(reverse_loc, user_lat, user_lng)
                if rev_data:
                    address = rev_data.raw.get('address', {})
                    city = address.get('city') or address.get('town') or address.get('county') or address.get('state')
                    if city:
                        search_location_str = f"{city}, {address.get('state', '')}"
                        print(f"Reverse geocode success: {location} -> {search_location_str}", flush=True)
            except Exception as e:
                print(f"Reverse geocode failed: {e}", flush=True)
        else:
            def geocode_loc(loc_str):
                return geolocator.geocode(loc_str, timeout=5)
            try:
                user_loc_data = await asyncio.to_thread(geocode_loc, location)
                if user_loc_data:
                    user_lat, user_lng = user_loc_data.latitude, user_loc_data.longitude
            except Exception as e:
                print(f"Warning: Geocoding user location '{location}' failed: {e}", flush=True)

    distance_context = f" within {maxDistance} miles" if maxDistance else ""
    query = f"top hospitals medical centers {search_location_str}{distance_context} treating {diagnosis} {equipment or ''}"

    headers = {
        "X-API-Key": ydc_api_key,
    }
    
    payload = {
        "query": query,
        "count": 10
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get("https://ydc-index.io/v1/search", headers=headers, params=payload)
            resp.raise_for_status()
            data = resp.json()
            
            # Extract standard web search results — handle both v1 (hits) and v2 (results.web) shapes
            web_results = data.get("hits", []) or data.get("results", {}).get("web", [])

            # If first query returned nothing, retry with a simpler query (drop equipment/distance)
            if not web_results:
                simple_query = f"top hospitals medical centers {search_location_str} treating {diagnosis}"
                print(f"[search_hospitals] First query returned 0 results, retrying with simplified query...", flush=True)
                retry_resp = await client.get("https://ydc-index.io/v1/search", headers=headers, params={"query": simple_query, "count": 10})
                retry_resp.raise_for_status()
                retry_data = retry_resp.json()
                web_results = retry_data.get("hits", []) or retry_data.get("results", {}).get("web", [])
                print(f"[search_hospitals] Retry returned {len(web_results)} results", flush=True)

            all_text = ""
            for hit in web_results:
                snippets = hit.get("snippets", [])
                if snippets:
                    all_text += " ".join(snippets) + "\n"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] [search_hospitals] You.com Search Snippets:\n{all_text[:300]}...\n", flush=True)

            import random

            centers = []
            seen_names = set()

            # ── Gemini batch call: clean names + proper rationales for all hits ──────
            gemini_api_key_g = os.getenv("GEMINI_API_KEY")
            ai_enriched: list = []
            if gemini_api_key_g and web_results:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_api_key_g)
                    g_model = genai.GenerativeModel("gemini-2.5-flash")
                    raw_for_gemini = [
                        {
                            "title": h.get("title", ""),
                            "url": h.get("url", ""),
                            "snippet": (h.get("description", "") or " ".join(h.get("snippets", [])))[:400],
                        }
                        for h in web_results[:10]
                    ]
                    g_prompt = (
                        f'You are a medical facility data extractor. These are web search results for '
                        f'hospitals treating "{diagnosis}" near {search_location_str}.\n\n'
                        f'STRICT RULES for "name":\n'
                        f'- Must be the official name of a HOSPITAL, MEDICAL CENTER, or CLINIC institution\n'
                        f'- NEVER use service or department names as the facility name. '
                        f'  Examples of INVALID names: "Interventional Radiology", "Imaging Services", '
                        f'  "Imaging and Radiology", "Home", "CT Scan", "MRI Services"\n'
                        f'- If the page title is a service/department, extract the institution name from '
                        f'  the URL domain or snippet instead. '
                        f'  Example: title="Imaging Services", domain="tmh.org" → name="Tallahassee Memorial Hospital"\n'
                        f'- Strip location prefixes from names. '
                        f'  Example: "Tallahassee, FL • American Health Imaging" → "American Health Imaging"\n'
                        f'- If two results belong to the SAME institution, use the EXACT same name for both\n'
                        f'- If no valid institution name can be determined at all, return null for that entry\n\n'
                        f'For each result also extract:\n'
                        f'"rationale": 2–3 concise sentences explaining why this facility is a strong '
                        f'match for treating {diagnosis}\n\n'
                        f'Input:\n{json.dumps(raw_for_gemini)}\n\n'
                        f'Return ONLY a JSON array, same length and order as input, no markdown:\n'
                        f'[{{"name": "..." or null, "rationale": "..."}}]'
                    )
                    def _call_gemini():
                        return g_model.generate_content(g_prompt)
                    g_resp = await asyncio.to_thread(_call_gemini)
                    g_text = g_resp.text.strip()
                    for prefix in ("```json", "```"):
                        if g_text.startswith(prefix):
                            g_text = g_text[len(prefix):]
                    if g_text.endswith("```"):
                        g_text = g_text[:-3]
                    g_text = g_text.strip()
                    # Recover from truncated JSON by trimming to the last complete object
                    if not g_text.endswith("]"):
                        last_close = g_text.rfind("},")
                        if last_close != -1:
                            g_text = g_text[:last_close + 1] + "]"
                    ai_enriched = json.loads(g_text)
                    print(f"[search_hospitals] Gemini enriched {len(ai_enriched)} entries", flush=True)
                except Exception as e:
                    print(f"[search_hospitals] Gemini enrichment failed, using raw titles: {e}", flush=True)
                    ai_enriched = []

            # Pad so we can safely index by position
            while len(ai_enriched) < len(web_results):
                ai_enriched.append({})

            for i, hit in enumerate(web_results):
                if len(centers) >= 10:
                    break

                gd = ai_enriched[i] if i < len(ai_enriched) else {}

                # Skip entries Gemini couldn't resolve to a real institution name
                if gd.get("name") is None and i < len(ai_enriched):
                    continue

                name = (gd.get("name") or "").strip()
                if not name:
                    # Gemini call failed entirely — use title as last resort
                    name = hit.get("title", "").split(" | ")[0].split(" - ")[0].strip()
                name = name.replace("...", "").strip() or f"Medical Center {len(centers) + 1}"

                if name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())

                url = hit.get("url", "")

                # Rationale: Gemini output first, fall back to raw snippet
                rationale = (gd.get("rationale") or "").strip()
                if not rationale:
                    rationale = hit.get("description", "") or " ".join(hit.get("snippets", []))
                if not rationale:
                    rationale = "Specialized care facility."

                # --- Geopy Coordinates ---
                h_lat = user_lat + random.uniform(-0.06, 0.06)
                h_lng = user_lng + random.uniform(-0.06, 0.06)
                try:
                    geo_query = f"{name}, {search_location_str}"
                    h_loc_data = await asyncio.to_thread(geocode_loc, geo_query)
                    if h_loc_data:
                        h_lat, h_lng = h_loc_data.latitude, h_loc_data.longitude
                    else:
                        h_loc_data_fallback = await asyncio.to_thread(geocode_loc, name)
                        if h_loc_data_fallback:
                            h_lat, h_lng = h_loc_data_fallback.latitude, h_loc_data_fallback.longitude
                except Exception as e:
                    print(f"Geocoding hospital '{name}' failed: {e}", flush=True)

                # --- OSRM ETA ---
                travel_str = f"{max(1, (i * 15) // 60)}h {(i * 15) % 60}m"
                try:
                    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{user_lng},{user_lat};{h_lng},{h_lat}?overview=false"
                    osrm_resp = await client.get(osrm_url)
                    if osrm_resp.status_code == 200:
                        route_data = osrm_resp.json()
                        if route_data.get("routes"):
                            dur = route_data["routes"][0].get("duration", 0)
                            hours, minutes = int(dur // 3600), int((dur % 3600) // 60)
                            travel_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                            print(f"[OSRM] ETA for '{name}': {travel_str}", flush=True)
                except Exception as e:
                    print(f"OSRM ETA failed for {name}: {e}", flush=True)

                centers.append({
                    "name": name,
                    "url": url,
                    "capability": str(99 - i) + "%",
                    "travel": travel_str,
                    "reason": rationale,
                    "lat": h_lat,
                    "lng": h_lng,
                })
            
            if centers:
                return {"centers": centers}
            else:
                raise ValueError("No results found in You.com Search")
            
    except Exception as e:
        print(f"Failed to fetch or parse You.com data: {e}")
        import traceback
        import random
        traceback.print_exc()
        # Gemini fallback: generate real hospital suggestions when You.com fails entirely
        gemini_api_key_fb = os.getenv("GEMINI_API_KEY")
        if gemini_api_key_fb:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key_fb)
                g_model_fb = genai.GenerativeModel("gemini-2.5-flash")
                eq_hint = f" with capabilities: {equipment}" if equipment else ""
                fb_prompt = (
                    f'List 5 real hospitals or medical centers near {search_location_str} '
                    f'known for treating "{diagnosis}"{eq_hint}. '
                    f'Return ONLY a valid JSON array (no markdown fences), each item:\n'
                    f'{{"name":"Institution Name","url":"https://official-website.org",'
                    f'"rationale":"2-3 sentences on why this facility excels at treating {diagnosis}",'
                    f'"lat":12.345,"lng":-67.890}}\n'
                    f'Use real approximate coordinates. Do NOT use 0.0 placeholder values.'
                )
                def _fb_gemini():
                    return g_model_fb.generate_content(fb_prompt)
                fb_resp = await asyncio.to_thread(_fb_gemini)
                fb_text = fb_resp.text.strip()
                for prefix in ("```json", "```"):
                    if fb_text.startswith(prefix):
                        fb_text = fb_text[len(prefix):]
                if fb_text.endswith("```"):
                    fb_text = fb_text[:-3]
                fb_centers_raw = json.loads(fb_text.strip())
                if isinstance(fb_centers_raw, list) and fb_centers_raw:
                    fb_centers = []
                    for i, c in enumerate(fb_centers_raw):
                        fb_centers.append({
                            "name": c.get("name", f"Medical Center {i + 1}"),
                            "url": c.get("url", ""),
                            "capability": str(99 - i * 4) + "%",
                            "travel": "TBD",
                            "reason": c.get("rationale", "Specialized care facility."),
                            "lat": c.get("lat") or (user_lat + random.uniform(-1.5, 1.5)),
                            "lng": c.get("lng") or (user_lng + random.uniform(-1.5, 1.5)),
                        })
                    print(f"[search_hospitals] Gemini fallback returned {len(fb_centers)} centers", flush=True)
                    return {"centers": fb_centers}
            except Exception as fb_e:
                print(f"[search_hospitals] Gemini fallback also failed: {fb_e}", flush=True)
        return {"centers": []}


# ──────────────────────────────────────────────────────────────────────────────
# MedGemma text endpoints: chat, synthesis, term explanation (text-only calls)
# ──────────────────────────────────────────────────────────────────────────────
def _clean_reply(reply: str, marker: str | None = None) -> str:
    """Strip echoed prompt markers, boxed answers and repeated lines from a model reply."""
    if marker and marker in reply:
        reply = reply.split(marker)[-1]
    reply = reply.split("Final Answer")[0]
    reply = re.sub(r"\\boxed\{([^}]*)\}", r"\1", reply).strip()
    reply = re.sub(r"^[^\w*#\-]+", "", reply)
    seen, lines = set(), []
    for line in (line.strip() for line in reply.split("\n")):
        key = re.sub(r"\W+", "", line.lower())
        if line and key not in seen:
            seen.add(key)
            lines.append(line)
    return "\n".join(lines).strip()


def _profile_context(profile: dict, *, include_outcome: bool) -> str:
    profile = normalize_profile(profile)
    patient, presentation = profile["patient"], profile["presentation"]
    assessment, findings, summary = profile["assessment"], profile["findings"], profile["summary"]
    lines = [
        ("Demographics", " ".join(str(v) for v in (f"{patient['age_years']}y" if patient["age_years"] is not None else None,
                                                    patient["sex"]) if v)),
        ("Comorbidities", ", ".join(patient["comorbidities"])),
        ("Chief complaint", presentation["chief_complaint"]),
        ("History", (presentation["hpi"] or "")[:600]),
        ("Imaging", f"{profile['study'].get('modality') or ''} {profile['study'].get('body_region') or ''}".strip()),
        ("Imaging findings", "; ".join(findings["imaging_findings"])),
        ("Lab findings", "; ".join(findings["lab_findings"])),
        ("Diagnosis", assessment["diagnosis_primary"]),
        ("Differential", ", ".join(assessment["differential"])),
    ]
    if include_outcome:
        lines += [("Treatment", "; ".join(profile["management"]["treatments"] + profile["management"]["procedures"])),
                  ("Outcome", profile["outcome"]["detail"]), ("Follow-up", profile["outcome"]["follow_up"]),
                  ("Authors' conclusion", summary["conclusion"])]
    return "\n".join(f"- {label}: {value}" for label, value in lines if value)


@app.post("/chat_twin")
async def chat_twin(
    query: str = Form(...),
    case_text: str = Form(...),
    current_profile: Optional[str] = Form(default=None),
    twin_profile: Optional[str] = Form(default=None),
    history: Optional[str] = Form(default=None),
):
    """
    Grounded Q&A (retrieval-augmented generation): MedGemma answers using only
    the retrieved twin case and the current patient's case report.
    ``history`` is a JSON list of {role, content} prior turns.
    """
    twin_ctx = case_text[:1500]
    if twin_profile:
        try:
            twin_ctx = _profile_context(json.loads(twin_profile), include_outcome=True) + f"\n- Source narrative: {case_text[:900]}"
        except (ValueError, TypeError):
            pass
    current_ctx = ""
    if current_profile:
        try:
            current_ctx = _profile_context(json.loads(current_profile), include_outcome=False)
        except (ValueError, TypeError):
            logger.warning("Ignoring unparsable current_profile")
    turns: list[dict[str, str]] = []
    if history:
        try:
            turns = [{"role": t["role"], "content": str(t["content"])[:1200]} for t in json.loads(history)
                     if isinstance(t, dict) and t.get("role") in ("user", "assistant")][-6:]
        except (ValueError, TypeError, KeyError):
            turns = []
    prompt = (
        "You are a clinical reasoning assistant helping a clinician compare a current patient with a "
        "similar published case (a 'twin'). Answer using only the two case summaries below; if they do not "
        "contain the answer, say so. Keep the answer under 120 words, use Markdown bullets and **bold** key "
        "terms, and do not give a definitive treatment order for the current patient.\n\n"
        f"## Twin case (published case report)\n{twin_ctx}\n\n"
        f"## Current patient\n{current_ctx or '- Not provided'}\n\n"
        f"Question: {query}"
    )
    started = time.perf_counter()
    try:
        reply = await asyncio.to_thread(query_text, prompt, model=MEDGEMMA_MODEL, max_tokens=400, history=turns)
        reply = _clean_reply(reply) or "I don't have enough information in these two cases to answer that."
        return {"reply": reply, "trace": [_step(MEDGEMMA_MODEL, "Grounded answer from twin + current case", started)]}
    except Exception as e:
        logger.exception("MedGemma chat failed")
        raise HTTPException(status_code=502, detail=f"MedGemma could not answer right now: {e}")


@app.post("/enhance_profile")
async def enhance_profile(
    profile_json: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """MedGemma synthesis of the case (text-only) plus an imaging read when an image is supplied."""
    try:
        profile_data = json.loads(profile_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Profile is not valid JSON: {e}")
    ctx = _profile_context(profile_data, include_outcome=False)
    synthesis_prompt = (
        "You are an expert clinical reasoning assistant. Review this structured case report and write an "
        "'AI Clinical Synthesis': 3-4 short Markdown bullets with insights that are NOT a restatement of the "
        "text — e.g. likely differentials to consider, risk factors, red flags, and what information is "
        "missing. Bold key terms. No introduction.\n\n"
        f"## Case report\n{ctx}"
    )
    tasks = [asyncio.to_thread(query_text, synthesis_prompt, model=MEDGEMMA_MODEL, max_tokens=350)]
    trace_labels = [(MEDGEMMA_MODEL, "Clinical synthesis (text-only)")]
    if file and file.filename:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        imaging_prompt = (
            "You are an expert radiologist/clinician. Describe the key visible findings in this image and how "
            "they relate to the case context below, in 2-3 short sentences. Bold key terms. Say what is unclear."
            f"\n\n## Case context\n{ctx[:800]}"
        )
        tasks.append(asyncio.to_thread(query_medgemma, img, prompt=imaging_prompt, max_tokens=250))
        trace_labels.append((MEDGEMMA_MODEL, "Image read in case context"))
    started = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if isinstance(results[0], Exception):
        logger.error("MedGemma synthesis failed: %s", results[0])
        raise HTTPException(status_code=502, detail="MedGemma could not generate the clinical synthesis. Please retry.")
    synthesis = _clean_reply(results[0])
    imaging = None
    if len(results) > 1 and not isinstance(results[1], Exception):
        imaging = _clean_reply(results[1][0].get("generated_text", ""))
    trace = [_step(model, task, started) for model, task in trace_labels]
    return {"synthesis": synthesis or "Unable to generate clinical synthesis.", "imaging_context": imaging, "trace": trace}


LOCAL_LANGUAGES = {"hi": "Hindi", "mr": "Marathi"}


def _local_language_prompt(english: str, term: str, language: str) -> str:
    return (
        f"Rewrite the following explanation of the medical term \"{term}\" for a patient who speaks {language}. "
        f"Write in simple, everyday spoken {language} in Devanagari script, the way a kind family doctor would "
        "explain it: 2-3 short sentences, no difficult words. Keep the original English medical term once in "
        "brackets so the patient can match it to their report. Use only the facts in the explanation below; "
        "do not add causes, treatments or advice that it does not contain. Reply with the explanation only.\n\n"
        f"Explanation:\n{english}"
    )


@app.post("/explain_selection")
async def explain_selection(
    selected_text: str = Form(...),
    context: str = Form(default=""),
    audience: str = Form(default="clinician"),
    language: str = Form(default="en"),
):
    """MedGemma explains a highlighted term in context, for a clinician or in plain language.

    With ``language`` hi/mr, a two-model chain runs: MedGemma writes the plain-
    language English explanation (medical accuracy), then Gemma 4 rewrites it in
    simple Hindi or Marathi (multilingual fluency) without adding facts.
    """
    if language not in ("en", *LOCAL_LANGUAGES):
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    if language != "en":
        audience = "patient"
    term = selected_text.strip()[:200]
    level = ("in plain language a patient could understand, avoiding jargon" if audience == "patient"
             else "for a clinical audience")
    prompt = (
        f"Explain the medical term or phrase \"{term}\" in 1-2 sentences, {level}. "
        f"Use the surrounding context only to pick the right meaning: \"{context[:500].strip()}\". "
        "Start directly with the explanation."
    )
    started = time.perf_counter()
    try:
        raw = await asyncio.to_thread(query_text, prompt, model=MEDGEMMA_MODEL, max_tokens=160)
    except Exception as e:
        logger.exception("MedGemma explain failed")
        raise HTTPException(status_code=502, detail=f"MedGemma explanation unavailable: {e}")
    sentences = re.split(r"(?<=[.!?])\s+", _clean_reply(raw))
    english = " ".join(sentences[:3]).strip()
    trace = [_step(MEDGEMMA_MODEL, "Explain highlighted term" + (" in plain English" if audience == "patient" else ""),
                   started)]
    if language == "en":
        return {"explanation": english, "language": "en", "trace": trace}

    name = LOCAL_LANGUAGES[language]
    started = time.perf_counter()
    try:
        local = await asyncio.to_thread(query_text, _local_language_prompt(english, term, name), model=GEMMA_MODEL,
                                        max_tokens=400, temperature=0.2, thinking=False)
    except Exception as e:
        logger.exception("Gemma 4 translation failed")
        raise HTTPException(status_code=502, detail=f"Gemma 4 could not write the {name} explanation: {e}")
    trace.append(_step(GEMMA_MODEL, f"Rewrite for a patient in simple {name}", started))
    return {"explanation": _clean_reply(local), "explanation_en": english, "language": language, "trace": trace}


# ──────────────────────────────────────────────────────────────────────────────
# /extract  – unstructured notes + images -> structured case report
# ──────────────────────────────────────────────────────────────────────────────
IMAGE_READ_PROMPT = (
    "Describe the key findings visible in this medical image as a short list of findings, the way a "
    "specialist would dictate them (modality and body region first, then findings). Mention notable "
    "normal findings only if relevant. Do not give a final diagnosis. Maximum 80 words."
)


def _image_read_prompt(collection: str | None) -> str:
    """Tell MedGemma what kind of image it is (from MedSigLIP's routing) so it reads it as that.

    Without this, MedGemma has guessed wrong, e.g. describing bone X-ray findings for a
    photograph of psoriatic legs.
    """
    if not collection:
        return IMAGE_READ_PROMPT
    kind = COLLECTIONS[collection]["zero_shot"].removeprefix("a ").removeprefix("an ")
    return (
        f"This image is {COLLECTIONS[collection]['zero_shot']}. Describe the key findings visible in it as a "
        f"short list, the way a specialist reading a {kind} would dictate them. Mention notable normal "
        "findings only if relevant. Do not give a final diagnosis. Maximum 80 words."
    )


@app.post("/extract")
async def extract(
    images: Optional[List[UploadFile]] = File(default=None),
    notes: str = Form(default=""),
    notes_file: Optional[UploadFile] = File(default=None),
):
    """
    Build a clean structured case report:
      1. MedSigLIP zero-shot labels the image type (no training needed).
      2. MedGemma reads each image (up to 3) into findings text.
      3. Gemma 4 turns notes + image reads into the case-report JSON schema.
    The regex extractor runs only if Gemma 4 is unreachable, and the response says so.
    """
    notes_text = notes
    if notes_file:
        try:
            raw = await notes_file.read()
            document_text = extract_document_text(notes_file.filename, notes_file.content_type, raw)
            notes_text = (notes_text + "\n" + document_text).strip()
        except DocumentExtractionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    pil_images: list[Image.Image] = []
    image_names: list[str] = []
    for upload in images or []:
        if not upload.filename:
            continue
        image_names.append(upload.filename)
        try:
            pil_images.append(Image.open(io.BytesIO(await upload.read())).convert("RGB"))
        except Exception:
            logger.warning("Skipping unreadable image %s", upload.filename)
    if not notes_text.strip() and not pil_images:
        raise HTTPException(status_code=400, detail="Add clinical notes or an image to build a case report.")

    trace: list[dict] = []
    routing = None
    image_reads: list[str] = []
    if pil_images:
        started = time.perf_counter()
        try:
            chosen, scores = await asyncio.to_thread(_route_collection, pil_images[0])
            routing = {"collection": chosen, "scores": scores}
            trace.append(_step("MedSigLIP", f"Zero-shot image type: {scores[0]['label']}", started))
        except Exception as exc:
            trace.append(_step("MedSigLIP", f"Zero-shot classification unavailable: {exc}", started))
        started = time.perf_counter()
        read_prompt = _image_read_prompt(routing["collection"] if routing else None)
        # temperature 0: the same image gives the same read, so demos are repeatable.
        reads = await asyncio.gather(*[asyncio.to_thread(query_medgemma, img, prompt=read_prompt, max_tokens=220, temperature=0)
                                       for img in pil_images[:3]], return_exceptions=True)
        for read in reads:
            if not isinstance(read, Exception):
                text = _clean_reply(read[0].get("generated_text", ""))
                if text:
                    image_reads.append(text)
        trace.append(_step(MEDGEMMA_MODEL, f"Read {len(image_reads)} image(s) into findings", started,
                           output=image_reads))

    case_id, image_id = str(uuid.uuid4()), str(uuid.uuid4())
    base = empty_profile(article_id=case_id, image_id=image_id, modality=None, body_region=None)
    base["provenance"].update({"dataset_name": None, "pmc_id": None})
    method = GEMMA_MODEL
    try:
        fields, step = await asyncio.to_thread(extract_profile, intake_prompt(notes_text, image_reads))
        trace.append(step)
        profile = merge_profile(base, fields)
        # A new patient has no outcome yet, whatever the notes imply.
        profile["outcome"] = base["outcome"]
    except Exception as exc:
        logger.exception("Gemma 4 extraction failed; using regex fallback")
        method = "regex-fallback"
        profile = _regex_profile(notes_text, image_names, "\n".join(image_reads))
        trace.append({"model": "regex", "task": f"Fallback extraction (Gemma 4 unavailable: {exc})", "ms": 0})
    profile["case_id"], profile["image_id"], profile["profile_id"] = case_id, image_id, f"{case_id}:{image_id}"
    if routing and routing["collection"]:
        spec = COLLECTIONS[routing["collection"]]
        profile["study"]["collection"] = routing["collection"]
        profile["study"]["modality"] = profile["study"].get("modality") or spec["modality"]
        profile["study"]["body_region"] = profile["study"].get("body_region") or spec["body_region"]
    if image_reads:
        profile["extra_fields"]["ai_image_read"] = image_reads
    return {"profile": profile, "method": method, "routing": routing, "trace": trace}


def _regex_profile(text: str, image_names: list[str], medgemma_insight: str = "") -> dict:
    """
    Last-resort keyword extractor used only when Gemma 4 is unreachable.
    """
    case_id = str(uuid.uuid4())
    image_id = str(uuid.uuid4())

    profile: dict = {
        "profile_id": f"{case_id}:{image_id}",
        "case_id": case_id,
        "image_id": image_id,
        "patient": {
            "age_years": None,
            "sex": None,
            "immunocompromised": None,
            "weight_kg": None,
            "comorbidities": [],
            "medications": [],
            "allergies": None,
        },
        "presentation": {
            "chief_complaint": None,
            "symptom_duration": None,
            "hpi": None,
            "pmh": None,
        },
        "study": {
            "modality": None,
            "body_region": None,
            "view_position": None,
            "radiology_region": None,
            "caption": None,
            "image_type": None,
            "image_subtype": None,
            "image_url": None,
            "storage_path": None,
        },
        "assessment": {
            "diagnosis_primary": None,
            "suspected_primary": [],
            "differential": [],
            "urgency": None,
            "infectious_concern": None,
            "icu_candidate": None,
        },
        "findings": {
            "lungs": {
                "consolidation_present": "no",
                "consolidation_locations": [],
                "consolidation_extent": "unknown",
                "atelectasis_present": "no",
                "atelectasis_locations": [],
                "edema_present": "no",
                "edema_pattern": "unknown",
            },
            "pleura": {
                "effusion_present": "no",
                "effusion_side": "unknown",
                "effusion_size": "unknown",
                "pneumothorax_present": "no",
                "pneumothorax_side": "unknown",
            },
            "cardiomediastinal": {
                "cardiomegaly": "no",
                "mediastinal_widening": "no",
            },
            "devices": {
                "lines_tubes_present": "no",
                "device_list": [],
            },
        },
        "summary": {
            "one_liner": None,
            "key_points": [],
            "red_flags": [],
        },
        "outcome": {
            "success": None,
            "detail": None,
        },
        "provenance": {
            "dataset_name": None,
            "pmc_id": None,
            "pmid": None,
            "doi": None,
            "article_title": None,
            "journal": None,
            "year": None,
            "authors": [],
            "license": None,
            "source_url": None,
        },
        "tags": {
            "ml_labels": [],
            "gt_labels": [],
            "keywords": [],
            "mesh_terms": [],
        },
    }

    # ── Patient ──────────────────────────────────────────────────────────────
    age_m = re.search(r"(\d{1,3})\s*[- ]?(?:year|yr)s?[- ]?old", text, re.I)
    if age_m:
        profile["patient"]["age_years"] = int(age_m.group(1))

    if re.search(r"\bfemale\b|\bwoman\b", text, re.I):
        profile["patient"]["sex"] = "female"
    elif re.search(r"\bmale\b|\bman\b", text, re.I):
        profile["patient"]["sex"] = "male"

    if re.search(r"immunocompromised|immunosuppressed", text, re.I):
        profile["patient"]["immunocompromised"] = "yes"
    elif text.strip():
        profile["patient"]["immunocompromised"] = "no"

    comorbidity_map = [
        (r"hypertension|HTN", "hypertension"),
        (r"type 2 diabet|T2DM|DM2", "type 2 diabetes"),
        (r"type 1 diabet|T1DM|DM1", "type 1 diabetes"),
        (r"atrial fibrillation|AF\b|AFib", "atrial fibrillation"),
        (r"heart failure|CHF", "heart failure"),
        (r"COPD|chronic obstructive", "COPD"),
        (r"asthma", "asthma"),
        (r"cirrhosis|liver cirrhosis", "liver cirrhosis"),
        (r"hepatocellular carcinoma|HCC", "hepatocellular carcinoma"),
        (r"chronic kidney|CKD", "chronic kidney disease"),
        (r"coronary artery disease|CAD", "coronary artery disease"),
        (r"obesity", "obesity"),
    ]
    comorbidities = [label for pattern, label in comorbidity_map if re.search(pattern, text, re.I)]
    profile["patient"]["comorbidities"] = comorbidities

    if re.search(r"no known allerg", text, re.I):
        profile["patient"]["allergies"] = "no known allergies"

    # ── Presentation ─────────────────────────────────────────────────────────
    cc_m = re.search(
        r"(?:present(?:ing)? with|complaint of|admitted for|scheduled for)\s+([^.!?\n]{5,120})",
        text, re.I
    )
    if cc_m:
        profile["presentation"]["chief_complaint"] = cc_m.group(1).strip()

    dur_m = re.search(r"(?:for|over|duration of)\s+((?:\d+\s*)?(?:day|week|month|year)s?)", text, re.I)
    if dur_m:
        profile["presentation"]["symptom_duration"] = dur_m.group(1).strip()

    if len(text) > 40:
        profile["presentation"]["hpi"] = text[:600]

    if comorbidities:
        profile["presentation"]["pmh"] = ", ".join(comorbidities)

    # ── Study ────────────────────────────────────────────────────────────────
    combined = text + " " + " ".join(image_names)
    if re.search(r"ct|computed tomography", combined, re.I):
        profile["study"].update({"modality": "CT", "image_type": "radiology", "image_subtype": "ct"})
    elif re.search(r"mri", combined, re.I):
        profile["study"].update({"modality": "MRI", "image_type": "radiology", "image_subtype": "mri"})
    elif re.search(r"x[- ]?ray|cxr|chest x", combined, re.I):
        profile["study"].update({"modality": "CXR", "image_type": "radiology", "image_subtype": "x_ray"})
    elif image_names:
        profile["study"].update({"modality": "Imaging", "image_type": "radiology"})

    if re.search(r"thorax|chest|pulmonary|lung", text, re.I):
        profile["study"]["body_region"] = "thorax"
        profile["study"]["radiology_region"] = "thorax"
    elif re.search(r"abdomen|abdominal|liver", text, re.I):
        profile["study"]["body_region"] = "abdomen"
    elif re.search(r"brain|head|neuro", text, re.I):
        profile["study"]["body_region"] = "head"

    if re.search(r"\bPA\b|posteroanterior", text, re.I):
        profile["study"]["view_position"] = "PA"
    elif re.search(r"\bAP\b|anteroposterior", text, re.I):
        profile["study"]["view_position"] = "AP"

    # ── Assessment ───────────────────────────────────────────────────────────
    diag_map = [
        (r"scimitar", "scimitar syndrome"),
        (r"pneumonia", "community-acquired pneumonia"),
        (r"pulmonary embolism|PE\b", "pulmonary embolism"),
        (r"lung malignancy|lung cancer|NSCLC|SCLC", "lung malignancy"),
        (r"stroke|ischemic", "acute ischemic stroke"),
        (r"heart failure|pulmonary edema", "heart failure"),
        (r"pneumothorax", "pneumothorax"),
        (r"pleural effusion", "pleural effusion"),
        (r"aortic dissection", "aortic dissection"),
    ]
    for pattern, diag in diag_map:
        if re.search(pattern, text, re.I):
            profile["assessment"]["diagnosis_primary"] = diag
            profile["assessment"]["suspected_primary"] = [diag] + comorbidities[:2]
            break

    if re.search(r"urgent|emergency|stat", text, re.I):
        profile["assessment"]["urgency"] = "emergent"
    elif re.search(r"routine|elective|scheduled", text, re.I):
        profile["assessment"]["urgency"] = "routine"
    elif text.strip():
        profile["assessment"]["urgency"] = "semi-urgent"

    profile["assessment"]["infectious_concern"] = (
        "yes" if re.search(r"infection|sepsis|pneumonia|fever", text, re.I) else "no"
    )
    profile["assessment"]["icu_candidate"] = (
        "yes" if re.search(r"icu|intensive care|critical", text, re.I) else "no"
    )

    # ── Findings tweaks ──────────────────────────────────────────────────────
    # ── MedGemma Insight Integration ──────────────────────────────────────────
    # We combine the original text with MedGemma's findings for the regex extractor
    # to pick up confirmed findings from the image.
    analysis_text = text + "\n" + medgemma_insight

    if re.search(r"consolidation|consolidat", analysis_text, re.I):
        profile["findings"]["lungs"]["consolidation_present"] = "yes"
    if re.search(r"atelectasis|collapse", analysis_text, re.I):
        profile["findings"]["lungs"]["atelectasis_present"] = "yes"
    if re.search(r"edema|pulmonary edema", analysis_text, re.I):
        profile["findings"]["lungs"]["edema_present"] = "yes"
    if re.search(r"effusion|pleural fluid", analysis_text, re.I):
        profile["findings"]["pleura"]["effusion_present"] = "yes"
    if re.search(r"pneumothorax", analysis_text, re.I):
        profile["findings"]["pleura"]["pneumothorax_present"] = "yes"
    if re.search(r"cardiomegaly|enlarged heart|cardiomegal", analysis_text, re.I):
        profile["findings"]["cardiomediastinal"]["cardiomegaly"] = "yes"

    if medgemma_insight and not profile["summary"]["one_liner"]:
        profile["summary"]["one_liner"] = medgemma_insight[:200] + ("..." if len(medgemma_insight) > 200 else "")

    # ── Summary ──────────────────────────────────────────────────────────────
    age = profile["patient"]["age_years"]
    sex = profile["patient"]["sex"]
    diag = profile["assessment"]["diagnosis_primary"]
    cc   = profile["presentation"]["chief_complaint"]
    if age and sex and (diag or cc):
        comorbs = ", ".join(comorbidities[:3]) or "multiple comorbidities"
        profile["summary"]["one_liner"] = (
            f"{age}-year-old {sex} with {comorbs} presenting with {cc or diag}."
        )
    if diag:
        profile["summary"]["key_points"] = [f"Primary finding: {diag}"]

    # ── Extra Fields (schema expansion) ──────────────────────────────────────
    # Scan for clinical data that doesn't fit the base schema.
    # These are captured at ANY point during intake (any confidence level).
    extra_fields: dict = {}

    # Smoking / tobacco
    smoke_m = re.search(
        r"(?:smok(?:ing|er|es)|tobacco)[^\.\n]{0,60}?((?:\d+\s*)?(?:pack[- ]?year|cigarette|cigar|pipe)[^\.\n]{0,40})?",
        text, re.I
    )
    if smoke_m:
        detail = smoke_m.group(1)
        extra_fields["smoking_status"] = detail.strip() if detail and detail.strip() else "smoker"

    # Never smoked
    if re.search(r"non[- ]?smok|never smoked|no smoking", text, re.I):
        extra_fields["smoking_status"] = "non-smoker"

    # Alcohol use
    alcohol_m = re.search(r"alcohol[^\.\n]{0,80}", text, re.I)
    if alcohol_m:
        snippet = alcohol_m.group(0).strip()
        extra_fields["alcohol_use"] = snippet[:120]

    # BMI / weight / height
    bmi_m = re.search(r"BMI\s*(?:of\s*)?(\d{1,2}(?:\.\d)?)", text, re.I)
    if bmi_m:
        extra_fields["bmi"] = bmi_m.group(1)

    height_m = re.search(r"(\d{1,3})\s*(?:cm|ft|feet|inches?)", text, re.I)
    if height_m and "bmi" not in extra_fields:
        extra_fields["height"] = f"{height_m.group(1)} {height_m.group(0).split(height_m.group(1))[-1].strip()}"

    # Blood type
    blood_m = re.search(r"\b(A|B|AB|O)[+-]?\s*blood\s*type|\bblood\s*type\s*(A|B|AB|O)[+-]?\b", text, re.I)
    if blood_m:
        extra_fields["blood_type"] = (blood_m.group(1) or blood_m.group(2)).upper()

    # Family history
    fam_m = re.search(r"family\s*(?:history|hx)[^\.\n]{0,150}", text, re.I)
    if fam_m:
        extra_fields["family_history"] = fam_m.group(0).strip()[:200]

    # Occupation / employment
    occ_m = re.search(r"(?:occupation|works?\s*as|employed\s*(?:as|at)|profession)[^\.\n]{0,80}", text, re.I)
    if occ_m:
        extra_fields["occupation"] = occ_m.group(0).strip()[:120]

    # Ethnicity / race
    eth_m = re.search(
        r"(?:ethnicity|race|racial background)\s*[:\-]?\s*([A-Za-z\s\-]+)",
        text, re.I
    )
    if eth_m:
        extra_fields["ethnicity"] = eth_m.group(1).strip()[:60]

    # Vaccination status
    vax_m = re.search(r"(?:vaccin|immuniz)[^\.\n]{0,80}", text, re.I)
    if vax_m:
        extra_fields["vaccination"] = vax_m.group(0).strip()[:120]

    # Travel history
    travel_m = re.search(r"(?:travel(?:led|ed)?\s*(?:to|from)|recent\s*travel)[^\.\n]{0,100}", text, re.I)
    if travel_m:
        extra_fields["travel_history"] = travel_m.group(0).strip()[:150]

    # Functional status / ADLs
    func_m = re.search(r"(?:functional status|ADLs?|activities of daily|ambulates?|independent)[^\.\n]{0,80}", text, re.I)
    if func_m:
        extra_fields["functional_status"] = func_m.group(0).strip()[:120]

    # Code status / DNR
    code_m = re.search(r"(?:code\s*status|full\s*code|DNR|DNI|comfort\s*care)[^\.\n]{0,60}", text, re.I)
    if code_m:
        extra_fields["code_status"] = code_m.group(0).strip()[:80]

    # Social history (catch-all if not already captured)
    social_m = re.search(r"social\s*(?:history|hx)[^\.\n]{0,200}", text, re.I)
    if social_m:
        extra_fields["social_history"] = social_m.group(0).strip()[:250]

    profile = normalize_profile(profile)
    profile["extra_fields"] = extra_fields
    return profile

# ──────────────────────────────────────────────────────────────────────────────
# /analyze_hospital_page  – CrewAI Agent endpoint for doctor extraction
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/analyze_hospital_page")
async def analyze_hospital_page(
    url: str = Form(...),
    diagnosis: str = Form(...),
    hospital_name: str = Form(default=""),
    location: str = Form(default="")
):
    """
    Triggers the CrewAI agent to scrape the given hospital URL and return doctors
    specializing in the given diagnosis.
    """
    import asyncio
    import agents
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [analyze_hospital_page] Received request for {hospital_name} (location: {location})")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [analyze_hospital_page] LANGCHAIN_TRACING_V2={os.getenv('LANGCHAIN_TRACING_V2')}")
    
    try:
        # Run CrewAI synchronously inside an async thread to prevent blocking Uvicorn
        data = await asyncio.to_thread(agents.analyze_hospital_staff, url, diagnosis, hospital_name, location)
        return {"specialists": data}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Agent endpoint failed: {e}")
        return {"specialists": [], "error": str(e)}
