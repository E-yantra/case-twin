"""Strict clients for the local OpenAI-compatible gateway and MedSigLIP service."""

import base64
import io
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from PIL import Image

# The preparation/index commands run from the repository root, while Compose
# injects the same file as environment variables. Loading this adjacent file
# makes both execution paths use one backend configuration source.
load_dotenv(Path(__file__).with_name(".env"))

GATEWAY_BASE_URL = os.getenv("LOCAL_GATEWAY_BASE_URL", os.getenv("LOCAL_LLM_BASE_URL", "")).rstrip("/")
if GATEWAY_BASE_URL and not GATEWAY_BASE_URL.endswith("/v1"):
    GATEWAY_BASE_URL = f"{GATEWAY_BASE_URL}/v1"
GATEWAY_API_KEY = os.getenv("LOCAL_GATEWAY_API_KEY", os.getenv("LOCAL_LLM_API_KEY", "")).strip()
MEDGEMMA_MODEL = os.getenv("MEDGEMMA_MODEL", os.getenv("LOCAL_MODEL_NAME", "medgemma")).strip()
# Two-image comparison: the 27B reads chest X-rays more reliably than the 4B 1.5,
# which tended to copy its first-image read onto the second image.
MEDGEMMA_COMPARISON_MODEL = os.getenv("MEDGEMMA_COMPARISON_MODEL", "medgemma").strip() or "medgemma"
# Bounding boxes stay on MedGemma 1.5, which is trained for CXR localisation.
MEDGEMMA_LOCALIZATION_MODEL = os.getenv("MEDGEMMA_LOCALIZATION_MODEL", "medgemma-1.5").strip() or "medgemma-1.5"
GEMMA_MODEL = os.getenv("GEMMA_MODEL", os.getenv("LOCAL_GENERAL_MODEL_NAME", "gemma-4")).strip()
TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "qwen3-embedding-8b").strip()
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3").strip()
MEDSIGLIP_BASE_URL = os.getenv("MEDSIGLIP_BASE_URL", os.getenv("MEDSIGLIP_ENDPOINT", "")).rstrip("/")
MEDSIGLIP_API_KEY = os.getenv("MEDSIGLIP_API_KEY", "").strip()


def _image_b64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((768, 768))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("MedSigLIP returned an invalid or zero vector")
    return [value / norm for value in vector]


def _gateway_headers() -> dict[str, str]:
    if not GATEWAY_BASE_URL:
        raise RuntimeError("LOCAL_GATEWAY_BASE_URL is not configured")
    if not GATEWAY_API_KEY:
        raise RuntimeError("LOCAL_GATEWAY_API_KEY is not configured")
    return {"Authorization": f"Bearer {GATEWAY_API_KEY}"}


def query_local_model(
    prompt: str, *, model: str = MEDGEMMA_MODEL, image: Image.Image | None = None,
    images: list[Image.Image] | None = None, max_tokens: int = 300,
    stop_sequences: list[str] | None = None, temperature: float | None = None,
    image_first: bool = False, history: list[dict[str, str]] | None = None,
    json_schema: dict | None = None, thinking: bool | None = None, timeout: float = 120.0,
    base_url: str | None = None, api_key: str | None = None,
) -> list[dict[str, str]]:
    """Call the OpenAI-compatible chat API and preserve the legacy response shape.

    With no image the request is text-only; MedGemma answers text questions
    directly, so callers must not send placeholder images. ``json_schema``
    constrains the reply to that JSON Schema (llama.cpp grammar), and
    ``thinking=False`` disables Gemma 4's reasoning pass for faster extraction.
    ``base_url``/``api_key`` target another OpenAI-compatible server instead of the gateway.
    """
    if image is not None and images is not None:
        raise ValueError("Pass either image or images, not both")
    attached = images if images is not None else ([image] if image is not None else [])
    if attached:
        content: Any = [] if image_first else [{"type": "text", "text": prompt}]
        for item in attached:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_image_b64(item)}"}})
        if image_first:
            content.append({"type": "text", "text": prompt})
    else:
        content = prompt
    payload: dict[str, Any] = {
        "model": model,
        "messages": [*(history or []), {"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences
    if temperature is not None:
        payload["temperature"] = temperature
    if json_schema is not None:
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": json_schema}}
    if thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}
    if base_url:
        url, headers = f"{base_url.rstrip('/')}/chat/completions", ({"Authorization": f"Bearer {api_key}"} if api_key else {})
    else:
        url, headers = f"{GATEWAY_BASE_URL}/chat/completions", _gateway_headers()
    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    try:
        choice = response.json()["choices"][0]
        content_text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected local gateway chat response") from exc
    if not isinstance(content_text, str) or not content_text.strip():
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        raise ValueError(
            f"Local gateway returned empty chat content for model {model!r} "
            f"(finish_reason={finish_reason!r})"
        )
    return [{"generated_text": content_text}]


def query_medgemma(image: Image.Image, prompt: str = "Describe this chest X-ray.", max_tokens: int = 200,
                   stop_sequences: list[str] | None = None, temperature: float | None = None) -> list[dict[str, str]]:
    return query_local_model(prompt, model=MEDGEMMA_MODEL, image=image, max_tokens=max_tokens,
                             stop_sequences=stop_sequences, temperature=temperature)


def query_medgemma_comparison(current_image: Image.Image, historical_image: Image.Image,
                              prompt: str, max_tokens: int = 400) -> list[dict[str, str]]:
    return query_local_model(
        prompt, model=MEDGEMMA_COMPARISON_MODEL,
        images=[current_image, historical_image], max_tokens=max_tokens, temperature=0,
    )


def query_medgemma_read(image: Image.Image, prompt: str, max_tokens: int = 220) -> list[dict[str, str]]:
    """One image, one read: used twice in parallel so neither read can see the other."""
    return query_local_model(prompt, model=MEDGEMMA_COMPARISON_MODEL, image=image, max_tokens=max_tokens, temperature=0)


def query_medgemma_localization(image: Image.Image, prompt: str) -> list[dict[str, str]]:
    return query_local_model(
        prompt, model=MEDGEMMA_LOCALIZATION_MODEL, image=image,
        image_first=True, max_tokens=900, temperature=0,
    )


def generate_embedding(image: Image.Image) -> list[float]:
    """Generate one non-empty, normalized image vector; never silently substitute one."""
    if not MEDSIGLIP_BASE_URL:
        raise RuntimeError("MEDSIGLIP_BASE_URL is not configured")
    if not MEDSIGLIP_API_KEY:
        raise RuntimeError("MEDSIGLIP_API_KEY is not configured")
    response = httpx.post(
        f"{MEDSIGLIP_BASE_URL}/v1/embed_image",
        headers={"Authorization": f"Bearer {MEDSIGLIP_API_KEY}"},
        json={"images_b64": [_image_b64(image)]}, timeout=120.0,
    )
    response.raise_for_status()
    payload: Any = response.json()
    vector: Any = payload
    if isinstance(payload, dict):
        vector = payload.get("embedding") or payload.get("embeddings") or payload.get("vectors")
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        vector = vector[0]
    if not isinstance(vector, list) or not vector:
        raise ValueError("Unexpected MedSigLIP embedding response")
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("MedSigLIP returned a non-numeric vector") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("MedSigLIP returned non-finite vector values")
    return _normalise(values)


def query_text(prompt: str, *, model: str = MEDGEMMA_MODEL, max_tokens: int = 300, **kwargs: Any) -> str:
    """Text-only chat call returning the reply string."""
    return query_local_model(prompt, model=model, max_tokens=max_tokens, **kwargs)[0]["generated_text"].strip()


def _medsiglip_post(path: str, payload: dict) -> Any:
    if not MEDSIGLIP_BASE_URL:
        raise RuntimeError("MEDSIGLIP_BASE_URL is not configured")
    if not MEDSIGLIP_API_KEY:
        raise RuntimeError("MEDSIGLIP_API_KEY is not configured")
    response = httpx.post(f"{MEDSIGLIP_BASE_URL}{path}", headers={"Authorization": f"Bearer {MEDSIGLIP_API_KEY}"},
                          json=payload, timeout=120.0)
    response.raise_for_status()
    return response.json()


def _vector(values: Any) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError("Embedding service returned no vector")
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding service returned non-finite vector values")
    return _normalise(vector)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


def medsiglip_text_embedding(text: str) -> list[float]:
    """MedSigLIP text tower: lands in the same space as its image vectors.

    The text encoder takes at most 64 tokens and the server returns HTTP 500
    instead of truncating, so the text is kept short and shortened further on a 500.
    """
    for limit in (180, 110, 60):
        try:
            payload = _medsiglip_post("/v1/embed_text", {"texts": [_shorten(text, limit)]})
            break
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 500 or limit == 60:
                raise
    vectors = payload.get("embeddings") if isinstance(payload, dict) else payload
    return _vector(vectors[0] if isinstance(vectors, list) and vectors and isinstance(vectors[0], list) else vectors)


def medsiglip_classify(image: Image.Image, labels: list[str]) -> list[dict[str, float | str]]:
    """Zero-shot labels for one image, highest first. Scores are independent sigmoids."""
    payload = _medsiglip_post("/v1/classify", {"image_b64": _image_b64(image), "labels": labels})
    raw = payload.get("scores", payload.get("results", payload)) if isinstance(payload, dict) else payload
    if isinstance(raw, dict):
        items = [{"label": str(label), "score": float(score)} for label, score in raw.items()]
    elif isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        items = [{"label": str(item.get("label")), "score": float(item.get("score", item.get("probability", 0)))} for item in raw]
    elif isinstance(raw, list) and len(raw) == len(labels):
        items = [{"label": label, "score": float(score)} for label, score in zip(labels, raw)]
    else:
        raise ValueError("Unexpected MedSigLIP classify response")
    return sorted(items, key=lambda item: item["score"], reverse=True)


def text_embeddings(texts: list[str], *, query: bool = False) -> list[list[float]]:
    """Qwen3 text embeddings via the gateway; queries get the model's instruction prefix.

    The llama.cpp embedding server returned all-NaN vectors when several
    sequences share a batch, so texts are sent one per request and any invalid
    reply (e.g. co-batched with another client's request) is retried.
    """
    if query:
        texts = [f"Instruct: Given a clinical case description, retrieve similar published case reports\nQuery: {text}"
                 for text in texts]

    def call(batch: list[str]) -> list[Any]:
        response = httpx.post(f"{GATEWAY_BASE_URL}/embeddings", headers=_gateway_headers(),
                              json={"model": TEXT_EMBEDDING_MODEL, "input": batch}, timeout=180.0)
        response.raise_for_status()
        return [item["embedding"] for item in sorted(response.json()["data"], key=lambda item: item["index"])]

    def valid(values: Any) -> list[float] | None:
        try:
            return _vector(values)
        except (TypeError, ValueError):
            return None

    # One text per request: batching several sequences into one llama.cpp request
    # returned all-NaN vectors for ~25% of items (16/request: 12/48 NaN;
    # 1/request: 0/48). Requests are cheap, so never batch.
    vectors = [valid(call([text])[0]) for text in texts]
    for index, vector in enumerate(vectors):
        # Exponential backoff (~40 s total) outlasts another client's burst on the server.
        for attempt in range(8):
            if vector is not None:
                break
            time.sleep(min(10.0, 0.3 * 2 ** attempt))
            call(["reset"])  # a throwaway request shifts the server's slot state
            vector = valid(call([texts[index]])[0])
        if vector is None:
            raise ValueError(f"Text embedding service returned an invalid vector for item {index}")
        vectors[index] = vector
    return vectors  # type: ignore[return-value]


def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder relevance scores (raw logits), in document order."""
    root = GATEWAY_BASE_URL[:-3] if GATEWAY_BASE_URL.endswith("/v1") else GATEWAY_BASE_URL
    response = httpx.post(f"{root}/v1/rerank", headers=_gateway_headers(),
                          json={"model": RERANK_MODEL, "query": query, "documents": documents}, timeout=120.0)
    response.raise_for_status()
    scores = [0.0] * len(documents)
    for item in response.json()["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores
