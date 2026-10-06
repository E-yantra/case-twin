"""Gemma 4 structured case-report extraction shared by dataset preparation and live intake.

One JSON Schema constrains the model's output (llama.cpp turns it into a
grammar), so every profile in the twin library and every live query profile
has the same fields. ``manifest.merge_profile`` then drops anything outside the
canonical shape.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from local_ai import GEMMA_MODEL, query_local_model

_STR = {"type": ["string", "null"]}
_LIST = {"type": "array", "items": {"type": "string"}}
_YESNO = {"type": ["string", "null"], "enum": ["yes", "no", None]}


def _obj(**properties: Any) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}


PROFILE_SCHEMA = _obj(
    patient=_obj(age_years={"type": ["number", "null"]}, sex={"type": ["string", "null"], "enum": ["male", "female", None]},
                 immunocompromised=_YESNO, comorbidities=_LIST, medications=_LIST, allergies=_STR),
    presentation=_obj(chief_complaint=_STR, symptom_duration=_STR, hpi=_STR, pmh=_STR),
    study=_obj(modality=_STR, body_region=_STR, view_position=_STR),
    assessment=_obj(diagnosis_primary=_STR, suspected_primary=_LIST, differential=_LIST,
                    urgency={"type": ["string", "null"], "enum": ["emergent", "urgent", "semi-urgent", "routine", None]},
                    infectious_concern=_YESNO, icu_candidate=_YESNO),
    findings=_obj(
        imaging_findings=_LIST, lab_findings=_LIST,
        lungs=_obj(consolidation_present=_YESNO, consolidation_locations=_LIST, atelectasis_present=_YESNO,
                   edema_present=_YESNO),
        pleura=_obj(effusion_present=_YESNO, effusion_side=_STR, pneumothorax_present=_YESNO, pneumothorax_side=_STR),
        cardiomediastinal=_obj(cardiomegaly=_YESNO, mediastinal_widening=_YESNO),
    ),
    management=_obj(treatments=_LIST, procedures=_LIST),
    summary=_obj(one_liner=_STR, key_points=_LIST, red_flags=_LIST, conclusion=_STR),
    outcome=_obj(success=_YESNO, detail=_STR, follow_up=_STR),
)

_RULES = """Rules:
- Use only facts stated in the source. Never infer, guess or add a diagnosis that is not written.
- diagnosis_primary: only a diagnosis stated as established. A diagnosis written as uncertain ("?psoriasis",
  "likely CAP", "suspected", "r/o", "vs") goes in suspected_primary; the alternatives it is weighed against go
  in differential. Strip the "?" from the item text.
- Use null (or []) when the source does not say. Absence of a mention is null, not "no".
- Lung/pleura/cardiomediastinal yes/no fields apply only to chest findings; otherwise null.
- imaging_findings: short phrases of what the imaging showed (any modality). lab_findings: abnormal labs/pathology.
- hpi: a clean 2-4 sentence history of presenting illness in plain clinical English.
- one_liner: one sentence "<age>-year-old <sex> with <key history> presenting with <problem>".
- conclusion: the final diagnosis or teaching point the case reaches, one sentence.
- outcome.detail: what happened to the patient (response to treatment, recovery, death, follow-up result).
- Keep the source's medical terms; expand an abbreviation only when the source defines it."""


def article_prompt(abstract: str, narrative: str) -> str:
    return f"""You are building a structured case report from a published clinical case.
{_RULES}

Abstract:
{abstract or "(none)"}

Case narrative:
{narrative or "(none)"}"""


_INTAKE_RULES = """- red_flags: list every documented danger sign, quoting the value: SpO2 < 92%, RR > 24, HR > 120,
  systolic BP < 90, altered consciousness, haemoptysis, unexplained weight loss, chest pain with
  instability, sepsis criteria, an acute haemoglobin drop or other sign of bleeding (especially after
  surgery or trauma). Only signs actually written in the notes or image reads.
- urgency: "emergent" if any red flag shows physiological instability (low SpO2, high RR, hypotension,
  altered consciousness); "urgent" for other red flags or an acute presentation; else null.
- Cross-check the notes against the AI image reads. If they disagree (e.g. side, location, presence of a
  finding), add a red flag of the form "Discrepancy: notes say <X>; AI image read says <Y>. Verify." and do
  not resolve it yourself: copy each source's finding into its own field as written."""


def intake_prompt(notes: str, image_reads: list[str]) -> str:
    reads = "\n".join(f"- Image {index}: {read}" for index, read in enumerate(image_reads, start=1)) or "(no images)"
    return f"""You are turning a clinician's unstructured notes into a clean structured case report.
{_RULES}
- The image reads below are AI (MedGemma) descriptions, not radiologist reports: copy their findings into
  imaging_findings but do not use them alone to set a diagnosis. Outcome fields stay null for a new case.
{_INTAKE_RULES}

Clinician notes:
{notes.strip() or "(none)"}

AI image reads:
{reads}"""


def parse_model_json(text: str) -> dict:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            parsed, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model did not return a JSON object")


def extract_profile(prompt: str, *, max_tokens: int = 2000, attempts: int = 2, model: str | None = None,
                    base_url: str | None = None, timeout: float = 300.0) -> tuple[dict, dict]:
    """Return (profile fields, trace step). Raises ValueError when every attempt fails.

    Defaults to Gemma 4 on the gateway; ``model``/``base_url`` let offline dataset
    preparation use another OpenAI-compatible server with the same schema.
    """
    model = model or GEMMA_MODEL
    started = time.perf_counter()
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            reply = query_local_model(prompt, model=model, max_tokens=max_tokens, temperature=0,
                                      json_schema=PROFILE_SCHEMA, thinking=False, timeout=timeout, base_url=base_url)
            profile = parse_model_json(reply[0]["generated_text"])
            return profile, {"model": model, "task": "Structure notes into a case report (JSON schema)",
                             "ms": round((time.perf_counter() - started) * 1000)}
        except (ValueError, KeyError, httpx.HTTPError) as exc:
            last_error = exc
    raise ValueError(f"Structured extraction failed: {last_error}")
