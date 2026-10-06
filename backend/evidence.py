"""Quote-backed clinical field updates for notes, articles, and figure captions."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from local_ai import GEMMA_MODEL, query_local_model
from manifest import empty_profile

EXTRACTION_VERSION = "source-evidence/v2"
STATUSES = {"present", "absent", "suspected", "historical", "resolved", "uncertain"}
FIELDS = {
    "patient.age_years": int, "patient.sex": str,
    "patient.comorbidities": str, "presentation.chief_complaint": str,
    "presentation.symptom_duration": str, "presentation.pmh": str,
    "assessment.diagnosis_primary": str, "assessment.urgency": str,
    "outcome.detail": str,
    "findings.lungs.consolidation_present": str,
    "findings.lungs.opacity_present": str,
    "findings.lungs.atelectasis_present": str,
    "findings.lungs.edema_present": str,
    "findings.pleura.effusion_present": str,
    "findings.pleura.pneumothorax_present": str,
    "findings.cardiomediastinal.cardiomegaly": str,
    "findings.cardiomediastinal.mediastinal_gas_present": str,
}
FINDINGS = {field for field in FIELDS if field.startswith("findings.")}
CLINICAL = FINDINGS | {"assessment.diagnosis_primary"}
FIELD_TERMS = {
    "findings.lungs.consolidation_present": r"consolidat",
    "findings.lungs.opacity_present": r"opacit|ground[ -]glass|infiltrat",
    "findings.lungs.atelectasis_present": r"atelectas|lung collapse|lobar collapse",
    "findings.lungs.edema_present": r"(?:pulmonary|lung) edema",
    "findings.pleura.effusion_present": r"pleural effusion|pleural fluid",
    "findings.pleura.pneumothorax_present": r"pneumothorax",
    "findings.cardiomediastinal.cardiomegaly": r"cardiomegal|enlarged heart",
    "findings.cardiomediastinal.mediastinal_gas_present": r"mediastinal gas|pneumomediastinum|continuous diaphragm sign|Naclerio.s V sign",
}
NEGATION = re.compile(r"\b(?:no|without|absent|absence of|negative for|free of|denies|ruled out|not present)\b", re.I)
RESOLVED = re.compile(r"\b(?:resol(?:ved|ving|ution)|cleared|disappeared|no longer)\b", re.I)
HISTORICAL = re.compile(r"\b(?:history of|prior|previous|formerly|in the past|status post)\b", re.I)
SUSPECTED = re.compile(r"\b(?:suspect(?:ed)?|possible|perhaps|may have|could be|rule out|r/o|concern for|query)\b", re.I)

# These patterns are used only to veto a model claim whose timing or negation
# conflicts with the text. A generic phrase such as "no tension" must not turn
# an explicitly stated pneumothorax into an absent finding.
NEGATED_PREFIX = re.compile(
    r"\b(?:no|without)(?:\s+(?:evidence|signs?|finding|any|a|an|the|visible|"
    r"radiographic|obvious|appreciable|definite|demonstrable|of))*\s*$|"
    r"\b(?:absence of|negative for|free of|denies|ruled out|not)\s*$", re.I,
)
NEGATED_SUFFIX = re.compile(
    r"^\s*(?:(?:is|are|was|were|remains|appears)\s+)?"
    r"(?:absent|not present|not seen|not visuali[sz]ed|not identified|"
    r"not detected|not demonstrated|not appreciated|ruled out)\b", re.I,
)
RESOLVED_PREFIX = re.compile(
    r"\b(?:complete\s+)?(?:resolution of|resolved|resolving|cleared|"
    r"disappeared|no longer)\s*$", re.I,
)
RESOLVED_SUFFIX = re.compile(
    r"^\s*(?:(?:is|has|was|has been)\s+)?"
    r"(?:resolved|resolving|cleared|disappeared|no longer present|gone)\b", re.I,
)
HISTORICAL_PREFIX = re.compile(
    r"\b(?:history of|remote history of|prior|previous|previously|formerly|"
    r"in the past|status post)\s*$", re.I,
)
HISTORICAL_SUFFIX = re.compile(r"^\s*(?:in the past|historically|previously)\b", re.I)
SUSPECTED_PREFIX = re.compile(
    r"\b(?:suspect(?:ed)?|possible|possibly|perhaps|concern for|rule out|r/o|"
    r"query|may have|could be|might be)\s*$", re.I,
)
SUSPECTED_SUFFIX = re.compile(r"^\s*(?:is|may be|could be|might be)?\s*(?:suspected|possible)\b", re.I)


def _json_object(reply: str) -> dict:
    if not isinstance(reply, str):
        raise ValueError("Model response is not text")
    value = reply.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.I | re.S)
    if fence:
        value = fence.group(1)
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("updates"), list):
        raise ValueError("Model response needs an updates array")
    return parsed


def _source_clause(source: str, quote: str) -> str:
    start = source.find(quote)
    if start < 0:
        return ""
    # Limit timing/negation checks to the clause that contains the exact quote.
    left = max(source.rfind(mark, 0, start) for mark in ".;:\n") + 1
    right_candidates = [source.find(mark, start + len(quote)) for mark in ".;:\n"]
    right = min((i for i in right_candidates if i >= 0), default=len(source))
    return source[left:right].strip()


def _field_term(field: str, value: Any) -> str | None:
    if field in FIELD_TERMS:
        return FIELD_TERMS[field]
    if field == "assessment.diagnosis_primary" and isinstance(value, str) and value.strip():
        return re.escape(value.strip())
    return None


def _field_has_qualifier(text: str, field: str, value: Any, status: str) -> bool:
    term = _field_term(field, value)
    if not term:
        return False
    for match in re.finditer(term, text, re.I):
        before, after = text[:match.start()], text[match.end():]
        if status == "absent" and (NEGATED_PREFIX.search(before) or NEGATED_SUFFIX.search(after)):
            return True
        if status == "resolved" and (RESOLVED_PREFIX.search(before) or RESOLVED_SUFFIX.search(after)):
            return True
        if status == "historical" and (HISTORICAL_PREFIX.search(before) or HISTORICAL_SUFFIX.search(after)):
            return True
        if status == "suspected" and (SUSPECTED_PREFIX.search(before) or SUSPECTED_SUFFIX.search(after)):
            return True
    return False


def _valid_status(status: str, clause: str, field: str, quote: str, value: Any) -> bool:
    if field not in CLINICAL:
        return True
    # Checks only veto contradictory assertions. They never create a finding.
    if status == "present":
        return not any(_field_has_qualifier(clause, field, value, qualifier)
                       for qualifier in ("absent", "resolved", "historical", "suspected"))
    if status == "absent":
        return _field_has_qualifier(clause, field, value, "absent") and _field_has_qualifier(quote, field, value, "absent")
    if status == "resolved":
        return _field_has_qualifier(clause, field, value, "resolved") and _field_has_qualifier(quote, field, value, "resolved")
    if status == "historical":
        return _field_has_qualifier(clause, field, value, "historical") and _field_has_qualifier(quote, field, value, "historical")
    if status == "suspected":
        return _field_has_qualifier(clause, field, value, "suspected") and _field_has_qualifier(quote, field, value, "suspected")
    return True


def validate_updates(reply: str, sources: dict[str, str], allowed: set[str] | None = None) -> tuple[list[dict], list[str]]:
    parsed = _json_object(reply)
    updates, warnings = [], []
    allowed = allowed or set(FIELDS)
    for index, item in enumerate(parsed["updates"]):
        if not isinstance(item, dict) or set(item) != {"field", "value", "status", "quote", "source"}:
            warnings.append(f"Update {index + 1} has invalid fields")
            continue
        field, value, status, quote, source_id = (item[key] for key in ("field", "value", "status", "quote", "source"))
        def warn(message: str) -> None:
            warnings.append(f"Update {index + 1} field={field!r} status={status!r}: {message}")
        if (not isinstance(field, str) or not isinstance(status, str) or field not in allowed
                or field not in FIELDS or status not in STATUSES):
            warn("has an unsupported field or status")
            continue
        if not isinstance(quote, str) or not quote.strip():
            warn("has an invalid quote")
            continue
        if not isinstance(source_id, str) or source_id not in sources:
            warn(f"source id {source_id!r} is not provided")
            continue
        if quote not in sources[source_id]:
            warn("quote does not occur in its source")
            continue
        expected = FIELDS[field]
        if expected is int:
            valid_value = type(value) is int and 0 < value <= 120
        else:
            valid_value = isinstance(value, str) and bool(value.strip()) and len(value) <= 160
        if not valid_value:
            warn("has an invalid value")
            continue
        if field in FINDINGS and str(value).lower() not in {"yes", "no"}:
            warn("must use yes or no")
            continue
        if field == "patient.sex":
            sex_aliases = {"female": "female", "woman": "female", "women": "female", "girl": "female",
                           "male": "male", "man": "male", "men": "male", "boy": "male"}
            canonical_sex = sex_aliases.get(value.strip().lower())
            if canonical_sex is None:
                warn("has an unsupported sex value")
                continue
            value = canonical_sex
        if field in FINDINGS and (status == "present") != (value.lower() == "yes") and status in {"present", "absent"}:
            warn("value contradicts status")
            continue
        clause = _source_clause(sources[source_id], quote)
        if not _valid_status(status, clause, field, quote, value):
            warn("status contradicts its source clause")
            continue
        if field in FINDINGS and not re.search(FIELD_TERMS[field], clause, re.I):
            warn("does not name its finding in the source clause")
            continue
        quote_lower = quote.lower()
        if field == "patient.age_years" and str(value) not in quote:
            warn("age is not stated in its quote")
            continue
        if field == "patient.sex":
            sex_term = r"\b(?:female|woman|women|girl)\b" if value == "female" else r"\b(?:male|man|men|boy)\b"
            if not re.search(sex_term, quote, re.I):
                warn("sex is not stated in its quote")
                continue
        if field == "patient.comorbidities" and value.lower() not in quote_lower:
            aliases = {"hypertension": r"\bhtn\b", "type 2 diabetes": r"\bt2dm\b|\bdm2\b",
                       "type 1 diabetes": r"\bt1dm\b|\bdm1\b", "atrial fibrillation": r"\bafib?\b",
                       "heart failure": r"\bchf\b", "chronic kidney disease": r"\bckd\b",
                       "coronary artery disease": r"\bcad\b"}
            if not re.search(aliases.get(value.lower(), r"(?!)"), quote, re.I):
                warn("comorbidity is not stated in its quote")
                continue
        if field in {"presentation.chief_complaint", "presentation.symptom_duration", "presentation.pmh", "assessment.urgency", "outcome.detail"} and value.lower() not in quote_lower:
            warn("value is not stated in its quote")
            continue
        if field == "assessment.diagnosis_primary" and str(value).lower() not in quote.lower():
            warn("diagnosis is not in its quote")
            continue
        updates.append({"field": field, "value": value, "status": status, "quote": quote, "source": source_id})
    return updates, warnings


def extract_claims(sources: dict[str, str], allowed: set[str] | None = None, max_tokens: int = 1800) -> tuple[list[dict], str, list[str]]:
    allowed = allowed or set(FIELDS)
    if not any(value.strip() for value in sources.values()):
        return [], "no_text", []
    prompt = (
        "Extract only explicit assertions from the supplied clinical text. Return one JSON object with "
        '"updates": [{"field": string, "value": string or integer, "status": "present"|"absent"|"suspected"|"historical"|"resolved"|"uncertain", "quote": exact source substring, "source": source id}]. '
        "Output compact JSON only: no analysis, explanation, or markdown. Each quote must be copied exactly, including negation and timing. Do not infer diagnoses from symptoms or imaging. "
        "Do not convert a suspected, historical, resolved, or absent disease to a current positive. "
        "Use yes/no for finding values. Do not equate opacity or ground glass with consolidation. Omit unsupported fields. Conflicting claims may each be returned. "
        f"Allowed fields: {json.dumps(sorted(allowed))}. Sources: {json.dumps(sources, ensure_ascii=False)}"
    )
    try:
        response = query_local_model(prompt, model=GEMMA_MODEL, max_tokens=max_tokens, temperature=0,
                                     response_format={"type": "json_object"})
        updates, warnings = validate_updates(response[0]["generated_text"], sources, allowed)
        return updates, "complete", warnings
    except Exception as exc:
        return [], "failed", [f"Structured extraction failed: {type(exc).__name__}: {exc}"]


def apply_updates(profile: dict, updates: list[dict], *, source_fields: set[str] | None = None) -> dict:
    result = copy.deepcopy(profile)
    grouped: dict[str, list[dict]] = {}
    for update in updates:
        if source_fields is None or update["field"] in source_fields:
            grouped.setdefault(update["field"], []).append(update)
    for field, claims in grouped.items():
        path = field.split(".")
        section = result
        for part in path[:-1]:
            section = section[part]
        statuses = {claim["status"] for claim in claims}
        values = {str(claim["value"]).lower() for claim in claims}
        if len(statuses) > 1 or len(values) > 1:
            section[path[-1]] = None if field != "patient.comorbidities" else []
        elif statuses == {"present"}:
            value = claims[-1]["value"]
            if field == "patient.comorbidities":
                section[path[-1]] = list(dict.fromkeys([*section[path[-1]], value]))
            elif field in FINDINGS:
                section[path[-1]] = "yes"
            else:
                section[path[-1]] = value
        elif statuses == {"absent"} and field in FINDINGS:
            section[path[-1]] = "no"
        else:
            section[path[-1]] = None if field != "patient.comorbidities" else []
    return result


def new_current_profile() -> dict:
    profile = empty_profile(article_id="unknown", image_id="unknown")
    profile["profile_id"] = ""
    profile["case_id"] = ""
    profile["image_id"] = ""
    profile["study"]["modality"] = None
    profile["study"]["body_region"] = None
    profile["study"]["image_type"] = None
    profile["study"]["image_subtype"] = None
    profile["provenance"]["dataset_name"] = None
    profile["provenance"]["pmc_id"] = None
    return profile


def supported_positive(updates: list[dict]) -> set[tuple[str, str]]:
    """Positive, non-conflicted clinical terms only; used by retrieval and chat."""
    grouped: dict[str, list[dict]] = {}
    for item in updates:
        if isinstance(item, dict) and item.get("field") in CLINICAL:
            grouped.setdefault(item["field"], []).append(item)
    return {(field, str(items[0]["value"]).lower()) for field, items in grouped.items()
            if len({i.get("status") for i in items}) == 1
            and items[0].get("status") == "present"
            and len({str(i.get("value")).lower() for i in items}) == 1}
