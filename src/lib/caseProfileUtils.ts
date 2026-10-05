import type { CaseProfile } from "./caseProfileTypes";
import { emptyProfile } from "./caseProfileTypes";
import { API_BASE } from "./api";
import type { AiTraceStep, RoutingScore } from "./twinApi";
// ─── Confidence scoring ────────────────────────────────────────────────────

export const PROFILE_READY_THRESHOLD = 60; // Confidence % required before matching.

interface ConfidenceField {
    label: string;
    filled: (p: CaseProfile) => boolean;
}

const CONFIDENCE_FIELDS: ConfidenceField[] = [
    { label: "Patient age", filled: (p) => p.patient.age_years != null },
    { label: "Patient sex", filled: (p) => Boolean(p.patient.sex) },
    { label: "Comorbidities", filled: (p) => p.patient.comorbidities.length > 0 },
    { label: "Chief complaint", filled: (p) => Boolean(p.presentation.chief_complaint) },
    { label: "HPI", filled: (p) => Boolean(p.presentation.hpi) },
    { label: "PMH", filled: (p) => Boolean(p.presentation.pmh) },
    { label: "Imaging modality", filled: (p) => Boolean(p.study.modality) },
    { label: "Body region", filled: (p) => Boolean(p.study.body_region) },
    { label: "Image available", filled: (p) => Boolean(p.study.image_url) },
    { label: "Imaging findings", filled: (p) => (p.findings.imaging_findings ?? []).length > 0 },
    { label: "Primary diagnosis", filled: (p) => Boolean(p.assessment.diagnosis_primary) },
    { label: "Urgency", filled: (p) => Boolean(p.assessment.urgency) },
    { label: "Summary one-liner", filled: (p) => Boolean(p.summary.one_liner) },
    { label: "Key points", filled: (p) => p.summary.key_points.length > 0 },
];

export function computeProfileConfidence(profile: CaseProfile): {
    score: number;
    filled: number;
    total: number;
    missing: string[];
} {
    const total = CONFIDENCE_FIELDS.length;
    const filledFields = CONFIDENCE_FIELDS.filter((f) => f.filled(profile));
    const missingFields = CONFIDENCE_FIELDS.filter((f) => !f.filled(profile));

    return {
        score: Math.round((filledFields.length / total) * 100),
        filled: filledFields.length,
        total,
        missing: missingFields.map((f) => f.label),
    };
}

// ─── Extraction ────────────────────────────────────────────────────────────

export interface ExtractionResult {
    profile: CaseProfile;
    /** "gemma-4" normally; "regex-fallback" when the backend could not reach Gemma 4. */
    method: string;
    routing: { collection: string | null; scores: RoutingScore[] } | null;
    trace: AiTraceStep[];
}

/**
 * Backend pipeline: MedSigLIP zero-shot image type -> MedGemma image read ->
 * Gemma 4 structured case report. Errors are thrown, never replaced by a
 * silent client-side guess, so the UI always shows what really produced the profile.
 */
export async function extractCaseProfile(
    images: File[],
    notes: string,
    notesFile: File | null
): Promise<ExtractionResult> {
    const form = new FormData();
    images.forEach((img) => form.append("images", img));
    if (notesFile) form.append("notes_file", notesFile);
    form.append("notes", notes);

    const res = await fetch(`${API_BASE}/extract`, { method: "POST", body: form });
    if (!res.ok) {
        const payload = await res.json().catch(() => null) as { detail?: string } | null;
        throw new Error(payload?.detail || `Extraction failed (${res.status})`);
    }
    return await res.json() as ExtractionResult;
}

// ─── Merge ─────────────────────────────────────────────────────────────────

const isEmptyValue = (value: unknown) =>
    value === null || value === undefined || value === "" || (Array.isArray(value) && value.length === 0);

/** Recursively merge: a filled value in `next` wins, an empty one keeps `base`. */
function deepMerge<T>(base: T, next: T): T {
    if (isPlainObject(base) && isPlainObject(next)) {
        const result: Record<string, unknown> = { ...base };
        for (const [key, value] of Object.entries(next)) {
            result[key] = key in result ? deepMerge(result[key], value) : value;
        }
        return result as T;
    }
    return isEmptyValue(next) ? base : next;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Merge two profiles field by field, keeping already-captured values that the new extraction left empty. */
export function mergeProfiles(base: CaseProfile, next: CaseProfile): CaseProfile {
    return deepMerge(deepMerge(emptyProfile(), base), next);
}
