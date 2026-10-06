import type { CaseProfile } from "./caseProfileTypes";
import { emptyProfile } from "./caseProfileTypes";
import {
    extractCaseProfile,
    computeProfileConfidence,
    mergeProfiles,
    PROFILE_READY_THRESHOLD,
} from "./caseProfileUtils";
import { generateAgenticFollowup, summarizePatch } from "./agenticCopilot";
import type { AiTraceStep } from "./twinApi";

// ─── Types ─────────────────────────────────────────────────────────────────

export type OrchestratorPhase =
    | "greeting"
    | "extracting"
    | "patching"
    | "questioning"
    | "ready"
    | "expanded";

export type MessageType =
    | "text"
    | "file_attach"
    | "thinking"
    | "confidence_update"
    | "field_patch"
    | "schema_expansion"
    | "cta";

export interface OrchestratorMessage {
    id: string;
    role: "assistant" | "user";
    type: MessageType;
    content: string;
    /** Files attached by the user */
    files?: Array<{ name: string; type: string; preview?: string }>;
    /** Which fields were patched in this turn */
    patchedFields?: string[];
    /** Confidence score to display inline */
    confidence?: number;
}

export interface OrchestratorState {
    profile: CaseProfile;
    phase: OrchestratorPhase;
    messages: OrchestratorMessage[];
    /** The field label we are currently questioning about */
    currentQuestion: string | null;
    readyToProceed: boolean;
    /** Every clinician note so far; each turn re-extracts from the full record. */
    notesSoFar: string;
    /**
     * Checklist fields the clinician already answered with nothing to record
     * ("no comorbidities", "unknown"). They stay empty in the profile, so without
     * this list the same follow-up question would repeat forever.
     */
    answeredFields: string[];
}

// ─── Constants ─────────────────────────────────────────────────────────────

// ─── Helpers ───────────────────────────────────────────────────────────────

let _msgId = 0;
function msgId(): string {
    return `msg-${Date.now()}-${++_msgId}`;
}

function assistantMsg(content: string, type: MessageType = "text", extra?: Partial<OrchestratorMessage>): OrchestratorMessage {
    return { id: msgId(), role: "assistant", type, content, ...extra };
}

function thinkingMsg(content: string): OrchestratorMessage {
    return { id: msgId(), role: "assistant", type: "thinking", content };
}

// ─── Initial State ─────────────────────────────────────────────────────────

export function createInitialState(): OrchestratorState {
    return {
        profile: emptyProfile(),
        phase: "greeting",
        messages: [
            assistantMsg(
                "Clinical Copilot online. I'll guide you through building a complete case profile.\n\nDrop JPEG, PNG, or WebP imaging studies or documents (PDF, DOCX, TXT) directly into the chat, or paste a clinical note. I'll extract, structure, and ask follow-up questions until the profile is ready for routing.",
            ),
        ],
        currentQuestion: null,
        readyToProceed: false,
        notesSoFar: "",
        answeredFields: [],
    };
}

// ─── Core Orchestration ────────────────────────────────────────────────────

export interface ProcessTurnInput {
    userText: string;
    files: File[];
    currentState: OrchestratorState;
}

export interface ProcessTurnOutput {
    newState: OrchestratorState;
    /** Model steps that ran this turn, for the AI pipeline panel. */
    trace: AiTraceStep[];
}

export async function processIntakeTurn(input: ProcessTurnInput): Promise<ProcessTurnOutput> {
    const { userText, files, currentState } = input;
    const { profile: prevProfile } = currentState;

    const outgoingMessages: OrchestratorMessage[] = [];

    // 1. Build the user message
    const userMessage: OrchestratorMessage = {
        id: msgId(),
        role: "user",
        type: files.length > 0 && !userText.trim() ? "file_attach" : "text",
        content: userText.trim(),
        files: files.map(f => ({
            name: f.name,
            type: f.type,
            preview: f.type.startsWith("image/") ? URL.createObjectURL(f) : undefined,
        })),
    };
    outgoingMessages.push(userMessage);

    // Capture local image URL for the first uploaded image
    const firstImageFile = files.find(f => f.type === "image/jpeg" || f.type === "image/png" || f.type === "image/webp");
    const localImageUrl = firstImageFile ? URL.createObjectURL(firstImageFile) : null;

    // ── Unified path: always run extraction for any non-empty text or files ──
    // Short text (e.g. "He drinks heavy alcohol") is just as important as long text.
    // We run the same extraction pipeline for everything — the client-side fallback
    // handles it instantly when the backend is offline.
    const hasInput = files.length > 0 || userText.trim().length > 0;

    if (!hasInput) {
        // Nothing to process — return state unchanged
        return { newState: currentState, trace: [] };
    }

    const images = files.filter(f => f.type === "image/jpeg" || f.type === "image/png" || f.type === "image/webp");
    const docs = files.filter(f => !images.includes(f));
    const notesFile = docs[0] ?? null;

    // Gemma 4 sees the whole record each turn, so a short answer ("he smokes")
    // is interpreted in context instead of in isolation.
    const notesSoFar = [currentState.notesSoFar, userText.trim()].filter(Boolean).join("\n");
    let newProfile: CaseProfile;
    let trace: AiTraceStep[] = [];
    let method = "";
    try {
        const result = await extractCaseProfile(images, notesSoFar, notesFile);
        newProfile = result.profile;
        trace = result.trace;
        method = result.method;
    } catch (error) {
        // Keep the previous profile and say exactly what failed.
        const reason = error instanceof Error ? error.message : "The extraction service is unavailable.";
        return {
            newState: {
                ...currentState,
                notesSoFar,
                messages: [...currentState.messages, userMessage,
                    assistantMsg(`I couldn't build the case report: ${reason}\n\nYour notes are kept; send another message to retry.`)],
            },
            trace: [],
        };
    }

    // Merge with existing profile (keep already-captured fields)
    let mergedProfile = mergeProfiles(prevProfile, newProfile);

    // Inject local image URL if an image was uploaded
    if (localImageUrl && !mergedProfile.study.image_url) {
        mergedProfile = { ...mergedProfile, study: { ...mergedProfile.study, image_url: localImageUrl } };
    }

    const conf = computeProfileConfidence(mergedProfile);
    const patchedFields = diffProfileFields(prevProfile, mergedProfile);
    const expandedFields = diffExtraFields(prevProfile.extra_fields ?? {}, mergedProfile.extra_fields ?? {});

    // Build assistant response
    // The clinician replied to the last question but the field is still empty:
    // that is an answer ("none" / "not known"), so do not ask it again.
    const answeredFields = [...(currentState.answeredFields ?? [])];
    const asked = currentState.currentQuestion;
    if (asked && userText.trim() && !answeredFields.includes(asked) && !hasValue(mergedProfile, asked)) {
        answeredFields.push(asked);
    }
    const followup = generateAgenticFollowup(mergedProfile, conf.score, answeredFields);
    const patchSummary = patchedFields.length > 0
        ? summarizePatch(patchedFields)
        : null;
    const expandSummary = expandedFields.length > 0
        ? `✓ Extended — captured: ${expandedFields.join(", ")}.`
        : null;

    const methodNote = method === "regex-fallback"
        ? "⚠ Gemma 4 was unreachable, so a basic keyword extractor filled this profile. Check it carefully."
        : null;
    const assistantContent = [
        methodNote,
        patchSummary ?? expandSummary,
        followup.message,
    ].filter(Boolean).join("\n\n");

    const confMsg: OrchestratorMessage = {
        id: msgId(),
        role: "assistant",
        type: "confidence_update",
        content: `Profile completeness: ${conf.score}%`,
        confidence: conf.score,
    };

    const assistantReply = assistantMsg(assistantContent);

    const allMessages: OrchestratorMessage[] = [
        userMessage,
        patchedFields.length > 0
            ? { id: msgId(), role: "assistant" as const, type: "field_patch" as MessageType, content: "", patchedFields }
            : null,
        expandedFields.length > 0
            ? { id: msgId(), role: "assistant" as const, type: "schema_expansion" as MessageType, content: "", patchedFields: expandedFields }
            : null,
        confMsg,
        assistantReply,
    ].filter(Boolean) as OrchestratorMessage[];

    if (conf.score >= PROFILE_READY_THRESHOLD) {
        allMessages.push(assistantMsg(
            "The profile is comprehensive. You can proceed to find case matches.",
            "cta",
            { confidence: conf.score }
        ));
    }

    const hasExtraFields = Object.keys(mergedProfile.extra_fields ?? {}).length > 0;
    const nextPhase = conf.score >= PROFILE_READY_THRESHOLD
        ? (hasExtraFields ? "expanded" : "ready")
        : "questioning";

    return {
        newState: {
            profile: mergedProfile,
            phase: nextPhase,
            messages: [...currentState.messages, ...allMessages],
            currentQuestion: followup.priority_fields[0] ?? null,
            answeredFields,
            readyToProceed: conf.score >= PROFILE_READY_THRESHOLD,
            notesSoFar,
        },
        trace,
    };
}


/** True when the profile has a non-empty value at a dotted path such as "patient.comorbidities". */
function hasValue(profile: CaseProfile, path: string): boolean {
    const value = path.split(".").reduce<unknown>(
        (node, key) => (node && typeof node === "object" ? (node as Record<string, unknown>)[key] : undefined), profile);
    return !(value === null || value === undefined || value === "" || (Array.isArray(value) && value.length === 0));
}

// ─── Profile Diff ──────────────────────────────────────────────────────────

/** Returns human-readable labels for fields that changed between two profiles */
function diffProfileFields(prev: CaseProfile, next: CaseProfile): string[] {
    const changed: string[] = [];

    const check = (label: string, a: unknown, b: unknown) => {
        const isEmpty = (v: unknown) =>
            v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);
        if (isEmpty(a) && !isEmpty(b)) changed.push(label);
        else if (!isEmpty(a) && !isEmpty(b) && JSON.stringify(a) !== JSON.stringify(b)) {
            changed.push(label);
        }
    };

    check("Age", prev.patient.age_years, next.patient.age_years);
    check("Sex", prev.patient.sex, next.patient.sex);
    check("Comorbidities", prev.patient.comorbidities, next.patient.comorbidities);
    check("Allergies", prev.patient.allergies, next.patient.allergies);
    check("Medications", prev.patient.medications, next.patient.medications);
    check("Chief complaint", prev.presentation.chief_complaint, next.presentation.chief_complaint);
    check("HPI", prev.presentation.hpi, next.presentation.hpi);
    check("PMH", prev.presentation.pmh, next.presentation.pmh);
    check("Symptom duration", prev.presentation.symptom_duration, next.presentation.symptom_duration);
    check("Imaging modality", prev.study.modality, next.study.modality);
    check("Body region", prev.study.body_region, next.study.body_region);
    check("View position", prev.study.view_position, next.study.view_position);
    check("Primary diagnosis", prev.assessment.diagnosis_primary, next.assessment.diagnosis_primary);
    check("Urgency", prev.assessment.urgency, next.assessment.urgency);
    check("Differential", prev.assessment.differential, next.assessment.differential);
    check("Summary", prev.summary.one_liner, next.summary.one_liner);
    check("Imaging findings", prev.findings.imaging_findings, next.findings.imaging_findings);
    check("Lab findings", prev.findings.lab_findings, next.findings.lab_findings);
    check("Red flags", prev.summary.red_flags, next.summary.red_flags);

    return changed;
}

/** Returns new or changed extra_field keys */
function diffExtraFields(
    prev: Record<string, string | string[]>,
    next: Record<string, string | string[]>
): string[] {
    const changed: string[] = [];
    for (const key of Object.keys(next)) {
        if (JSON.stringify(next[key]) !== JSON.stringify(prev[key])) {
            // Format key nicely: "smoking_status" → "Smoking Status"
            changed.push(key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()));
        }
    }
    return changed;
}

// ─── Utility ───────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}
