import type { CaseProfile } from "./caseProfileTypes";

export interface AgenticFollowup {
    thought: string;
    message: string;
    priority_fields: string[];
}

// ─── Priority checklist for follow-up questioning ──────────────────────────

interface PriorityField {
    key: string;
    label: string;
    category: string;
    question: string;
}

const PRIORITY_CHECKLIST: PriorityField[] = [
    {
        key: "patient.age_years",
        label: "Patient age",
        category: "Patient",
        question: "How old is the patient?",
    },
    {
        key: "patient.sex",
        label: "Patient sex",
        category: "Patient",
        question: "What is the patient's biological sex?",
    },
    {
        key: "presentation.chief_complaint",
        label: "Chief complaint",
        category: "Presentation",
        question: "What is the patient's chief complaint or primary reason for this referral?",
    },
    {
        key: "presentation.hpi",
        label: "Clinical history (HPI)",
        category: "Presentation",
        question: "Can you share the history of present illness — symptom onset, progression, and key timeline?",
    },
    {
        key: "patient.comorbidities",
        label: "Comorbidities",
        category: "Patient",
        question: "Does the patient have any significant comorbidities — hypertension, diabetes, COPD, cardiac history?",
    },
    {
        key: "presentation.pmh",
        label: "Past medical history (PMH)",
        category: "Presentation",
        question: "What is the patient's past medical history — prior diagnoses, surgeries, or hospitalizations?",
    },
    {
        key: "assessment.diagnosis_primary",
        label: "Primary diagnosis",
        category: "Assessment",
        question: "What is the suspected or working primary diagnosis at this point?",
    },
    {
        key: "assessment.urgency",
        label: "Clinical urgency",
        category: "Assessment",
        question: "How would you classify the urgency — emergent, semi-urgent, or routine?",
    },
    {
        key: "study.modality",
        label: "Imaging modality",
        category: "Imaging",
        question: "What imaging modality was used — CT, MRI, CXR, or another?",
    },
];

function getNestedValue(obj: Record<string, unknown>, path: string): unknown {
    return path.split(".").reduce<unknown>((prev, curr) => {
        if (prev && typeof prev === "object") return (prev as Record<string, unknown>)[curr];
        return undefined;
    }, obj);
}

function isEmpty(val: unknown): boolean {
    return val === null || val === undefined || val === "" || (Array.isArray(val) && val.length === 0);
}

// ─── Agentic follow-up generator ───────────────────────────────────────────

export function generateAgenticFollowup(
    profile: CaseProfile, confidence: number, answeredFields: string[] = [],
): AgenticFollowup {
    // Skip fields the clinician already answered with "none"/"unknown".
    const missingPriority = PRIORITY_CHECKLIST.filter(item => {
        if (answeredFields.includes(item.key)) return false;
        const val = getNestedValue(profile as unknown as Record<string, unknown>, item.key);
        // The question asks for a "suspected or working" diagnosis, so a suspected one answers it.
        if (item.key === "assessment.diagnosis_primary" && !isEmpty(profile.assessment.suspected_primary)) return false;
        return isEmpty(val);
    });

    const extraCount = Object.keys(profile.extra_fields ?? {}).length;

    let thought = "";
    let message = "";

    if (confidence === 0) {
        thought = "Profile empty. Awaiting initial data.";
        message =
            "I've set up your case profile. Share imaging impressions, paste a referral note, or upload a file and I'll begin populating the structured fields.";
    } else if (confidence < 35) {
        const first = missingPriority[0];
        thought = `Very low confidence (${confidence}%). Nudging for ${first?.label ?? "more data"}.`;
        message = `I've started building the profile. To move forward: ${first?.question ?? "Can you share more clinical detail?"}`;
    } else if (confidence < 60) {
        const first = missingPriority[0];
        thought = `Moderate progress (${confidence}%). Focus on ${first?.category ?? "details"}.`;
        message = first
            ? `Profile is taking shape. Next: ${first.question}`
            : "Looking good — a bit more detail would help strengthen the assessment.";
    } else if (confidence < 100) {
        const first = missingPriority[0];
        thought = `Good confidence (${confidence}%). Polishing.`;
        message = first
            ? `Profile is strong. One more detail: ${first.question}`
            : "The profile is very comprehensive. Ready to proceed when you are.";
    } else {
        thought = "Base profile 100% complete.";
        if (extraCount > 0) {
            message =
                `Base profile is complete ✓. ${extraCount} extended field${extraCount > 1 ? "s" : ""} captured beyond the schema. You can continue sharing supplemental information — I'll keep enriching the record.`;
        } else {
            message =
                "This case profile is detailed and complete. I've captured all critical fields — you can proceed to finding case matches, or continue to add supplemental information.";
        }
    }

    return {
        thought,
        message,
        priority_fields: missingPriority.map(m => m.key),
    };
}

// ─── Targeted question per field ───────────────────────────────────────────

export function getTargetedQuestion(fieldKey: string): string {
    const item = PRIORITY_CHECKLIST.find(f => f.key === fieldKey);
    return item?.question ?? "Can you provide more detail on this aspect of the case?";
}

// ─── Single-field patch from a conversational answer ──────────────────────

export function patchProfileFromAnswer(
    fieldKey: string,
    answer: string,
    profile: CaseProfile
): CaseProfile {
    const patched: CaseProfile = JSON.parse(JSON.stringify(profile)) as CaseProfile;
    const text = answer.trim();

    switch (fieldKey) {
        case "patient.age_years": {
            const m = text.match(/(\d{1,3})/);
            if (m) patched.patient.age_years = parseInt(m[1], 10);
            break;
        }
        case "patient.sex": {
            if (/female|woman|F\b/i.test(text)) patched.patient.sex = "female";
            else if (/male|man|M\b/i.test(text)) patched.patient.sex = "male";
            break;
        }
        case "presentation.chief_complaint": {
            patched.presentation.chief_complaint = text;
            break;
        }
        case "presentation.hpi": {
            patched.presentation.hpi = text;
            break;
        }
        case "presentation.pmh": {
            patched.presentation.pmh = text;
            // Also extract comorbidities
            const comorbMap: [RegExp, string][] = [
                [/hypertension|HTN/i, "hypertension"],
                [/type 2 diabet|T2DM/i, "type 2 diabetes"],
                [/type 1 diabet|T1DM/i, "type 1 diabetes"],
                [/atrial fibrillation|AF\b|AFib/i, "atrial fibrillation"],
                [/heart failure|CHF/i, "heart failure"],
                [/COPD|chronic obstructive/i, "COPD"],
                [/asthma/i, "asthma"],
                [/CKD|chronic kidney/i, "chronic kidney disease"],
                [/CAD|coronary artery/i, "coronary artery disease"],
                [/cancer|malignancy/i, "malignancy"],
            ];
            const found = comorbMap.filter(([re]) => re.test(text)).map(([, l]) => l);
            if (found.length > 0) {
                patched.patient.comorbidities = [
                    ...new Set([...patched.patient.comorbidities, ...found]),
                ];
            }
            break;
        }
        case "patient.comorbidities": {
            const comorbMap: [RegExp, string][] = [
                [/hypertension|HTN/i, "hypertension"],
                [/type 2 diabet|T2DM/i, "type 2 diabetes"],
                [/type 1 diabet|T1DM/i, "type 1 diabetes"],
                [/atrial fibrillation|AF\b|AFib/i, "atrial fibrillation"],
                [/heart failure|CHF/i, "heart failure"],
                [/COPD|chronic obstructive/i, "COPD"],
                [/asthma/i, "asthma"],
                [/CKD|chronic kidney/i, "chronic kidney disease"],
                [/CAD|coronary artery/i, "coronary artery disease"],
                [/obesity/i, "obesity"],
                [/cancer|malignancy/i, "malignancy"],
                [/diabetes/i, "diabetes"],
            ];
            const found = comorbMap.filter(([re]) => re.test(text)).map(([, l]) => l);
            if (found.length > 0) {
                patched.patient.comorbidities = [
                    ...new Set([...patched.patient.comorbidities, ...found]),
                ];
            } else if (text.length > 3) {
                // Treat as free-text comorbidity
                patched.patient.comorbidities = [...patched.patient.comorbidities, text];
            }
            break;
        }
        case "assessment.diagnosis_primary": {
            patched.assessment.diagnosis_primary = text;
            if (!patched.assessment.suspected_primary.includes(text)) {
                patched.assessment.suspected_primary = [text, ...patched.assessment.suspected_primary];
            }
            break;
        }
        case "assessment.urgency": {
            if (/emergent|emergency|stat|critical|immediate/i.test(text)) {
                patched.assessment.urgency = "emergent";
            } else if (/semi[- ]urgent|soon|within|24h/i.test(text)) {
                patched.assessment.urgency = "semi-urgent";
            } else if (/routine|elective|scheduled|non[- ]urgent/i.test(text)) {
                patched.assessment.urgency = "routine";
            } else {
                patched.assessment.urgency = text;
            }
            break;
        }
        case "study.modality": {
            if (/CT|computed tomography/i.test(text)) patched.study.modality = "CT";
            else if (/MRI/i.test(text)) patched.study.modality = "MRI";
            else if (/x[- ]?ray|CXR|chest x/i.test(text)) patched.study.modality = "CXR";
            else if (/PET/i.test(text)) patched.study.modality = "PET";
            else if (/ultrasound|US\b/i.test(text)) patched.study.modality = "Ultrasound";
            else patched.study.modality = text;
            break;
        }
        default: {
            // Free-text answer: try to store as an extra_field using the fieldKey as the key
            // (e.g., "extra_fields.smoking_status" → patched.extra_fields.smoking_status)
            if (fieldKey.startsWith("extra_fields.")) {
                const subKey = fieldKey.replace("extra_fields.", "");
                patched.extra_fields = { ...(patched.extra_fields ?? {}), [subKey]: text };
            }
            break;
        }
    }

    // Regenerate one-liner if we have enough
    if (patched.patient.age_years && patched.patient.sex && patched.assessment.diagnosis_primary) {
        const meds = patched.patient.comorbidities.slice(0, 2).join(", ");
        patched.summary.one_liner = `${patched.patient.age_years}-year-old ${patched.patient.sex} with ${meds || "relevant comorbidities"} presenting with ${patched.presentation.chief_complaint ?? patched.assessment.diagnosis_primary}.`;
    }

    return patched;
}

// ─── Patch summary ─────────────────────────────────────────────────────────

export function summarizePatch(fields: string[]): string {
    if (fields.length === 0) return "";
    if (fields.length === 1) return `✓ Profile updated — captured ${fields[0]}.`;
    if (fields.length <= 3) return `✓ Profile updated — captured: ${fields.join(", ")}.`;
    return `✓ Profile updated — captured ${fields.length} fields: ${fields.slice(0, 3).join(", ")}, and ${fields.length - 3} more.`;
}
