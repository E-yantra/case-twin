// Full CaseTwin case profile schema
// Mirrors backend/manifest.py and the Gemma 4 extraction schema in backend/extraction.py

export interface CaseProfile {
    profile_id: string;
    case_id: string;
    image_id: string;

    patient: PatientInfo;
    presentation: PresentationInfo;
    study: StudyInfo;
    assessment: AssessmentInfo;
    findings: FindingsInfo;
    management: ManagementInfo;
    summary: SummaryInfo;
    outcome: OutcomeInfo;
    provenance: ProvenanceInfo;
    tags: TagsInfo;

    /**
     * Dynamic key-value pairs for any information that doesn't fit the base schema.
     * Populated at any point during the intake chat — before or after 100% base completeness.
     * Never sent to the matching endpoint; used only for display and record enrichment.
     */
    extra_fields: Record<string, string | string[]>;
}

export interface PatientInfo {
    age_years: number | null;
    sex: string | null;
    immunocompromised: string | null;
    weight_kg: number | null;
    comorbidities: string[];
    medications: string[];
    allergies: string | null;
}

export interface PresentationInfo {
    chief_complaint: string | null;
    symptom_duration: string | null;
    hpi: string | null;
    pmh: string | null;
}

export interface StudyInfo {
    /** Twin-library collection (cxr, chest_ct, derm, fundus, histopath), set by MedSigLIP routing. */
    collection?: string | null;
    modality: string | null;
    body_region: string | null;
    view_position: string | null;
    radiology_region: string | null;
    caption: string | null;
    image_type: string | null;
    image_subtype: string | null;
    image_url: string | null;
    storage_path: string | null;
}

export interface AssessmentInfo {
    diagnosis_primary: string | null;
    suspected_primary: string[];
    differential: string[];
    urgency: string | null;
    infectious_concern: string | null;
    icu_candidate: string | null;
    clinical_synthesis?: string | null;
}

export interface LungsFindings {
    consolidation_present: string | null;
    consolidation_locations: string[];
    consolidation_extent: string | null;
    atelectasis_present: string | null;
    atelectasis_locations: string[];
    edema_present: string | null;
    edema_pattern: string | null;
}

export interface PleuraFindings {
    effusion_present: string | null;
    effusion_side: string | null;
    effusion_size: string | null;
    pneumothorax_present: string | null;
    pneumothorax_side: string | null;
}

export interface CardiomediastinalFindings {
    cardiomegaly: string | null;
    mediastinal_widening: string | null;
}

export interface DevicesFindings {
    lines_tubes_present: string | null;
    device_list: string[];
}

export interface FindingsInfo {
    /** Short phrases of what imaging showed, any modality. */
    imaging_findings: string[];
    lab_findings: string[];
    lungs: LungsFindings;
    pleura: PleuraFindings;
    cardiomediastinal: CardiomediastinalFindings;
    devices: DevicesFindings;
}

export interface ManagementInfo {
    treatments: string[];
    procedures: string[];
}

export interface SummaryInfo {
    one_liner: string | null;
    key_points: string[];
    red_flags: string[];
    /** The final diagnosis or teaching point a published case reaches. */
    conclusion: string | null;
}

export interface OutcomeInfo {
    success: string | null;
    detail: string | null;
    follow_up: string | null;
}

export interface ProvenanceInfo {
    dataset_name: string | null;
    pmc_id: string | null;
    pmid: string | null;
    doi: string | null;
    article_title: string | null;
    journal: string | null;
    year: number | null;
    authors: string[];
    license: string | null;
    source_url: string | null;
}

export interface TagsInfo {
    ml_labels: string[];
    gt_labels: string[];
    keywords: string[];
    mesh_terms: string[];
}

export function emptyProfile(): CaseProfile {
    return {
        profile_id: "",
        case_id: "",
        image_id: "",
        patient: {
            age_years: null,
            sex: null,
            immunocompromised: null,
            weight_kg: null,
            comorbidities: [],
            medications: [],
            allergies: null,
        },
        presentation: {
            chief_complaint: null,
            symptom_duration: null,
            hpi: null,
            pmh: null,
        },
        study: {
            collection: null,
            modality: null,
            body_region: null,
            view_position: null,
            radiology_region: null,
            caption: null,
            image_type: null,
            image_subtype: null,
            image_url: null,
            storage_path: null,
        },
        assessment: {
            diagnosis_primary: null,
            suspected_primary: [],
            differential: [],
            urgency: null,
            infectious_concern: null,
            icu_candidate: null,
        },
        findings: {
            imaging_findings: [],
            lab_findings: [],
            lungs: {
                consolidation_present: null,
                consolidation_locations: [],
                consolidation_extent: null,
                atelectasis_present: null,
                atelectasis_locations: [],
                edema_present: null,
                edema_pattern: null,
            },
            pleura: {
                effusion_present: null,
                effusion_side: null,
                effusion_size: null,
                pneumothorax_present: null,
                pneumothorax_side: null,
            },
            cardiomediastinal: {
                cardiomegaly: null,
                mediastinal_widening: null,
            },
            devices: {
                lines_tubes_present: null,
                device_list: [],
            },
        },
        management: {
            treatments: [],
            procedures: [],
        },
        summary: {
            one_liner: null,
            key_points: [],
            red_flags: [],
            conclusion: null,
        },
        outcome: {
            success: null,
            detail: null,
            follow_up: null,
        },
        provenance: {
            dataset_name: null,
            pmc_id: null,
            pmid: null,
            doi: null,
            article_title: null,
            journal: null,
            year: null,
            authors: [],
            license: null,
            source_url: null,
        },
        tags: {
            ml_labels: [],
            gt_labels: [],
            keywords: [],
            mesh_terms: [],
        },
        extra_fields: {},
    };
}
