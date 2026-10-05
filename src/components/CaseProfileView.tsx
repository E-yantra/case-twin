import { cn } from "@/lib/utils";
import { SelectionExplainPopover } from "@/components/SelectionExplainPopover";
import type {
    AssessmentInfo,
    CardiomediastinalFindings,
    CaseProfile,
    DevicesFindings,
    FindingsInfo,
    LungsFindings,
    OutcomeInfo,
    PatientInfo,
    PleuraFindings,
    PresentationInfo,
    ProvenanceInfo,
    StudyInfo,
    SummaryInfo,
    TagsInfo,
} from "@/lib/caseProfileTypes";
import {
    Activity,
    AlertTriangle,
    BookOpen,
    CheckCircle2,
    ChevronRight,
    ClipboardList,
    ExternalLink,
    FileImage,
    FlaskConical,
    Heart,
    Layers,
    Loader2,
    Microscope,
    Stethoscope,
    Tag,
    User,
    Wind,
    XCircle,
    Zap,
} from "lucide-react";

import ReactMarkdown from "react-markdown";

// ─── Helper atoms ─────────────────────────────────────────────────────────────

function SectionCard({
    icon,
    title,
    accent = "slate",
    children,
    className,
}: {
    icon: React.ReactNode;
    title: string;
    accent?: string;
    children: React.ReactNode;
    className?: string;
}) {
    return (
        <section className={cn("mb-8 border-b border-zinc-100 pb-8 last:border-0 last:pb-0", className)}>
            <div className="flex items-center gap-2.5 mb-4">
                <span className="text-zinc-500 flex items-center justify-center h-6 w-6 rounded-md bg-zinc-100 border border-zinc-200/60">
                    {icon}
                </span>
                <h3 className="text-[16px] font-semibold text-zinc-900 tracking-tight">{title}</h3>
            </div>
            <div className="space-y-4 pl-[34px]">{children}</div>
        </section>
    );
}

function Field({
    label,
    value,
    wide,
}: {
    label: string;
    value: React.ReactNode;
    wide?: boolean;
}) {
    return (
        <div className={cn("space-y-1", wide && "col-span-2")}>
            <p className="text-[13px] font-medium text-zinc-500">{label}</p>
            <div className="text-[14px] leading-relaxed text-zinc-900">{value}</div>
        </div>
    );
}

function Chip({ label, variant = "neutral" }: { label: string; variant?: "neutral" | "primary" | "success" | "warning" | "danger" }) {
    const map = {
        neutral: "bg-zinc-100 text-zinc-700",
        primary: "bg-blue-50/80 text-blue-700",
        success: "bg-emerald-50/80 text-emerald-700",
        warning: "bg-amber-50/80 text-amber-700",
        danger: "bg-rose-50/80 text-rose-700",
    };
    return (
        <span className={cn("inline-flex items-center rounded-md px-2 py-0.5 text-[12px] font-medium", map[variant])}>
            {label}
        </span>
    );
}

function YesNoBadge({ value }: { value: string | null }) {
    if (!value) return <span className="text-zinc-400 text-[14px]">—</span>;
    const isYes = value.toLowerCase() === "yes";
    const isNo = value.toLowerCase() === "no";
    if (isYes) return <span className="inline-flex items-center gap-1.5 text-zinc-900 text-[14px] font-medium"><CheckCircle2 className="h-4 w-4 text-emerald-500" />{value}</span>;
    if (isNo) return <span className="inline-flex items-center gap-1.5 text-zinc-500 text-[14px] font-medium"><XCircle className="h-4 w-4 text-zinc-400" />{value}</span>;
    return <span className="text-zinc-900 text-[14px]">{value}</span>;
}

function isEmpty(val: any): boolean {
    if (val === null || val === undefined) return true;
    if (typeof val === "string" && val.trim() === "") return true;
    if (Array.isArray(val) && val.length === 0) return true;
    return false;
}

// ─── Sections ─────────────────────────────────────────────────────────────────

function PatientSection({ patient }: { patient: PatientInfo }) {
    const hasAny = !isEmpty(patient.age_years) || !isEmpty(patient.sex) || !isEmpty(patient.weight_kg) ||
        !isEmpty(patient.immunocompromised) || !isEmpty(patient.comorbidities) ||
        !isEmpty(patient.medications) || !isEmpty(patient.allergies);

    if (!hasAny) return null;

    return (
        <SectionCard icon={<User className="h-4 w-4" />} title="Patient" accent="blue">
            <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                {!isEmpty(patient.age_years) && <Field label="Age" value={`${patient.age_years} yrs`} />}
                {!isEmpty(patient.sex) && <Field label="Sex" value={patient.sex} />}
                {!isEmpty(patient.weight_kg) && <Field label="Weight" value={`${patient.weight_kg} kg`} />}
                {!isEmpty(patient.immunocompromised) && <Field label="Immunocompromised" value={<YesNoBadge value={patient.immunocompromised} />} />}

                {!isEmpty(patient.comorbidities) && (
                    <Field wide label="Comorbidities" value={
                        <div className="flex flex-wrap gap-1.5 pt-0.5">
                            {patient.comorbidities.map((c) => <Chip key={c} label={c} variant="neutral" />)}
                        </div>
                    } />
                )}

                {!isEmpty(patient.medications) && (
                    <Field wide label="Medications" value={
                        <div className="flex flex-wrap gap-1.5 pt-0.5">
                            {patient.medications.map((m) => <Chip key={m} label={m} variant="primary" />)}
                        </div>
                    } />
                )}

                {!isEmpty(patient.allergies) && <Field label="Allergies" value={patient.allergies} />}
            </div>
        </SectionCard>
    );
}

function PresentationSection({ pres }: { pres: PresentationInfo }) {
    const hasAny = !isEmpty(pres.chief_complaint) || !isEmpty(pres.symptom_duration) ||
        !isEmpty(pres.hpi) || !isEmpty(pres.pmh);

    if (!hasAny) return null;

    return (
        <SectionCard icon={<Stethoscope className="h-4 w-4" />} title="Presentation" accent="violet">
            <div className="space-y-3">
                {!isEmpty(pres.chief_complaint) && <Field label="Chief Complaint" value={pres.chief_complaint} />}
                {!isEmpty(pres.symptom_duration) && <Field label="Symptom Duration" value={pres.symptom_duration} />}
                {!isEmpty(pres.hpi) && (
                    <Field label="HPI" value={
                        <div className="text-[14px] leading-relaxed text-zinc-800 border-l-[3px] border-zinc-200 pl-3 py-1 mt-1 break-words">{pres.hpi}</div>
                    } />
                )}
                {!isEmpty(pres.pmh) && <Field label="PMH" value={pres.pmh} />}
            </div>
        </SectionCard>
    );
}

function StudySection({ study, enhancedImaging }: { study: StudyInfo, enhancedImaging?: string }) {
    const hasAny = !isEmpty(study.image_url) || !isEmpty(study.modality) ||
        !isEmpty(study.view_position) || !isEmpty(study.body_region) ||
        !isEmpty(study.image_type) || !isEmpty(study.image_subtype) ||
        !isEmpty(enhancedImaging);

    if (!hasAny) return null;

    return (
        <SectionCard icon={<FileImage className="h-4 w-4" />} title="Imaging Study" accent="cyan">
            <div className="space-y-3">
                {!isEmpty(study.image_url) && (
                    <div className="overflow-hidden rounded-xl border border-zinc-200/60 bg-zinc-50/50">
                        <img
                            src={study.image_url as string}
                            alt={study.caption ?? "Medical image"}
                            className="w-full max-h-64 object-contain"
                        />
                        {!isEmpty(study.caption) && (
                            <p className="px-3 py-2 text-[12px] text-zinc-500 bg-white border-t border-zinc-100">{study.caption}</p>
                        )}
                    </div>
                )}
                <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                    {!isEmpty(study.modality) && <Field label="Modality" value={<Chip label={study.modality as string} variant="primary" />} />}
                    {!isEmpty(study.view_position) && <Field label="View Position" value={study.view_position as string} />}
                    {!isEmpty(study.body_region) && <Field label="Body Region" value={study.body_region as string} />}
                    {!isEmpty(study.image_subtype ?? study.image_type) && <Field label="Image Type" value={(study.image_subtype ?? study.image_type) as string} />}
                </div>
                {enhancedImaging && (
                    <div className="mt-4 rounded-xl border border-emerald-100 bg-emerald-50/30 p-4 shadow-sm">
                        <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-emerald-100/50">
                            <Activity className="h-4 w-4 text-emerald-600" />
                            <p className="text-[13.5px] font-semibold text-emerald-900 tracking-tight">AI Imaging Context</p>
                        </div>
                        <div className="prose prose-sm prose-zinc max-w-none text-[13.5px] leading-relaxed text-zinc-800 marker:text-emerald-600 prose-strong:font-semibold prose-strong:text-zinc-900">
                            <ReactMarkdown>{enhancedImaging}</ReactMarkdown>
                        </div>
                    </div>
                )}
            </div>
        </SectionCard>
    );
}

function AssessmentSection({ assessment }: { assessment: AssessmentInfo }) {
    const hasAny = !isEmpty(assessment.diagnosis_primary) || !isEmpty(assessment.suspected_primary) ||
        !isEmpty(assessment.differential) || !isEmpty(assessment.urgency) ||
        !isEmpty(assessment.infectious_concern) || !isEmpty(assessment.icu_candidate);

    if (!hasAny) return null;

    const urgencyVariant = (u: string | null) => {
        if (!u) return "neutral" as const;
        if (/emergent/i.test(u)) return "danger" as const;
        if (/semi/i.test(u)) return "warning" as const;
        return "neutral" as const;
    };

    return (
        <SectionCard icon={<Microscope className="h-4 w-4" />} title="Assessment" accent="emerald">
            <div className="space-y-3">
                {!isEmpty(assessment.diagnosis_primary) && (
                    <div className="rounded-xl bg-zinc-50/80 border border-zinc-200/80 px-4 py-3">
                        <p className="text-[13px] font-medium text-zinc-500 mb-1">Primary Diagnosis</p>
                        <p className="text-[15px] font-semibold text-zinc-900 capitalize">{assessment.diagnosis_primary}</p>
                    </div>
                )}

                {!isEmpty(assessment.suspected_primary) && (
                    <Field label="Suspected (Primary)" value={
                        <div className="flex flex-wrap gap-1.5 pt-0.5">
                            {assessment.suspected_primary.map((d) => <Chip key={d} label={d} variant="success" />)}
                        </div>
                    } />
                )}

                {!isEmpty(assessment.differential) && (
                    <Field label="Differential" value={
                        <ul className="space-y-1.5 mt-1">
                            {assessment.differential.map((d) => (
                                <li key={d} className="flex items-start gap-2 text-[14px] text-zinc-800">
                                    <span className="mt-[6px] h-1.5 w-1.5 rounded-full bg-zinc-400 shrink-0" /> {d}
                                </li>
                            ))}
                        </ul>
                    } />
                )}

                <div className="grid grid-cols-2 gap-3">
                    {!isEmpty(assessment.urgency) && (
                        <Field label="Urgency" value={<Chip label={assessment.urgency!} variant={urgencyVariant(assessment.urgency)} />} />
                    )}
                    {!isEmpty(assessment.infectious_concern) && (
                        <Field label="Infection" value={<YesNoBadge value={assessment.infectious_concern} />} />
                    )}
                    {!isEmpty(assessment.icu_candidate) && (
                        <Field label="ICU Candidate" value={<YesNoBadge value={assessment.icu_candidate} />} />
                    )}
                </div>
            </div>
        </SectionCard>
    );
}

function FindingRow({ label, value }: { label: string; value: string | null }) {
    if (isEmpty(value)) return null;
    return (
        <div className="flex items-center justify-between py-1.5 border-b border-zinc-100 last:border-0">
            <span className="text-[13px] text-zinc-500">{label}</span>
            <span className="text-[13px] font-medium text-zinc-800">{value}</span>
        </div>
    );
}

function FindingsSection({ findings }: { findings: FindingsInfo }) {
    const { lungs, pleura, cardiomediastinal, devices } = findings;

    const hasLungs = !isEmpty(lungs.consolidation_present) || !isEmpty(lungs.atelectasis_present) ||
        !isEmpty(lungs.edema_present) || (!isEmpty(lungs.edema_pattern) && lungs.edema_pattern !== "unknown");

    const hasPleura = !isEmpty(pleura.effusion_present) || !isEmpty(pleura.pneumothorax_present) ||
        (!isEmpty(pleura.effusion_side) && pleura.effusion_side !== "unknown");

    const hasCardio = !isEmpty(cardiomediastinal.cardiomegaly) || !isEmpty(cardiomediastinal.mediastinal_widening);

    const hasDevices = !isEmpty(devices.lines_tubes_present) || !isEmpty(devices.device_list);

    const imaging = findings.imaging_findings ?? [];
    const labs = findings.lab_findings ?? [];

    if (!hasLungs && !hasPleura && !hasCardio && !hasDevices && imaging.length === 0 && labs.length === 0) return null;

    return (
        <SectionCard icon={<Wind className="h-4 w-4" />} title="Findings" accent="slate">
            {(imaging.length > 0 || labs.length > 0) && (
                <div className="mb-4 space-y-3">
                    {imaging.length > 0 && (
                        <div>
                            <p className="text-[13px] font-medium text-zinc-500 mb-1.5">Imaging</p>
                            <ul className="space-y-1">
                                {imaging.map((f) => <li key={f} className="text-[14px] text-zinc-800">• {f}</li>)}
                            </ul>
                        </div>
                    )}
                    {labs.length > 0 && (
                        <div>
                            <p className="text-[13px] font-medium text-zinc-500 mb-1.5">Labs / pathology</p>
                            <ul className="space-y-1">
                                {labs.map((f) => <li key={f} className="text-[14px] text-zinc-800">• {f}</li>)}
                            </ul>
                        </div>
                    )}
                </div>
            )}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {hasLungs && (
                    <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/50 p-4 shadow-sm">
                        <p className="text-[13px] font-semibold text-zinc-700 mb-2">Lungs</p>
                        <FindingRow label="Consolidation" value={lungs.consolidation_present} />
                        <FindingRow label="Atelectasis" value={lungs.atelectasis_present} />
                        <FindingRow label="Edema" value={lungs.edema_present} />
                        {lungs.edema_pattern && lungs.edema_pattern !== "unknown" && (
                            <FindingRow label="Edema pattern" value={lungs.edema_pattern} />
                        )}
                    </div>
                )}

                {hasPleura && (
                    <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/50 p-4 shadow-sm">
                        <p className="text-[13px] font-semibold text-zinc-700 mb-2">Pleura</p>
                        <FindingRow label="Effusion" value={pleura.effusion_present} />
                        {pleura.effusion_side && pleura.effusion_side !== "unknown" && (
                            <FindingRow label="Effusion side" value={pleura.effusion_side} />
                        )}
                        <FindingRow label="Pneumothorax" value={pleura.pneumothorax_present} />
                    </div>
                )}

                {hasCardio && (
                    <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/50 p-4 shadow-sm">
                        <p className="text-[13px] font-semibold text-zinc-700 mb-2">Cardiomediastinal</p>
                        <FindingRow label="Cardiomegaly" value={cardiomediastinal.cardiomegaly} />
                        <FindingRow label="Mediastinal widening" value={cardiomediastinal.mediastinal_widening} />
                    </div>
                )}

                {hasDevices && (
                    <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/50 p-4 shadow-sm">
                        <p className="text-[13px] font-semibold text-zinc-700 mb-2">Devices / Lines</p>
                        <FindingRow label="Present" value={devices.lines_tubes_present} />
                        {!isEmpty(devices.device_list) && (
                            <div className="pt-1.5 flex flex-wrap gap-1">
                                {devices.device_list.map((d) => (
                                    <span key={d} className="inline-flex items-center rounded-md bg-zinc-200/60 text-zinc-700 px-2 py-0.5 text-[12px] font-medium">{d}</span>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </SectionCard>
    );
}

function SummarySection({ summary }: { summary: SummaryInfo }) {
    const hasAny = !isEmpty(summary.one_liner) || !isEmpty(summary.key_points) || !isEmpty(summary.red_flags);

    if (!hasAny) return null;

    return (
        <SectionCard icon={<ClipboardList className="h-4 w-4" />} title="Summary" accent="amber">
            <div className="space-y-3">
                {!isEmpty(summary.one_liner) && (
                    <p className="text-[15px] leading-relaxed text-zinc-800 italic border-l-[3px] border-zinc-300 pl-3 py-1">{summary.one_liner}</p>
                )}

                {!isEmpty(summary.key_points) && (
                    <div>
                        <p className="text-[13px] font-medium text-zinc-500 mb-2">Key Points</p>
                        <ul className="space-y-1.5">
                            {summary.key_points.map((kp) => (
                                <li key={kp} className="flex items-start gap-2 text-[14px] text-zinc-800">
                                    <span className="mt-[6px] h-1.5 w-1.5 rounded-full bg-zinc-400 shrink-0" />
                                    {kp}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}

                {!isEmpty(summary.red_flags) && (
                    <div>
                        <p className="text-[13px] font-medium text-rose-500 mb-2 flex items-center gap-1.5">
                            <AlertTriangle className="h-3.5 w-3.5" /> Red Flags
                        </p>
                        <ul className="space-y-1.5">
                            {summary.red_flags.map((rf) => (
                                <li key={rf} className="flex items-start gap-2 text-[14px] text-rose-700 bg-rose-50/50 rounded-xl px-4 py-2 border border-rose-100">
                                    <Zap className="h-4 w-4 text-rose-500 shrink-0 mt-0.5" />
                                    {rf}
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        </SectionCard>
    );
}

function OutcomeSection({ outcome }: { outcome: OutcomeInfo }) {
    const hasAny = !isEmpty(outcome.success) || !isEmpty(outcome.detail);
    if (!hasAny) return null;

    return (
        <SectionCard icon={<Heart className="h-4 w-4" />} title="Outcome" accent="emerald">
            <div className="flex items-start gap-3">
                {!isEmpty(outcome.success) && (
                    <span className={cn(
                        "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-[14px] font-medium shrink-0",
                        outcome.success === "yes"
                            ? "bg-emerald-50/80 text-emerald-700"
                            : "bg-rose-50/80 text-rose-700"
                    )}>
                        {outcome.success === "yes"
                            ? <CheckCircle2 className="h-3.5 w-3.5" />
                            : <XCircle className="h-3.5 w-3.5" />}
                        {outcome.success === "yes" ? "Successful" : "Unsuccessful"}
                    </span>
                )}
                {!isEmpty(outcome.detail) && (
                    <p className="text-[14px] leading-relaxed text-zinc-800 py-1">{outcome.detail}</p>
                )}
            </div>
        </SectionCard>
    );
}

function ProvenanceSection({ prov }: { prov: ProvenanceInfo }) {
    const hasData = prov.article_title || prov.journal || prov.pmc_id;
    if (!hasData) return null;

    return (
        <SectionCard icon={<BookOpen className="h-4 w-4" />} title="Source / Provenance" accent="slate">
            <div className="space-y-2">
                {prov.article_title && (
                    <p className="text-[14px] font-medium text-zinc-900 leading-relaxed">{prov.article_title}</p>
                )}
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-zinc-500">
                    {prov.journal && <span>{prov.journal}</span>}
                    {prov.year && <span>{prov.year}</span>}
                    {prov.pmid && <span>PMID: {prov.pmid}</span>}
                </div>
                {prov.authors.length > 0 && (
                    <p className="text-[13px] text-zinc-500">{prov.authors.join(", ")}</p>
                )}
                {prov.doi && (
                    <a
                        href={`https://doi.org/${prov.doi}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[13px] text-zinc-500 hover:text-zinc-900 transition-colors underline underline-offset-2"
                    >
                        DOI: {prov.doi} <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                )}
                {prov.source_url && !prov.doi && (
                    <a
                        href={prov.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 text-[13px] text-zinc-500 hover:text-zinc-900 transition-colors underline underline-offset-2"
                    >
                        View source <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                )}
            </div>
        </SectionCard>
    );
}

function TagsSection({ tags }: { tags: TagsInfo }) {
    const allKeywords = [...new Set([...tags.keywords, ...tags.ml_labels, ...tags.gt_labels])];
    const hasData = allKeywords.length > 0 || tags.mesh_terms.length > 0;
    if (!hasData) return null;

    return (
        <SectionCard icon={<Tag className="h-4 w-4" />} title="Tags & Labels" accent="violet">
            <div className="space-y-2.5">
                {allKeywords.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                        {allKeywords.map((k) => <Chip key={k} label={k} variant="neutral" />)}
                    </div>
                )}
                {tags.mesh_terms.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                        {tags.mesh_terms.map((m) => <Chip key={m} label={`MeSH: ${m}`} variant="primary" />)}
                    </div>
                )}
            </div>
        </SectionCard>
    );
}

function ExtraFieldsSection({ extra }: { extra: Record<string, string | string[]> }) {
    const entries = Object.entries(extra).filter(([, v]) => v !== null && v !== undefined && v !== "" && !(Array.isArray(v) && v.length === 0));
    if (entries.length === 0) return null;

    return (
        <SectionCard
            icon={<FlaskConical className="h-4 w-4" />}
            title="Additional Information"
        >
            <div className="grid grid-cols-1 gap-y-3 sm:grid-cols-2 sm:gap-x-6">
                {entries.map(([key, value]) => {
                    const label = key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
                    const displayValue = Array.isArray(value) ? value.join(", ") : value;
                    return (
                        <div key={key} className="space-y-0.5">
                            <p className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">{label}</p>
                            <p className="text-[14px] text-zinc-800 leading-snug break-words">{displayValue}</p>
                        </div>
                    );
                })}
            </div>
        </SectionCard>
    );
}


// ─── Main export ──────────────────────────────────────────────────────────────

export function CaseProfileView({
    profile,
    className,
    onEnhance,
    isEnhancing = false,
    enhancedSynthesis,
    enhancedImaging,
}: {
    profile: CaseProfile;
    className?: string;
    onEnhance?: () => void;
    isEnhancing?: boolean;
    enhancedSynthesis?: string;
    enhancedImaging?: string;
}) {
    const sections = [
        <PatientSection key="patient" patient={profile.patient} />,
        <PresentationSection key="presentation" pres={profile.presentation} />,
        <StudySection key="study" study={profile.study} enhancedImaging={enhancedImaging} />,
        <AssessmentSection key="assessment" assessment={profile.assessment} />,
        <FindingsSection key="findings" findings={profile.findings} />,
        <SummarySection key="summary" summary={profile.summary} />,
        <OutcomeSection key="outcome" outcome={profile.outcome} />,
        <ProvenanceSection key="provenance" prov={profile.provenance} />,
        <TagsSection key="tags" tags={profile.tags} />,
        <ExtraFieldsSection key="extra" extra={profile.extra_fields ?? {}} />,
    ];

    return (
        <SelectionExplainPopover>
            <div data-sel-root className={cn("space-y-4 relative", className)}>
                {/* Enhance Profile Button/Header */}
                {onEnhance && !enhancedSynthesis && (
                    <div className="flex justify-end mb-4">
                        <button
                            onClick={onEnhance}
                            disabled={isEnhancing}
                            className={cn(
                                "flex items-center gap-2 px-4 py-2 text-[13px] font-medium rounded-full transition-all shadow-sm border",
                                isEnhancing
                                    ? "bg-zinc-100 text-zinc-400 border-zinc-200 cursor-not-allowed"
                                    : "bg-white text-zinc-700 border-zinc-200/80 hover:bg-zinc-50 hover:text-zinc-900 active:bg-zinc-100 hover:shadow-md hover:border-zinc-300"
                            )}
                        >
                            {isEnhancing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4 text-emerald-600" />}
                            {isEnhancing ? "Enhancing Profile..." : "Enhance Profile"}
                        </button>
                    </div>
                )}

                {/* AI Clinical Synthesis Section */}
                {enhancedSynthesis && (
                    <SectionCard icon={<Activity className="h-4 w-4 text-white" />} title="AI Clinical Synthesis" accent="emerald" className="bg-emerald-50/40 p-5 rounded-2xl border border-emerald-100/50 -mx-5 px-5 pt-6 shadow-sm mb-6">
                        <div className="prose prose-sm prose-zinc max-w-none text-[14px] leading-relaxed text-zinc-800 marker:text-emerald-500 prose-strong:font-semibold prose-strong:text-zinc-900">
                            <ReactMarkdown>{enhancedSynthesis}</ReactMarkdown>
                        </div>
                    </SectionCard>
                )}

                {sections}
            </div>
        </SelectionExplainPopover>
    );
}

// Re-export for convenience
export type { CaseProfile };
