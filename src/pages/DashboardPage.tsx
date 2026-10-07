import { useCallback, useRef, useState, useMemo, useEffect, type ButtonHTMLAttributes, type ReactNode } from "react";
import { useDashboardStore } from "@/store/dashboardStore";
import { Check, FileText, Loader2, MapPin, Settings2, Stethoscope, FolderOpen, Plus, HeartPulse, CloudOff, Scan, Microscope, Activity, ChevronLeft, Building2, X, Phone, ChevronRight, Pill, Flag, RefreshCw } from "lucide-react";
import { CaseTopBar, type Step } from "@/components/CaseTopBar";
import { searchTwins, findHospitalsRoute, compareInsights, type ComparisonInsights, type MatchItem, type SearchResult } from "@/lib/twinApi";
import { API_BASE } from "@/lib/api";
import { computeProfileConfidence, PROFILE_READY_THRESHOLD } from "@/lib/caseProfileUtils";

import { type CaseProfile } from "@/lib/caseProfileTypes";
import { CaseProfileView } from "@/components/CaseProfileView";
import { AgenticCopilotPanel } from "@/components/AgenticCopilotPanel";
import { TwinProfileModal } from "@/components/TwinProfileModal";
import { TwinChatPanel } from "@/components/TwinChatPanel";
import { cn } from "@/lib/utils";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import ReactMarkdown from "react-markdown";
import { SelectionExplainPopover } from "@/components/SelectionExplainPopover";
import { AiPipelinePanel, ModelChip } from "@/components/AiPipelinePanel";
import { toast } from "sonner";

const defaultIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

type OutcomeVariant = "success" | "warning" | "neutral";

interface RouteCenter {
  name: string;
  url?: string;
  capability: string;
  travel: string;
  reason: string;
  lat?: number;
  lng?: number;
}

function SurfaceCard({
  className,
  children,
  label,
  title,
  id,
}: {
  className?: string;
  children: ReactNode;
  label?: string;
  title?: string;
  id?: string;
}) {
  return (
    <section id={id} className={cn("mr-surface", className)}>
      {(label || title) && (
        <div className="mb-3">
          {label && <p className="mr-label">{label}</p>}
          {title && <h2 className="mr-title">{title}</h2>}
        </div>
      )}
      {children}
    </section>
  );
}

function MedButton({
  className,
  variant,
  size = "default",
  fullWidth,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant: "primary" | "secondary" | "tertiary";
  size?: "default" | "sm";
  fullWidth?: boolean;
}) {
  return (
    <button
      className={cn(
        "mr-btn",
        variant === "primary" && "mr-btn--primary",
        variant === "secondary" && "mr-btn--secondary",
        variant === "tertiary" && "mr-btn--tertiary",
        size === "sm" && "mr-btn--sm",
        fullWidth && "mr-btn--full",
        className
      )}
      {...props}
    />
  );
}

function LabeledCheckbox({
  checked,
  label,
  onChange
}: {
  checked: boolean;
  label: string;
  onChange: (next: boolean) => void;
}) {
  return (
    <label className="inline-flex items-center gap-2 text-[15px] leading-[22px] text-[var(--mr-text)]">
      <button
        type="button"
        role="checkbox"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        data-checked={checked}
        className="mr-checkbox"
      >
        <Check className="h-3.5 w-3.5" />
      </button>
      <span>{label}</span>
    </label>
  );
}

function OutcomeBadge({ variant, label }: { variant: OutcomeVariant; label: string }) {
  return (
    <span
      className={cn(
        "mr-badge border border-zinc-200",
        variant === "success" && "mr-badge--success",
        variant === "warning" && "mr-badge--warning",
        variant === "neutral" && "mr-badge--neutral"
      )}
    >
      {label}
    </span>
  );
}

function UploadScreen({
  onImageFilePicked,
  onStepChange,
  uploadedFile,
}: {
  deIdentify?: boolean;
  saveToHistory?: boolean;
  onDeIdentifyChange?: (next: boolean) => void;
  onSaveHistoryChange?: (next: boolean) => void;
  onImageFilePicked: (file: File | null) => void;
  onStepChange: (step: Step) => void;
  uploadedFile?: File | null;
}) {
  const profile = useDashboardStore(s => s.profile);
  const setProfile = useDashboardStore(s => s.setProfile);
  const addTrace = useDashboardStore(s => s.addTrace);

  const conf = profile ? computeProfileConfidence(profile) : { score: 0, filled: 0, total: 13, missing: [] };

  const handleProfileUpdate = useCallback((updated: CaseProfile) => {
    setProfile(updated);
  }, [setProfile]);

  const handleFileForSearch = useCallback((file: File) => {
    onImageFilePicked(file);
  }, [onImageFilePicked]);

  const handleReadyToProceed = useCallback(() => {
    onStepChange(1); // advance to Review
  }, [onStepChange]);

  const [isEnhancing, setIsEnhancing] = useState(false);
  const enhancedSynthesis = useDashboardStore(s => s.enhancedSynthesis);
  const enhancedImaging = useDashboardStore(s => s.enhancedImaging);
  const setEnhanced = useDashboardStore(s => s.setEnhanced);

  const handleEnhanceProfile = async () => {
    if (!profile) return;

    setIsEnhancing(true);
    try {
      const fd = new FormData();
      fd.append("profile_json", JSON.stringify(profile));
      if (uploadedFile) {
        fd.append("file", uploadedFile);
      }

      const response = await fetch(`${API_BASE}/enhance_profile`, {
        method: "POST",
        body: fd,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Enhance failed (${response.status})`);
      }
      setEnhanced(data.synthesis, data.imaging_context ?? null);
      if (data.trace) addTrace("Clinical synthesis", data.trace);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "MedGemma could not enhance the profile.");
    } finally {
      setIsEnhancing(false);
    }
  };

  const containerRef = useRef<HTMLDivElement>(null);
  const [leftPct, setLeftPct] = useState(50);
  const isDragging = useRef(false);

  const onDividerMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setLeftPct(Math.max(25, Math.min(75, pct)));
    };
    const onMouseUp = () => {
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex h-full min-h-0 gap-0 pt-6"
      style={{ overflow: "hidden" }}
    >
      {/* ── Left: Live Case Profile ── */}
      <div
        className="flex flex-col gap-4 overflow-hidden h-full pr-3 pb-6 relative"
        style={{ width: `${leftPct}%`, minWidth: 0 }}
      >
        <div className="rounded-2xl border border-zinc-200/80 bg-white shadow-sm overflow-hidden flex flex-col flex-1 h-full min-h-0">
          {/* Header Section (Always Visible) */}
          <div className="px-8 pt-8 pb-5 border-b border-zinc-100 bg-zinc-50/50 flex items-start justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-zinc-200 bg-white shadow-sm">
                <FileText className="h-5 w-5 text-zinc-700" />
              </div>
              <div>
                <h2 className="text-[22px] font-semibold tracking-tight text-zinc-900 leading-none">Case Profile</h2>
                <p className="mt-1.5 flex items-center gap-1.5 text-[13px] font-medium text-zinc-500">
                  <Stethoscope className="h-3.5 w-3.5 text-[var(--mr-action)]" /> Structured by Gemma 4 · images read by MedGemma
                </p>
              </div>
            </div>

            {/* Inline Extraction Quality */}
            <div className="flex flex-col items-end">
              <span className={cn(
                "text-[12px] font-semibold px-2 py-0.5 rounded-md border",
                conf.score >= 80 ? "bg-emerald-50 text-emerald-700 border-emerald-200" : conf.score >= 50 ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-zinc-100 text-zinc-700 border-zinc-200"
              )}>
                {conf.score}% Complete
              </span>
            </div>
          </div>

          {/* Content Section */}
          <div className="flex-1 overflow-y-auto w-full relative">
            {profile && conf.score > 0 ? (
              <div className="p-8 pb-32 transition-all">
                <CaseProfileView
                  profile={profile}
                  onEnhance={handleEnhanceProfile}
                  isEnhancing={isEnhancing}
                  enhancedSynthesis={enhancedSynthesis ?? undefined}
                  enhancedImaging={enhancedImaging ?? undefined}
                />
              </div>
            ) : (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-8 bg-zinc-50/20 group">
                <div className="absolute inset-0 bg-gradient-to-tr from-white via-transparent to-zinc-50/30 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none"></div>

                <div className="flex flex-col items-center justify-center gap-5 text-center max-w-[320px] relative z-10 transition-transform duration-500 group-hover:-translate-y-1">
                  <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-white shadow-sm border border-zinc-200/60 ring-4 ring-white">
                    <div className="absolute inset-0 rounded-2xl bg-zinc-100/50 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                    <HeartPulse className="h-8 w-8 text-zinc-400 group-hover:text-rose-500 transition-colors duration-500" strokeWidth={1.5} />
                    <div className="absolute -bottom-1.5 -right-1.5 flex h-7 w-7 items-center justify-center rounded-full bg-white shadow-sm border border-zinc-200/60">
                      <FileText className="h-3.5 w-3.5 text-zinc-400" />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h3 className="text-[17px] font-semibold text-zinc-900 tracking-tight">Case Profile Empty</h3>
                    <p className="text-[14px] leading-relaxed text-zinc-500">
                      Talk to the Copilot on the right. Share patient evidence, labs, and imaging to watch the intelligent case profile automatically appear here.
                    </p>
                  </div>

                  <div className="mt-2 flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-zinc-200/60 shadow-sm text-[13px] font-medium text-zinc-500 transition-all duration-300 group-hover:shadow-md group-hover:border-zinc-300/60">
                    <Loader2 className="w-3.5 h-3.5 animate-[spin_3s_linear_infinite] text-zinc-400" />
                    Waiting for evidence...
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Drag Divider ── */}
      <div
        className="group relative z-10 flex w-3 shrink-0 cursor-col-resize items-center justify-center"
        onMouseDown={onDividerMouseDown}
      >
        <div className="h-full w-px bg-[var(--mr-border)] transition-colors group-hover:bg-[var(--mr-action)]" />
        <div className="absolute flex h-6 w-3 flex-col items-center justify-center gap-0.5 rounded-full">
          <span className="h-0.5 w-0.5 rounded-full bg-[var(--mr-text-secondary)] group-hover:bg-[var(--mr-action)]" />
          <span className="h-0.5 w-0.5 rounded-full bg-[var(--mr-text-secondary)] group-hover:bg-[var(--mr-action)]" />
          <span className="h-0.5 w-0.5 rounded-full bg-[var(--mr-text-secondary)] group-hover:bg-[var(--mr-action)]" />
        </div>
      </div>

      {/* ── Right: Full-height Agentic Copilot ── */}
      <div
        className="relative flex h-full min-h-[500px] flex-col pl-3"
        style={{ width: `${100 - leftPct}%`, minWidth: 0 }}
      >
        <AgenticCopilotPanel
          onProfileUpdate={handleProfileUpdate}
          onFileForSearch={handleFileForSearch}
          onReadyToProceed={handleReadyToProceed}
        />
      </div>
    </div>
  );
}



const OUTCOME_LABEL: Record<OutcomeVariant, string> = {
  success: "Favourable outcome",
  warning: "Poor outcome",
  neutral: "Outcome reported",
};

const CHANNEL_LABEL: Record<string, string> = {
  image: "Image",
  crossmodal: "Text→image",
  text: "Report",
  rerank: "Rerank",
};

const COLLECTION_LABEL: Record<string, string> = {
  cxr: "Chest X-ray",
  chest_ct: "Chest CT",
  derm: "Dermatology",
  fundus: "Fundus",
  histopath: "H&E pathology",
};

/** Per-channel similarity bars: the "why" behind a twin's score. */
function ScoreBreakdown({ item, compact }: { item: MatchItem; compact?: boolean }) {
  const entries = Object.entries(item.scores ?? {}).filter(([key]) => key in CHANNEL_LABEL);
  if (entries.length === 0) return null;
  return (
    <div className={cn("flex flex-wrap gap-x-3 gap-y-1", compact ? "text-[10px]" : "text-[11px]")} title="Similarity per signal, scaled to this library (weights in brackets)">
      {entries.map(([key, value]) => (
        <span key={key} className="inline-flex items-center gap-1 text-zinc-500"
          title={`raw ${key === "rerank" ? "logit" : "cosine"}: ${item.raw_scores?.[key as keyof typeof item.raw_scores]?.toFixed(3) ?? "n/a"}`}>
          <span className="font-medium text-zinc-600">{CHANNEL_LABEL[key]}</span>
          <span className="inline-block h-1.5 w-10 overflow-hidden rounded-full bg-zinc-100">
            <span className="block h-full rounded-full bg-[var(--mr-action)]" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
          </span>
          <span className="tabular-nums">{Math.round(value * 100)}</span>
          {!compact && item.weights?.[key] !== undefined && <span className="text-zinc-400">({item.weights[key]})</span>}
        </span>
      ))}
    </div>
  );
}

function MatchCard({
  item,
  selected,
  onSelect,
}: {
  item: MatchItem;
  selected: boolean;
  onSelect: () => void;
}) {
  const ringClass = item.score >= 75 ? "border-[var(--mr-action)] text-[var(--mr-action)]" : "border-[var(--mr-border)] text-[var(--mr-text)]";
  const outcomeKnown = item.outcome && item.outcome !== "Outcome not reported";

  return (
    <article
      className={cn(
        "mr-surface flex h-full flex-col gap-3 p-5 hover:shadow-md hover:border-[var(--mr-action)]/40 transition-all cursor-pointer group",
        selected && "border-[var(--mr-action)]"
      )}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between gap-3">
        <div className={cn("flex h-14 w-14 shrink-0 items-center justify-center rounded-full border-[3px] bg-white transition-colors group-hover:border-[var(--mr-action)] group-hover:text-[var(--mr-action)]", ringClass)}>
          <span className="text-[16px] font-semibold leading-[22px]">{item.score}%</span>
        </div>
        <div className="flex flex-col items-end gap-1.5 min-w-0">
          {outcomeKnown && <OutcomeBadge variant={item.outcomeVariant} label={OUTCOME_LABEL[item.outcomeVariant]} />}
          {item.collection && <span className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">{COLLECTION_LABEL[item.collection] ?? item.collection}</span>}
        </div>
      </div>

      <div className="space-y-1.5">
        <p className="text-[15px] font-semibold leading-[20px] text-zinc-900 group-hover:text-[var(--mr-action)] transition-colors line-clamp-2 break-words">{item.diagnosis}</p>
        <p className="text-[13px] leading-[19px] text-zinc-500 line-clamp-3 break-words">{item.summary}</p>
      </div>

      {(item.conclusion || outcomeKnown) && (
        <div className="rounded-lg border border-zinc-100 bg-zinc-50/70 px-3 py-2 text-[12px] leading-relaxed text-zinc-600 space-y-1">
          {item.conclusion && <p className="line-clamp-2"><span className="font-semibold text-zinc-800">Conclusion: </span>{item.conclusion}</p>}
          {outcomeKnown && <p className="line-clamp-2"><span className="font-semibold text-zinc-800">Outcome: </span>{item.outcome}</p>}
        </div>
      )}

      <div className="mt-auto space-y-2 pt-1">
        <ScoreBreakdown item={item} />
        <div className="flex flex-wrap text-[11px] text-zinc-400 gap-x-3 gap-y-1 font-medium">
          {item.pmc_id && <span className="flex items-center gap-1"><FileText className="h-3 w-3" /> {item.pmc_id}</span>}
          {item.year && <span>• {item.year}</span>}
          {item.journal && <span className="truncate max-w-[180px]">• {item.journal}</span>}
        </div>
      </div>
    </article>
  );
}

function FindingList({ items, empty }: { items: string[]; empty: string }) {
  if (items.length === 0) return <span className="text-zinc-400 italic text-[13px]">{empty}</span>;
  return (
    <ul className="flex flex-col gap-1 text-[13px]">
      {items.slice(0, 6).map((finding, i) => <li key={i}>• {finding}</li>)}
    </ul>
  );
}

/** What the published twin case found, did and concluded — the payoff of a twin match. */
function TwinStory({ item }: { item: MatchItem }) {
  const p = item.raw_payload ?? {};
  const treatments: string[] = [...(item.treatments ?? []), ...(p.management?.procedures ?? [])];
  const followUp: string | null = p.outcome?.follow_up ?? null;
  const outcomeKnown = item.outcome && item.outcome !== "Outcome not reported";
  const related = item.related_image_urls ?? [];
  if (!item.conclusion && !outcomeKnown && treatments.length === 0 && related.length === 0) return null;
  return (
    <div className="mb-8 rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm space-y-4">
      <h3 className="text-[18px] font-semibold text-zinc-900">What happened in the twin case</h3>
      <div className="grid gap-4 md:grid-cols-2 text-[14px] leading-relaxed text-zinc-700">
        {treatments.length > 0 && (
          <div>
            <p className="mb-1 flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-zinc-500"><Pill className="h-3.5 w-3.5" /> Management</p>
            <ul className="space-y-0.5">{treatments.slice(0, 6).map((t, i) => <li key={i}>• {t}</li>)}</ul>
          </div>
        )}
        {outcomeKnown && (
          <div>
            <p className="mb-1 flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-zinc-500"><Activity className="h-3.5 w-3.5" /> Outcome</p>
            <p>{item.outcome}</p>
            {followUp && <p className="mt-1 text-zinc-500">Follow-up: {followUp}</p>}
          </div>
        )}
        {item.conclusion && (
          <div className="md:col-span-2 rounded-lg bg-[var(--mr-action)]/5 border border-[var(--mr-action)]/15 px-4 py-3">
            <p className="mb-1 flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wider text-[var(--mr-action)]"><Flag className="h-3.5 w-3.5" /> Authors' conclusion</p>
            <p className="text-zinc-800">{item.conclusion}</p>
          </div>
        )}
      </div>
      {related.length > 0 && (
        <div>
          <p className="mb-2 text-[12px] font-semibold uppercase tracking-wider text-zinc-500">Other images from this case</p>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {related.map((url) => (
              <a key={url} href={url} target="_blank" rel="noreferrer" className="shrink-0">
                <img src={url} alt="Related case image" className="h-24 w-24 rounded-lg border border-zinc-200 object-cover hover:opacity-80" />
              </a>
            ))}
          </div>
        </div>
      )}
      <p className="text-[11px] text-zinc-400">Structured from the published case report. Highlight any term to have MedGemma explain it.</p>
    </div>
  );
}

function MatchesScreen({
  selectedMatch,
  onSelectMatch,
  // onContinueToRoute,
  items,
  isLoading,
  originalFile,
  originalProfile,
  searchMeta,
  onRefresh,
}: {
  selectedMatch: number | null;
  onSelectMatch: (index: number | null) => void;
  // onContinueToRoute: () => void;
  items: MatchItem[];
  isLoading: boolean;
  originalFile: File | null;
  originalProfile: CaseProfile | null;
  searchMeta: Omit<SearchResult, "matches"> | null;
  onRefresh: () => void;
}) {
  const selected = selectedMatch !== null ? items[selectedMatch] : null;
  const addTrace = useDashboardStore(s => s.addTrace);

  // The whole page scrolls now, so start each view (grid or twin detail) at the top.
  useEffect(() => {
    document.querySelector("main")?.scrollTo({ top: 0 });
  }, [selectedMatch]);
  const historicalCaption = typeof selected?.raw_payload?.study?.caption === "string"
    ? selected.raw_payload.study.caption : null;

  // Modals state
  const [showTwinProfile, setShowTwinProfile] = useState(false);
  const [showTwinChat, setShowTwinChat] = useState(false);

  // Keep track of which match ID we've loaded insights for
  const [lastInsightsMatchIdx, setLastInsightsMatchIdx] = useState<number | null>(null);

  const originalPreviewUrl = useMemo(() => {
    if (originalFile && originalFile.type.startsWith("image/")) {
      return URL.createObjectURL(originalFile);
    }
    return null;
  }, [originalFile]);

  const [showInsights, setShowInsights] = useState(false);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [insights, setInsights] = useState<ComparisonInsights | null>(null);
  const [currentImageSize, setCurrentImageSize] = useState<{ src: string; width: number; height: number } | null>(null);
  const [historicalImageSize, setHistoricalImageSize] = useState<{ src: string; width: number; height: number } | null>(null);

  const renderSuggestedBox = (box: [number, number, number, number] | null, size: { width: number; height: number } | null) => {
    if (!box || !size) return null;
    const [y0, x0, y1, x1] = box;
    return (
      <svg
        data-testid="suggested-finding-box"
        role="img"
        aria-label="AI-suggested finding region"
        className="pointer-events-none absolute inset-0 h-full w-full"
        viewBox={`0 0 ${size.width} ${size.height}`}
        preserveAspectRatio="xMidYMid meet"
      >
        <rect
          x={x0 * size.width / 1000}
          y={y0 * size.height / 1000}
          width={(x1 - x0) * size.width / 1000}
          height={(y1 - y0) * size.height / 1000}
          fill="rgba(37, 99, 235, 0.16)"
          stroke="#2563eb"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    );
  };

  // Clear insights if the selected match has changed
  useEffect(() => {
    if (selectedMatch !== lastInsightsMatchIdx) {
      setInsights(null);
      setInsightsError(null);
      setLastInsightsMatchIdx(selectedMatch);
    }
  }, [selectedMatch, lastInsightsMatchIdx]);

  // Load insights when selected case changes and toggle is active
  useEffect(() => {
    if (selected && originalFile && showInsights && !insights && !insightsLoading && !insightsError) {
      setInsightsLoading(true);
      compareInsights(originalFile, selected)
        .then((res) => {
          setInsights(res);
          if (res.trace) addTrace("Image comparison", res.trace);
        })
        .catch((err) => setInsightsError(err instanceof Error ? err.message : "Analysis failed. Please try again."))
        .finally(() => setInsightsLoading(false));
    }
  }, [selected, originalFile, showInsights, insights, insightsLoading, insightsError, addTrace]);

  // Handle toggle click
  const handleToggleInsights = () => {
    if (!showInsights) {
      setInsightsError(null);
      setShowInsights(true);
    } else {
      setShowInsights(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 pb-12">
      {/* Grid of twins (page scrolls as a whole; model details live in the AI pipeline drawer) */}
      {selected === null && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h1 className="text-[28px] font-semibold text-zinc-900 tracking-tight">Closest Case Twins</h1>
            <div className="flex items-center gap-3">
              {!isLoading && items.length > 0 && (
                <span className="text-[13px] font-medium text-zinc-500 tracking-wide uppercase">Top {items.length} results</span>
              )}
              <button onClick={onRefresh} disabled={isLoading} className="flex items-center gap-1 text-[12px] font-medium text-[var(--mr-action)] hover:underline disabled:opacity-50">
                <RefreshCw className="h-3.5 w-3.5" /> Re-run search
              </button>
            </div>
          </div>

          {isLoading ? (
            <div className="flex flex-col items-center justify-center gap-4 py-24 text-[var(--mr-text-secondary)] bg-zinc-50/50 rounded-2xl border border-dashed border-zinc-200">
              <Loader2 className="h-8 w-8 animate-spin text-[var(--mr-action)]" />
              <p className="text-[15px] font-medium text-zinc-600">Embedding the image and case report, searching and reranking twins…</p>
            </div>
          ) : items.length === 0 ? (
            <div className="flex flex-col items-center justify-center gap-4 py-24 text-[var(--mr-text-secondary)] bg-zinc-50/50 rounded-2xl border border-dashed border-zinc-200">
              <p className="text-[15px] font-medium text-zinc-600">No twins found. Add an image or more case details, then re-run the search.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
              {items.map((item, idx) => (
                <MatchCard key={item.id ?? idx} item={item} selected={false} onSelect={() => onSelectMatch(idx)} />
              ))}
            </div>
          )}
        </>
      )}

      {/* Right Detail Container (Big Canvas) */}
      {selected !== null && (
        <div className="relative flex w-full flex-col rounded-2xl border border-zinc-200/80 bg-white shadow-sm animate-in fade-in duration-300">
          {/* Canvas Header */}
          <div className="flex items-center justify-between border-b border-zinc-100 bg-zinc-50/50 px-6 py-4 shrink-0">
            <div className="flex items-center gap-3">
              <button
                onClick={() => onSelectMatch(null)}
                className="flex items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-3 py-1.5 text-[13px] font-medium text-zinc-700 hover:bg-zinc-100 hover:text-zinc-900 transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                Back to matches
              </button>
              <h2 className="text-[17px] font-semibold text-zinc-900">In-depth Comparison</h2>
            </div>
            <div className="flex items-center gap-3">
              {/* Run AI Analysis (MedGemma image comparison) is hidden for the demo: MedGemma's
                  reads of published figures were not reliable enough (left/right swaps, missed
                  findings). The /compare_insights endpoint is kept; restore this button to re-enable.
              <button
                onClick={handleToggleInsights}
                className={cn(
                  "flex items-center gap-2 px-3 py-1.5 rounded-full text-[13px] font-medium transition-all group border",
                  showInsights
                    ? "bg-[var(--mr-action)] text-white border-[var(--mr-action)] shadow-inner"
                    : "bg-white text-zinc-600 border-zinc-200 hover:border-[var(--mr-action)]/30 hover:bg-[var(--mr-action)]/5"
                )}
              >
                <Microscope className={cn("h-4 w-4", showInsights ? "text-white" : "text-[var(--mr-action)]")} />
                {showInsights ? "AI Analysis Active" : "Run AI Analysis"}
              </button>

              <div className="w-px h-5 bg-zinc-200 mx-1" />
              */}

              <MedButton variant="secondary" size="sm" onClick={() => setShowTwinProfile(true)}>
                Full Profile
              </MedButton>
              {/* <MedButton variant="primary" size="sm" onClick={onContinueToRoute}>
                Continue to routing
              </MedButton> */}
            </div>
          </div>

          {/* Canvas Content — highlight any term to have MedGemma explain it */}
          <SelectionExplainPopover className="bg-zinc-50/30 p-6 md:p-8">
            <div className="max-w-[1200px] mx-auto space-y-8 pb-10">

              {/* Dual Image Comparison Banner */}
              <div className="grid grid-flow-row md:grid-cols-2 gap-8 items-stretch pt-2">
                {/* Left: Original */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-[15px] font-semibold text-zinc-900 flex items-center gap-2">
                      Your Upload
                      <span className="bg-zinc-100 text-zinc-500 px-2 py-0.5 rounded text-[11px] uppercase tracking-wide">Current</span>
                    </h3>
                  </div>
                  <div className="aspect-[4/3] rounded-2xl border border-zinc-200 overflow-hidden bg-zinc-100 relative shadow-inner">
                    {originalPreviewUrl ? (
                      <>
                        <img
                          src={originalPreviewUrl}
                          alt="Your uploaded image"
                          className="w-full h-full object-contain bg-black/5"
                          onLoad={(event) => setCurrentImageSize({
                            src: originalPreviewUrl,
                            width: event.currentTarget.naturalWidth,
                            height: event.currentTarget.naturalHeight,
                          })}
                        />
                        {showInsights && insights && renderSuggestedBox(
                          insights.original_box,
                          currentImageSize?.src === originalPreviewUrl ? currentImageSize : null,
                        )}
                      </>
                    ) : (
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <FileText className="h-10 w-10 text-zinc-300 mb-2" />
                        <p className="text-[13px] text-zinc-500">No original image</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right: Matched Case */}
                <div className="space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <h3 className="text-[15px] font-semibold text-zinc-900 flex flex-wrap items-center gap-2">
                      Historical Case Twin
                      <OutcomeBadge variant={selected.outcomeVariant} label={`${selected.score}% Match`} />
                    </h3>
                    <div className="flex gap-2">
                      {/* Buttons moved elsewhere */}
                    </div>
                  </div>
                  <div className="aspect-[4/3] rounded-2xl border border-zinc-200 overflow-hidden bg-zinc-100 relative shadow-inner">
                    {selected.image_url ? (
                      <>
                        <img
                          src={selected.image_url}
                          alt="Twin case image"
                          className="w-full h-full object-contain bg-black/5"
                          onLoad={(event) => setHistoricalImageSize({
                            src: selected.image_url,
                            width: event.currentTarget.naturalWidth,
                            height: event.currentTarget.naturalHeight,
                          })}
                        />
                        {showInsights && insights && renderSuggestedBox(
                          insights.match_box,
                          historicalImageSize?.src === selected.image_url ? historicalImageSize : null,
                        )}
                      </>
                    ) : (
                      <div className="absolute inset-0 flex flex-col items-center justify-center">
                        <FileText className="h-10 w-10 text-zinc-300 mb-2" />
                        <p className="text-[13px] text-zinc-500">No image available</p>
                      </div>
                    )}
                  </div>
                  {(historicalCaption || selected.summary) && (
                    <p className="text-xs leading-relaxed text-zinc-500">
                      <span className="font-semibold text-zinc-600">
                        {historicalCaption ? "Published image caption:" : "Historical case summary:"}
                      </span> {historicalCaption || selected.summary}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <TwinStory item={selected} />

            {/* Streaming Single AI Insight */}
            {showInsights && (
              <div className="bg-[var(--mr-action)]/5 rounded-2xl border border-[var(--mr-action)]/20 p-6 animate-in fade-in slide-in-from-top-4 mb-6">
                <div className="flex items-start gap-4">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white border border-[var(--mr-action)]/30 text-[var(--mr-action)] shadow-sm">
                    {insightsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Microscope className="h-5 w-5" />}
                  </div>
                  <div className="flex-1 mt-0.5 space-y-2 overflow-hidden">
                    <h4 className="text-[16px] font-semibold text-zinc-900">AI Visual Comparison</h4>
                    <div className="text-[14px] leading-relaxed text-zinc-700 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                      {insightsLoading ? (
                        <p className="text-zinc-500 italic">Comparing visible findings in both images...</p>
                      ) : insightsError ? (
                        <div role="alert" className="space-y-3">
                          <p className="text-red-700">{insightsError}</p>
                          <button
                            type="button"
                            onClick={() => setInsightsError(null)}
                            className="rounded-lg border border-[var(--mr-action)] px-3 py-1.5 font-medium text-[var(--mr-action)] hover:bg-[var(--mr-action)]/5"
                          >
                            Retry analysis
                          </button>
                        </div>
                      ) : insights?.insights_text ? (
                        <div className="prose prose-zinc prose-sm md:prose-base max-w-none whitespace-pre-line
                            prose-p:leading-relaxed prose-p:text-zinc-700 
                            prose-headings:text-zinc-900 prose-headings:font-semibold 
                            prose-strong:text-zinc-900 prose-strong:font-semibold
                            prose-li:text-zinc-700 prose-ul:my-2 prose-li:my-1">
                          <ReactMarkdown>{insights.insights_text}</ReactMarkdown>
                          <p className="text-xs text-zinc-500">
                            AI-generated visual context: each image was read separately by MedGemma, then the reads were compared.
                            Sides (left/right) are left out because the model often confuses them on published figures;
                            {(insights.original_box || insights.match_box) ? " the boxes from MedGemma 1.5 show where each finding is." : " check location on the images."}
                            {" "}Check it against both images and the published caption.
                          </p>
                        </div>
                      ) : (
                        <p className="text-zinc-400 italic text-sm">No analysis available.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}


            <div className="w-full h-px bg-zinc-200/60" />

            {/* Structural Data Comparison */}
            <div>
              <h3 className="text-[18px] font-semibold text-zinc-900 mb-5">Clinical Comparison Matrix</h3>

              <div className="rounded-xl border border-zinc-200 bg-white overflow-hidden shadow-sm">
                <table className="w-full table-fixed text-left border-collapse text-[14px]">
                  <thead>
                    <tr className="bg-zinc-50 border-b border-zinc-200/80">
                      <th className="py-3 px-4 font-semibold text-zinc-500 uppercase tracking-wider text-xs w-1/4">Clinical Feature</th>
                      <th className="py-3 px-4 font-semibold text-zinc-900 border-l border-zinc-200/80 w-[37.5%]">Your Uploaded Case</th>
                      <th className="py-3 px-4 font-semibold text-zinc-900 border-l border-zinc-200/80 w-[37.5%] flex items-center gap-2">
                        Historical Twin <Check className="h-4 w-4 text-[var(--mr-success)]" />
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-100">

                    {/* Row 1: Demographics */}
                    <tr className="hover:bg-zinc-50/50 transition-colors">
                      <td className="py-3 px-4 text-zinc-600 font-medium">Demographics</td>
                      <td className="py-3 px-4 border-l border-zinc-200/80">
                        {originalProfile?.patient.age_years ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 text-xs font-semibold mr-2 border border-blue-100">
                            {originalProfile.patient.age_years}y
                          </span>
                        ) : "— "}
                        {originalProfile?.patient.sex ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-purple-50 text-purple-700 text-xs font-semibold border border-purple-100">
                            {originalProfile.patient.sex}
                          </span>
                        ) : ""}
                      </td>
                      <td className="py-3 px-4 border-l border-zinc-200/80">
                        {selected.age ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-blue-50 text-blue-700 text-xs font-semibold mr-2 border border-blue-100">
                            {selected.age}y
                          </span>
                        ) : "— "}
                        {selected.gender ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-md bg-purple-50 text-purple-700 text-xs font-semibold border border-purple-100">
                            {selected.gender}
                          </span>
                        ) : ""}
                      </td>
                    </tr>

                    {/* Row 2: Primary Diagnosis/Assessment */}
                    <tr className="hover:bg-zinc-50/50 transition-colors">
                      <td className="py-3 px-4 text-zinc-600 font-medium">Primary Indication</td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-900">
                        {originalProfile?.assessment.diagnosis_primary || "Pending determination"}
                      </td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-[var(--mr-action)] font-medium">
                        {selected.diagnosis}
                      </td>
                    </tr>

                    {/* Row 3: Key Findings */}
                    <tr className="hover:bg-zinc-50/50 transition-colors">
                      <td className="py-3 px-4 text-zinc-600 font-medium">Imaging Findings</td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-700">
                        <FindingList items={originalProfile?.findings.imaging_findings ?? []} empty="No imaging findings extracted." />
                      </td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-700">
                        <FindingList items={selected.imaging_findings ?? []} empty="Not reported in the case." />
                      </td>
                    </tr>

                    {/* Row 4: Comorbidities */}
                    <tr className="hover:bg-zinc-50/50 transition-colors">
                      <td className="py-3 px-4 text-zinc-600 font-medium">Comorbidities</td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-700 text-[13px]">
                        {originalProfile?.patient.comorbidities.join(", ") || <span className="text-zinc-400 italic">None recorded</span>}
                      </td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-700 text-[13px]">
                        {(selected.raw_payload?.patient?.comorbidities ?? []).join(", ") || <span className="text-zinc-400 italic">None recorded</span>}
                      </td>
                    </tr>

                    {/* Row 5: Evidence Base */}
                    <tr className="hover:bg-zinc-50/50 transition-colors">
                      <td className="py-3 px-4 text-zinc-600 font-medium">Evidence Base</td>
                      <td className="py-3 px-4 border-l border-zinc-200/80 text-zinc-400 text-sm">
                        Active clinical case
                      </td>
                      <td className="py-3 px-4 border-l border-zinc-200/80">
                        <div className="flex flex-col gap-1 text-[13px]">
                          <span className="font-semibold text-zinc-900 break-words line-clamp-2">{selected.article_title || selected.diagnosis}</span>
                          <span className="font-medium text-zinc-600 max-w-full truncate">{selected.facility}</span>
                          <div className="flex flex-wrap items-center gap-x-2 text-xs text-zinc-500 mt-1">
                            {selected.pmc_id && (
                              <a href={selected.source_url || `https://pmc.ncbi.nlm.nih.gov/articles/${selected.pmc_id}/`} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">
                                {selected.pmc_id}
                              </a>
                            )}
                            {selected.journal && <span className="truncate max-w-[120px]">• {selected.journal}</span>}
                            {selected.year && <span>• {selected.year}</span>}
                            {selected.license && <span>• {selected.license}</span>}
                          </div>
                        </div>
                      </td>
                    </tr>

                  </tbody>
                </table>
              </div>

            </div>

          </SelectionExplainPopover>
          {/* Chat FAB */}
          <button
            onClick={() => setShowTwinChat(true)}
            className="fixed bottom-6 right-6 flex items-center gap-2 px-4 py-2.5 rounded-full bg-zinc-900 text-white text-[13px] font-semibold shadow-xl hover:bg-zinc-800 hover:scale-105 active:scale-95 transition-all z-10"
            aria-label="Open clinical copilot"
          >
            <Activity className="h-4 w-4" />
            Ask Copilot
          </button>
        </div>
      )}

      {/* Render Twin Modals */}
      <TwinProfileModal
        isOpen={showTwinProfile}
        onClose={() => setShowTwinProfile(false)}
        match={selected}
      />
      <TwinChatPanel
        isOpen={showTwinChat}
        onClose={() => setShowTwinChat(false)}
        match={selected}
        currentProfile={originalProfile ?? null}
      />
    </div >
  );
}

function CenterRow({ center, onClick, condensed }: { center: RouteCenter, onClick?: () => void, condensed?: boolean }) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "flex flex-col gap-2.5 border-b border-zinc-100 p-4 transition-all hover:bg-zinc-50/80 active:bg-zinc-100/50 cursor-pointer group last:border-0",
        !condensed && "sm:grid sm:grid-cols-[1.5fr_1fr_1fr_2fr] sm:items-center sm:gap-4 sm:p-5"
      )}
    >
      <div className={cn("flex flex-col", condensed ? "gap-1" : "gap-1.5")}>
        <div className="flex items-start justify-between gap-2">
          <h3 className={cn("font-semibold text-zinc-900 group-hover:text-[var(--mr-action)] transition-colors", condensed ? "text-[14px] line-clamp-2 leading-tight" : "text-[15px]")}>
            {center.name}
          </h3>
        </div>
        {condensed && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-zinc-500 font-medium">
            <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5" />{center.travel}</span>
            <span className="flex items-center gap-1"><Activity className="w-3.5 h-3.5" />{center.capability}</span>
          </div>
        )}
      </div>

      {!condensed && (
        <div className="flex flex-col gap-1.5">
          <span className="text-[12px] font-medium text-zinc-500 uppercase tracking-wider">Capability</span>
          <div className="flex items-center gap-2">
            <span className="text-zinc-900 font-medium text-[14px]">{center.capability}</span>
            <span className="h-1.5 w-12 rounded-full bg-zinc-100 overflow-hidden border border-zinc-200/50">
              <span className="block h-full rounded-full bg-[var(--mr-action)] transition-all" style={{ width: center.capability }} />
            </span>
          </div>
        </div>
      )}

      {!condensed && (
        <div className="flex flex-col gap-1">
          <span className="text-[12px] font-medium text-zinc-500 uppercase tracking-wider">Travel Match</span>
          <p className="text-zinc-900 text-[14px] font-medium flex items-center gap-1.5">
            <MapPin className="w-4 h-4 text-zinc-400" />
            {center.travel}
          </p>
        </div>
      )}

      {!condensed && (
        <div className="flex flex-col gap-1">
          <span className="text-[12px] font-medium text-zinc-500 uppercase tracking-wider">AI Rationale</span>
          <p className="text-zinc-600 text-[13px] line-clamp-2 leading-relaxed">{center.reason}</p>
        </div>
      )}
    </div>
  );
}

const highlightKeywords = (text: string, condition: string | null = null, equipment: Record<string, boolean> = {}) => {
  if (!text) return text;
  const terms = [condition, ...Object.keys(equipment).filter(k => equipment[k])].filter(Boolean) as string[];
  if (terms.length === 0) return text;

  let wordsToHighlight: string[] = [];
  terms.forEach(t => {
    if (t.length > 3) {
      wordsToHighlight.push(t);
      if (t.includes(' ')) {
        wordsToHighlight.push(...t.split(' ').filter(w => w.length > 3));
      }
    }
  });

  wordsToHighlight = Array.from(new Set(wordsToHighlight.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))));
  const regex = new RegExp(`(${wordsToHighlight.join('|')})`, 'gi');
  const parts = text.split(regex);

  return (
    <p className="text-[15px] leading-[24px] text-zinc-700">
      {parts.map((part, i) =>
        regex.test(part) ? (
          <span key={i} className="bg-yellow-200/80 text-yellow-900 px-1 rounded-[4px] font-medium border border-yellow-300/50 shadow-sm">{part}</span>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  );
};

function RouteScreen({
  equipment,
  maxTravelTime,
  maxDistance,
  language,
  centers,
  isLoading,
  error,
  selectedHospital,
  patientCondition,
  onEquipmentToggle,
  onMaxTravelTimeChange,
  onMaxDistanceChange,
  onLanguageChange,
  onHospitalClick,
  onUpdateSearch,
  onProceedToMemo,
  userCoords
}: {
  equipment: Record<string, boolean>;
  maxTravelTime: number;
  maxDistance: string;
  language: string;
  centers: RouteCenter[];
  isLoading: boolean;
  error: string | null;
  selectedHospital: RouteCenter | null;
  patientCondition: string;
  onEquipmentToggle: (key: string, value: boolean) => void;
  onMaxTravelTimeChange: (value: number) => void;
  onMaxDistanceChange: (value: string) => void;
  onLanguageChange: (value: string) => void;
  onHospitalClick: (center: RouteCenter | null) => void;
  onUpdateSearch: () => void;
  onProceedToMemo: () => void;
  userCoords: string | null;
}) {
  const { extractedSpecialists, setExtractedSpecialists: setStoreSpecialists } = useDashboardStore();
  const [isExtractingSpecialists, setIsExtractingSpecialists] = useState(false);

  // Ref keeps latest cache snapshot without being a useEffect dependency,
  // so the effect only re-fires when the selected hospital actually changes.
  const extractedSpecialistsRef = useRef(extractedSpecialists);
  extractedSpecialistsRef.current = extractedSpecialists;

  // Tracks the AbortController of the current in-flight request so we can
  // cancel it immediately if the user switches to a different hospital.
  const inflightRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!selectedHospital) return;
    const name = selectedHospital.name;

    // Already have data for this hospital — nothing to do.
    if (extractedSpecialistsRef.current[name] !== undefined) return;

    // Cancel the previous request (different hospital) if still running.
    inflightRef.current?.abort();
    const controller = new AbortController();
    inflightRef.current = controller;

    setIsExtractingSpecialists(true);

    const formData = new FormData();
    formData.append("url", selectedHospital.url || "");
    formData.append("diagnosis", patientCondition);
    formData.append("hospital_name", name);
    if (userCoords) formData.append("location", userCoords);

    fetch(`${API_BASE}/analyze_hospital_page`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    })
      .then((res) => res.json())
      .then((data) => {
        if (!controller.signal.aborted) {
          setStoreSpecialists(name, data.specialists || []);
        }
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          setStoreSpecialists(name, []);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsExtractingSpecialists(false);
        }
      });
  }, [selectedHospital, patientCondition, userCoords, setStoreSpecialists]);

  const safeCenters = Array.isArray(centers) ? centers : [];
  const validCenters = safeCenters.filter(c => typeof c.lat === "number" && typeof c.lng === "number");
  const mapCenter: [number, number] = validCenters.length > 0 && validCenters[0].lat !== undefined && validCenters[0].lng !== undefined
    ? [validCenters[0].lat, validCenters[0].lng]
    : [39.8283, -98.5795]; // Default to US center


  return (
    <div className={cn(
      "flex h-full min-h-0 gap-6 transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)]",
      selectedHospital === null ? "flex-col lg:flex-row" : "flex-col lg:flex-row"
    )}>
      {/* Left Container */}
      <div className={cn(
        "flex flex-col gap-6 transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)]",
        selectedHospital === null
          ? "w-full lg:max-w-[720px] xl:max-w-[800px] shrink-0 opacity-100 h-full"
          : "hidden lg:flex lg:max-w-[360px] xl:max-w-[420px] shrink-0 w-full opacity-100 h-full"
      )}>

        {/* Header for List View */}
        {selectedHospital === null && (
          <div className="flex items-center justify-between shrink-0 mb-[-10px]">
            <h1 className="text-[28px] font-semibold text-zinc-900 tracking-tight">
              Routing Matches
            </h1>
            <span className="mr-badge mr-badge--neutral">Top {centers.length} Centers</span>
          </div>
        )}

        {/* Map Preview */}
        <SurfaceCard className={cn(
          "relative overflow-hidden p-0 border-zinc-200/80 shadow-sm z-0 transition-all duration-500 shrink-0",
          selectedHospital === null ? "h-[240px] rounded-2xl" : "h-[200px] rounded-2xl"
        )}>
          <div className="absolute inset-0 z-0">
            <MapContainer
              center={mapCenter}
              zoom={centers.length > 0 ? 4 : 3}
              scrollWheelZoom={true}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
              />
              {validCenters.map((center, idx) => (
                center.lat !== undefined && center.lng !== undefined && (
                  <Marker key={idx} position={[center.lat, center.lng]} icon={defaultIcon}>
                    <Popup>
                      <div className="px-1 py-0.5">
                        <strong className="block text-zinc-900 font-semibold text-[13px] mb-1">{center.name}</strong>
                        <div className="text-[12px] text-zinc-600 space-y-0.5">
                          <span className="block"><span className="font-medium text-zinc-500">Match:</span> {center.capability}</span>
                          <span className="block"><span className="font-medium text-zinc-500">Distance:</span> {center.travel}</span>
                        </div>
                      </div>
                    </Popup>
                  </Marker>
                )
              ))}
            </MapContainer>
          </div>
        </SurfaceCard>

        {/* Facility List */}
        <SurfaceCard className={cn(
          "gap-0 p-0 flex-1 min-h-0 overflow-y-auto border-zinc-200/80 shadow-sm",
          selectedHospital === null ? "rounded-2xl" : "rounded-2xl bg-zinc-50/30"
        )}>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-16 text-zinc-500">
              <Loader2 className="h-8 w-8 animate-spin mb-4 text-[var(--mr-action)]" />
              <p className="text-[15px] font-medium text-zinc-700">Analyzing specialized centers...</p>
              <p className="text-[13px] text-zinc-500 mt-1">Matching capabilities to patient needs</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center py-16 text-red-500 px-6 text-center">
              <CloudOff className="h-8 w-8 mb-4 opacity-80" />
              <p className="text-[15px] font-medium text-zinc-900">Routing Search Unavailable</p>
              <p className="text-[13px] mt-1.5 text-zinc-500 max-w-sm">{error}</p>
              <MedButton variant="secondary" size="sm" className="mt-4" onClick={onUpdateSearch}>Retry Search</MedButton>
            </div>
          ) : centers.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-zinc-400">
              <Building2 className="h-10 w-10 mb-4 opacity-30" />
              <p className="text-[15px] font-medium text-zinc-600">No matching centers found</p>
              <p className="text-[13px] text-zinc-500 mt-1">Try adjusting your equipment or distance filters.</p>
            </div>
          ) : (
            <div className="divide-y divide-zinc-100">
              {centers.map((center) => (
                <CenterRow
                  key={center.name}
                  center={center}
                  onClick={() => onHospitalClick(selectedHospital?.name === center.name ? null : center)}
                  condensed={selectedHospital !== null}
                />
              ))}
            </div>
          )}
        </SurfaceCard>
      </div>

      {/* Right Canvas / Filters Container */}
      {selectedHospital !== null ? (
        <div className="flex-1 rounded-2xl border border-zinc-200/80 bg-white shadow-sm flex flex-col overflow-hidden animate-in fade-in zoom-in-95 slide-in-from-right-8 duration-500 ease-[cubic-bezier(0.23,1,0.32,1)]">
          <div className="flex items-center justify-between border-b border-zinc-100 bg-white px-5 py-4 shrink-0">
            <div className="flex items-center gap-3">
              <button
                onClick={() => onHospitalClick(null)}
                className="flex items-center justify-center h-8 w-8 rounded-full hover:bg-zinc-100 transition-colors text-zinc-500 hover:text-zinc-900"
                aria-label="Close hospital details"
              >
                <ChevronLeft className="h-5 w-5" />
              </button>
              <h2 className="text-[15px] font-semibold text-zinc-900">Routing Options</h2>
            </div>
            <div className="flex items-center gap-2">
              <MedButton variant="primary" size="sm" onClick={onProceedToMemo}>
                Prepare Memo
              </MedButton>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto bg-white p-6 md:p-8">
            <div className="max-w-[800px] mx-auto space-y-8">
              <div className="space-y-4">
                <div className="space-y-1.5 flex flex-col items-start">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[var(--mr-action)]/10 text-[var(--mr-action)] text-[12px] font-semibold tracking-wide uppercase mb-1">
                    <Building2 className="w-3.5 h-3.5" /> Facility Details
                  </div>
                  <h3 className="text-[24px] md:text-[28px] font-semibold tracking-[-0.01em] text-zinc-900 leading-tight">{selectedHospital.name}</h3>
                </div>

                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1.5 text-[14px] text-zinc-600 font-medium bg-zinc-50 px-3 py-1 rounded-full border border-zinc-200/80">
                    <MapPin className="w-4 h-4 text-zinc-400" /> Approx Travel Time: {selectedHospital.travel}
                  </span>
                  <span className="flex items-center gap-1.5 text-[14px] text-zinc-600 font-medium bg-zinc-50 px-3 py-1 rounded-full border border-zinc-200/80">
                    <Activity className="w-4 h-4 text-zinc-400" /> Capability: {selectedHospital.capability}
                  </span>
                </div>
              </div>

              <div className="w-full h-px bg-zinc-100" />

              <section className="space-y-3">
                <h4 className="text-[13px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                  <Activity className="w-4 h-4" /> Selection Rationale
                </h4>
                <div className="flex flex-col gap-2 relative pl-4 before:absolute before:left-0 before:top-1 before:bottom-1 before:w-[3px] before:bg-indigo-500 before:rounded-full">
                  <p className="text-[15px] leading-relaxed text-zinc-800">
                    {highlightKeywords(selectedHospital.reason, patientCondition, equipment)}
                  </p>
                </div>
              </section>

              <div className="w-full h-px bg-zinc-100" />

              <section className="space-y-4 pb-8">
                <h4 className="text-[13px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                  <Stethoscope className="w-4 h-4" /> Top Specialists & Programs
                </h4>

                <div className="grid gap-4">
                  {isExtractingSpecialists && (!extractedSpecialists[selectedHospital.name] || extractedSpecialists[selectedHospital.name].length === 0) ? (
                    <div className="flex items-center gap-3 py-4 text-zinc-500">
                      <Loader2 className="w-5 h-5 animate-spin text-[var(--mr-action)]" />
                      <p className="text-[14px] font-medium">Reading facility directory...</p>
                    </div>
                  ) : extractedSpecialists[selectedHospital.name] && extractedSpecialists[selectedHospital.name].length > 0 ? (
                    <div className="grid gap-3">
                      {extractedSpecialists[selectedHospital.name].map((specialist, idx) => (
                        <div key={idx} className="p-4 rounded-xl border border-zinc-200/60 bg-zinc-50/50 hover:bg-zinc-50 hover:border-zinc-300/60 transition-colors">
                          <div className="flex items-start justify-between gap-2 mb-2">
                            <div className="flex-1 min-w-0">
                              <span className="block font-semibold text-[15px] text-zinc-900">{specialist.name}</span>
                              <span className="text-[13px] text-[var(--mr-action)] font-medium">{specialist.specialty}</span>
                              {specialist.credentials && (
                                <span className="block text-[12px] text-zinc-500 mt-0.5">{specialist.credentials}</span>
                              )}
                            </div>
                            {specialist.url && (
                              <a
                                href={specialist.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center justify-center p-1.5 rounded-md hover:bg-zinc-200/60 text-zinc-400 hover:text-zinc-700 transition-colors shrink-0"
                                aria-label="View Profile"
                              >
                                <ChevronRight className="w-4 h-4" />
                              </a>
                            )}
                          </div>
                          <p className="text-[14px] text-zinc-700 leading-relaxed mb-3">{specialist.context}</p>
                          {specialist.phone && (
                            <a href={`tel:${specialist.phone}`} className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--mr-action)] hover:text-blue-700">
                              <Phone className="w-3.5 h-3.5" /> {specialist.phone}
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex items-start gap-4 p-4 rounded-xl border border-zinc-200/60 bg-zinc-50/50">
                      <div className="flex h-8 w-8 items-center justify-center shrink-0 rounded-lg bg-white border border-zinc-200 shadow-sm text-zinc-400">
                        <Building2 className="w-4 h-4" />
                      </div>
                      <div className="flex-1">
                        <h5 className="font-semibold text-[14px] text-zinc-900 mb-1">{selectedHospital.name}</h5>
                        <p className="text-[14px] text-zinc-600 leading-relaxed mb-3">
                          We couldn't automatically find specific physician profiles for {patientCondition} at this facility.
                          However, {selectedHospital.name} is a highly capable medical center for your condition.
                        </p>
                        <div className="space-y-1.5 mt-4">
                          <p className="text-[13px] font-semibold text-zinc-900 uppercase tracking-wide">Next steps</p>
                          <ul className="text-[14px] text-zinc-600 space-y-1.5 ml-5 list-disc">
                            <li>Contact the hospital's main line to request a specialist referral</li>
                            <li>Ask specifically for the {patientCondition} department or related specialty</li>
                            <li>Request an appointment with a board-certified specialist</li>
                          </ul>
                        </div>
                        {selectedHospital.url && (
                          <a
                            href={selectedHospital.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 mt-4 text-[13px] font-medium text-[var(--mr-action)] hover:text-blue-700 transition-colors"
                          >
                            Visit Hospital Website <ChevronRight className="w-3 h-3" />
                          </a>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-none w-full lg:w-[320px] xl:w-[360px] animate-in fade-in slide-in-from-right-8 duration-500 delay-100 mb-8 lg:mb-0">
          <SurfaceCard className="rounded-2xl border-zinc-200/80 shadow-sm sticky top-0">
            <div className="flex items-center gap-2 mb-2 pb-3 border-b border-zinc-100">
              <Settings2 className="w-5 h-5 text-zinc-500" />
              <h2 className="text-[16px] font-semibold text-zinc-900">Routing Criteria</h2>
            </div>

            <div className="space-y-6 pt-2">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-[13px] font-medium text-zinc-900">Facility Capabilities</p>
                </div>
                <div className="space-y-2.5 bg-zinc-50/50 p-3.5 rounded-xl border border-zinc-100">
                  {Object.entries(equipment).map(([name, checked]) => (
                    <LabeledCheckbox
                      key={name}
                      checked={checked}
                      label={name}
                      onChange={(next) => onEquipmentToggle(name, next)}
                    />
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-[13px] font-medium text-zinc-900">Maximum Travel Time</p>
                  <span className="text-[12px] font-semibold bg-zinc-100 text-zinc-700 px-2 py-0.5 rounded-md">{maxTravelTime}h</span>
                </div>
                <div className="px-1">
                  <input
                    type="range"
                    min={0}
                    max={6}
                    step={0.5}
                    className="mr-slider"
                    value={maxTravelTime}
                    onChange={(event) => onMaxTravelTimeChange(Number(event.target.value))}
                  />
                  <div className="flex items-center justify-between text-[11px] font-medium text-zinc-400 mt-2">
                    <span>0h</span>
                    <span>3h</span>
                    <span>6h</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-[13px] font-medium text-zinc-900">Search Radius</p>
                </div>
                <select
                  className="mr-select h-10 text-[14px] leading-5 bg-zinc-50 border-zinc-200 hover:border-zinc-300 transition-colors"
                  value={maxDistance}
                  onChange={(e) => onMaxDistanceChange(e.target.value)}
                >
                  <option value="10">Within 10 miles</option>
                  <option value="25">Within 25 miles</option>
                  <option value="50">Within 50 miles</option>
                  <option value="100">Within 100 miles</option>
                  <option value="250">Within 250 miles</option>
                  <option value="500">Within 500 miles</option>
                </select>
              </div>

              <div className="pt-2">
                <button
                  className="w-full flex items-center justify-center gap-2 h-11 rounded-xl bg-zinc-900 text-white font-medium text-[14px] hover:bg-zinc-800 active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                  onClick={onUpdateSearch}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing Routes...
                    </>
                  ) : (
                    <>
                      <MapPin className="w-4 h-4" /> Update Facility Matches
                    </>
                  )}
                </button>
              </div>
            </div>
          </SurfaceCard>
        </div>
      )}
    </div>
  );
}

interface MemoScreenProps {
  selectedMatch: MatchItem | null;
  selectedHospital: RouteCenter | null;
  requiredEquipment: Record<string, boolean>;
}

function MemoScreen({ selectedMatch, selectedHospital, requiredEquipment }: MemoScreenProps) {
  const profile = useDashboardStore(s => s.profile);
  const [isCopied, setIsCopied] = useState(false);

  // Derived Patient Context
  const diagnosis = profile?.assessment?.diagnosis_primary || selectedMatch?.diagnosis || "Suspected acute condition";
  const primaryFinding = profile?.findings?.lungs?.consolidation_present === "yes" ? "consolidation" : "opacity";
  const age = profile?.patient?.age_years || "Adult";
  const gender = profile?.patient?.sex || "patient";
  const synthesis = profile?.assessment?.clinical_synthesis || "Patient requires transfer for higher level of care and specialized management.";

  // Derived Routing Context
  const targetFacilityName = selectedHospital?.name || "Specialized Care Center";
  const targetRationale = selectedHospital?.reason || "Identified high clinical concordance with outcomes from specialized care centers. Automated routing suggests facility with capable unit.";
  const travelTime = selectedHospital?.travel || "TBD";

  const activeEquipment = Object.entries(requiredEquipment)
    .filter(([_, isActive]) => isActive)
    .map(([name]) => name);

  return (
    <div className="mx-auto flex h-full min-h-0 max-w-[1200px] flex-col gap-6 lg:flex-row print:m-0 print:block print:h-auto print:max-w-none print:p-0">
      {/* Main Memo Content */}
      <div className="flex-1 w-full max-w-[800px] h-full overflow-y-auto pr-2 pb-8 custom-scrollbar print:max-w-none print:overflow-visible print:p-0">
        <SurfaceCard id="memo-content" className="gap-6 p-8 md:p-10 rounded-2xl shadow-sm border-zinc-200/80 bg-white print:border-none print:shadow-none print:p-0">
          <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-zinc-100 pb-6 gap-4 md:gap-0">
            <div className="space-y-1.5 flex flex-col items-start">
              <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[var(--mr-action)]/10 text-[var(--mr-action)] text-[12px] font-semibold tracking-wide uppercase mb-1">
                <FileText className="w-3.5 h-3.5" /> Clinical Summary Memo
              </div>
              <h1 className="text-[28px] md:text-[32px] font-semibold leading-tight tracking-tight text-zinc-900">
                Interfacility Transfer Protocol
              </h1>
              <p className="text-[14px] text-zinc-500 font-medium">Auto-generated by MedRoute AI System</p>
            </div>
            <div className="mt-4 md:mt-0 text-left md:text-right">
              <p className="text-[13px] font-semibold text-zinc-900">Date: {new Date().toLocaleDateString()}</p>
              <p className="text-[13px] text-zinc-500 mt-1">Status: <span className="text-amber-600 font-medium bg-amber-50 px-2 py-0.5 rounded-sm border border-amber-100">Draft - Requires Attending Review</span></p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pb-6 border-b border-zinc-100">
            {/* Reason for transfer */}
            <section className="space-y-2">
              <h2 className="text-[12px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5" /> Indication for Transfer
              </h2>
              <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-100 text-[14px] leading-relaxed text-zinc-800 font-medium h-[calc(100%-24px)] flex flex-col justify-center">
                Need for higher-acuity diagnostic workup and specialized management of {diagnosis.toLowerCase()}, requiring facilities beyond current institutional capabilities.
              </div>
            </section>

            {/* Case Summary */}
            <section className="space-y-2">
              <h2 className="text-[12px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <Stethoscope className="w-3.5 h-3.5" /> Clinical Presentation
              </h2>
              <div className="p-4 rounded-xl bg-zinc-50 border border-zinc-200/60 text-[14px] leading-relaxed text-zinc-700 h-[calc(100%-24px)]">
                {age}-year-old {gender} arriving for evaluation. {synthesis} See attached structured profile for full history.
              </div>
            </section>
          </div>

          {/* Matched Precedent */}
          {selectedMatch && (
            <section className="space-y-3 pb-6 border-b border-zinc-100">
              <h2 className="text-[12px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <HeartPulse className="w-3.5 h-3.5" /> Matching Case Precedent
              </h2>
              <div className="flex flex-col gap-2 relative pl-4 before:absolute before:left-0 before:top-1 before:bottom-1 before:w-[3px] before:bg-rose-400 before:rounded-full">
                <p className="text-[14px] font-semibold text-zinc-900">
                  Case Context: {selectedMatch.diagnosis} <span className="text-zinc-500 font-normal">({Math.round((selectedMatch.score || 0) * 100)}% visual match)</span>
                </p>
                <p className="text-[14px] leading-relaxed text-zinc-700">
                  {selectedMatch.outcome} This established precedent informs the current transfer protocol and required level of care.
                </p>
              </div>
            </section>
          )}

          {/* Target Facility Match */}
          <section className="space-y-3 pb-6 border-b border-zinc-100">
            <h2 className="text-[12px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
              <Building2 className="w-3.5 h-3.5" /> Target Facility Match
            </h2>
            <div className="flex flex-col gap-3 relative pl-4 before:absolute before:left-0 before:top-1 before:bottom-1 before:w-[3px] before:bg-indigo-500 before:rounded-full">
              <div className="flex items-center justify-between">
                <p className="text-[15px] font-bold text-zinc-900">{targetFacilityName}</p>
                <span className="text-[12px] font-medium text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded-md border border-indigo-100">
                  {travelTime} Travel Time
                </span>
              </div>
              <p className="text-[14px] leading-relaxed text-zinc-700">
                {highlightKeywords(targetRationale, diagnosis, requiredEquipment)}
              </p>

              {activeEquipment.length > 0 && (
                <div className="pt-2">
                  <p className="text-[12px] font-semibold uppercase tracking-wider text-zinc-500 mb-2">Required Capabilities</p>
                  <div className="flex flex-wrap gap-2">
                    {activeEquipment.map(eq => (
                      <span key={eq} className="text-[12px] font-medium text-zinc-700 bg-zinc-100 px-2 py-1 rounded-md border border-zinc-200">
                        {eq}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="space-y-3">
            <h2 className="text-[12px] font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
              <Plus className="w-3.5 h-3.5" /> Recommended Trajectory
            </h2>
            <ul className="grid sm:grid-cols-2 gap-3 mt-2">
              {[
                "Specialist consultation (e.g. Pulmonology)",
                "Advanced imaging protocol (CT/MRI)",
                "Multidisciplinary review",
                "Establishment of baseline functional status"
              ].map((step, i) => (
                <li key={i} className="flex gap-3 p-3 rounded-lg border border-zinc-100 hover:border-zinc-200 transition-colors bg-white shadow-sm">
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-zinc-100 text-[11px] font-bold text-zinc-600 border border-zinc-200">
                    {i + 1}
                  </div>
                  <span className="text-[14px] text-zinc-700 mt-0.5">{step}</span>
                </li>
              ))}
            </ul>
          </section>

        </SurfaceCard>
      </div>

      {/* Sidebar Actions */}
      <div className="w-full lg:w-[320px] xl:w-[360px] flex-shrink-0 animate-in fade-in slide-in-from-right-8 duration-500 print:hidden">
        <div className="sticky top-0 space-y-4">
          {/* Attachment Box */}
          <SurfaceCard className="rounded-xl border-zinc-200/80 shadow-sm p-5 space-y-4 bg-zinc-50/50">
            <h2 className="text-[14px] font-semibold text-zinc-900 flex items-center gap-2">
              <FolderOpen className="w-4 h-4 text-zinc-500" /> Attached Artifacts
            </h2>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white border border-zinc-200/60 shadow-sm hover:shadow-md transition-shadow cursor-pointer">
                <div className="flex items-center gap-2.5 overflow-hidden">
                  <div className="p-1.5 rounded-md bg-blue-50 text-blue-600"><FileText className="h-4 w-4" /></div>
                  <span className="text-[13px] font-medium text-zinc-700 truncate">Extracted_Profile.json</span>
                </div>
                <Check className="w-4 h-4 text-emerald-500" />
              </div>
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white border border-zinc-200/60 shadow-sm hover:shadow-md transition-shadow cursor-pointer">
                <div className="flex items-center gap-2.5 overflow-hidden">
                  <div className="p-1.5 rounded-md bg-purple-50 text-purple-600"><Scan className="h-4 w-4" /></div>
                  <span className="text-[13px] font-medium text-zinc-700 truncate">Source_Imaging_Record.cxr</span>
                </div>
                <Check className="w-4 h-4 text-emerald-500" />
              </div>
            </div>
          </SurfaceCard>

          {/* Export Actions */}
          <SurfaceCard className="rounded-xl border-zinc-200/80 shadow-sm p-5 bg-white">
            <h2 className="text-[14px] font-semibold text-zinc-900 mb-2">Export Protocol</h2>
            <div className="space-y-2.5 mt-2">
              <button
                onClick={() => window.print()}
                className="w-full flex items-center justify-center gap-2 h-10 rounded-lg bg-zinc-900 text-white font-medium text-[13px] hover:bg-zinc-800 transition-colors"
              >
                Download PDF Report
              </button>
              <button
                onClick={() => {
                  const memoText = document.getElementById('memo-content');
                  if (memoText) {
                    navigator.clipboard.writeText(memoText.innerText);
                    setIsCopied(true);
                    setTimeout(() => setIsCopied(false), 2000);
                  }
                }}
                className="w-full flex items-center justify-center gap-2 h-10 rounded-lg bg-white border border-zinc-200 text-zinc-700 font-medium text-[13px] hover:bg-zinc-50 transition-colors"
              >
                {isCopied ? "Copied!" : "Copy to Clipboard"}
              </button>
              <button
                onClick={() => alert("Translation service initiated... This feature requires backend integration.")}
                className="w-full flex items-center justify-center gap-2 h-10 rounded-lg bg-white border border-zinc-200 text-zinc-700 font-medium text-[13px] hover:bg-zinc-50 transition-colors"
              >
                Translate Document
              </button>
            </div>
          </SurfaceCard>

          {/* Disclaimer */}
          <div className="px-2 text-center">
            <p className="text-[11px] leading-relaxed text-zinc-400">
              <span className="font-semibold text-zinc-500">Clinical Decision Support Only.</span><br />
              Final determination of care trajectory remains with the reviewing physician.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function searchKey(profile: CaseProfile | null, file: File | null): string {
  const { image_url: _imageUrl, ...study } = profile?.study ?? ({} as CaseProfile["study"]);
  return JSON.stringify([profile ? { ...profile, study } : null, file ? [file.name, file.size, file.lastModified] : null]);
}

export function DashboardPage() {
  const step = useDashboardStore(s => s.step);
  const setStep = useDashboardStore(s => s.setStep);
  const selectedMatch = useDashboardStore(s => s.selectedMatch);
  const setSelectedMatch = useDashboardStore(s => s.setSelectedMatch);
  const [deIdentify, setDeIdentify] = useState(true);
  const [saveToHistory, setSaveToHistory] = useState(true);
  const [maxTravelTime, setMaxTravelTime] = useState(3);
  const [maxDistance, setMaxDistance] = useState<string>("50");
  const [language, setLanguage] = useState("English");
  const [equipment, setEquipment] = useState<Record<string, boolean>>({
    "Interventional radiology": true,
    "Robotic surgery": false,
    "Pediatric ICU": false,
    "3T MRI": true
  });
  const uploadedFile = useDashboardStore(s => s.uploadedFile);
  const setUploadedFile = useDashboardStore(s => s.setUploadedFile);
  const matchResults = useDashboardStore(s => s.matchResults);
  const searchMeta = useDashboardStore(s => s.searchMeta);
  const isSearching = useDashboardStore(s => s.isSearching);
  const searchError = useDashboardStore(s => s.searchError);
  const setSearchState = useDashboardStore(s => s.setSearchState);
  const profile = useDashboardStore(s => s.profile);
  const addTrace = useDashboardStore(s => s.addTrace);

  const [routeCenters, setRouteCenters] = useState<RouteCenter[]>([]);
  const [isRouting, setIsRouting] = useState(false);
  const [routingError, setRoutingError] = useState<string | null>(null);
  const [selectedHospital, setSelectedHospital] = useState<RouteCenter | null>(null);
  const [userCoords, setUserCoords] = useState<string | null>(null);

  const fetchRoute = async () => {
    setSelectedHospital(null); // Clear selection on new search to show filters/map
    const condition = selectedMatch !== null ? matchResults[selectedMatch].diagnosis : "complex respiratory condition";
    setIsRouting(true);
    setRoutingError(null);

    // Try to get user location
    let searchLocation = userCoords || "New York, NY";

    if (!userCoords && "geolocation" in navigator) {
      try {
        const coords = await new Promise<string>((resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            (pos) => resolve(`${pos.coords.latitude}, ${pos.coords.longitude}`),
            (err) => reject(err),
            { timeout: 30000, enableHighAccuracy: true }
          );
        });
        searchLocation = coords;
        setUserCoords(coords);
      } catch (e) {
        console.warn("Geolocation failed or timed out, using fallback NYC.", e);
        searchLocation = "New York, NY";
      }
    }

    try {
      const results = await findHospitalsRoute(
        condition,
        searchLocation,
        equipment,
        maxTravelTime,
        maxDistance
      );
      setRouteCenters(results);
    } catch (err) {
      setRoutingError(err instanceof Error ? err.message : "Routing search failed");
    } finally {
      setIsRouting(false);
    }
  };

  const runSearch = async () => {
    const { profile: profileData, uploadedFile: file } = useDashboardStore.getState();
    setSearchState({ isSearching: true, searchError: null, selectedMatch: null });
    try {
      // The image is optional: without one, the backend matches on the case report
      // (text embeddings + reranker) and MedSigLIP's text-to-image space.
      // Top 9 twins fill the 3×3 matches grid.
      const { matches, ...meta } = await searchTwins(file, profileData, "auto", 9);
      setSearchState({ matchResults: matches, searchMeta: meta, lastSearchKey: searchKey(profileData, file) });
      addTrace("Twin search", meta.trace);
    } catch (err) {
      setSearchState({ matchResults: [], searchMeta: null, searchError: err instanceof Error ? err.message : "Search failed" });
    } finally {
      setSearchState({ isSearching: false });
    }
  };

  const handleStepChange = async (next: Step) => {
    const profileData = useDashboardStore.getState().profile;
    if (
      next === 1 &&
      (!profileData || computeProfileConfidence(profileData).score < PROFILE_READY_THRESHOLD)
    ) {
      toast.error("There is no case profile ready yet.");
      return;
    }

    // When advancing to the Matches step (1), search again if the case changed
    if (next === 1 && searchKey(profileData, uploadedFile) !== useDashboardStore.getState().lastSearchKey) {
      setStep(next);
      await runSearch();    } else if (next === 2 && routeCenters.length === 0) {
      setStep(next);
      await fetchRoute();
    } else {
      setStep(next);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--mr-page)] text-[var(--mr-text)] print:h-auto print:overflow-visible print:bg-white">
      <CaseTopBar active={step} onStepChange={handleStepChange} />

      <main
        className={cn("mr-container flex min-h-0 flex-1 flex-col pb-6 pt-8", step === 0 ? "overflow-hidden" : "overflow-auto", "print:p-0 print:m-0 print:overflow-visible print:block print:h-auto")}
      >
        {step === 0 ? (
          <UploadScreen
            deIdentify={deIdentify}
            saveToHistory={saveToHistory}
            onDeIdentifyChange={setDeIdentify}
            onSaveHistoryChange={setSaveToHistory}
            onImageFilePicked={setUploadedFile}
            onStepChange={handleStepChange}
            uploadedFile={uploadedFile}
          />
        ) : null}

        {step === 1 ? (
          <>
            {searchError && (
              <div className="mb-6 rounded-2xl border border-red-200/80 bg-red-50/50 p-4 shadow-sm backdrop-blur-md max-w-2xl mx-auto">
                <div className="flex gap-3.5 items-start">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-red-100/80 text-red-600 shadow-sm border border-red-200">
                    <CloudOff className="h-4 w-4" strokeWidth={2.5} />
                  </div>
                  <div className="flex-1 mt-0.5">
                    <h3 className="text-[14px] font-semibold text-red-900 tracking-tight">Search Unavailable</h3>
                    <p className="mt-1 text-[13px] text-red-700 leading-relaxed font-medium">{searchError}</p>
                  </div>
                </div>
              </div>
            )}
            <MatchesScreen
              selectedMatch={selectedMatch}
              onSelectMatch={setSelectedMatch}
              // onContinueToRoute={() => handleStepChange(2)}
              items={matchResults}
              isLoading={isSearching}
              originalFile={uploadedFile}
              originalProfile={profile}
              searchMeta={searchMeta}
              onRefresh={runSearch}
            />
          </>
        ) : null}

        {step === 2 ? (
          <RouteScreen
            equipment={equipment}
            maxTravelTime={maxTravelTime}
            maxDistance={maxDistance}
            language={language}
            centers={routeCenters}
            isLoading={isRouting}
            error={routingError}
            selectedHospital={selectedHospital}
            patientCondition={selectedMatch !== null && matchResults[selectedMatch] ? matchResults[selectedMatch].diagnosis : "complex condition"}
            onEquipmentToggle={(key, value) =>
              setEquipment((current) => ({
                ...current,
                [key]: value
              }))
            }
            onMaxTravelTimeChange={setMaxTravelTime}
            onMaxDistanceChange={setMaxDistance}
            onLanguageChange={setLanguage}
            onHospitalClick={setSelectedHospital}
            onUpdateSearch={fetchRoute}
            onProceedToMemo={() => setStep(3)}
            userCoords={userCoords}
          />
        ) : null}

        {step === 3 ? (
          <MemoScreen
            selectedMatch={selectedMatch !== null ? matchResults[selectedMatch] : null}
            selectedHospital={selectedHospital}
            requiredEquipment={equipment}
          />
        ) : null}
      </main>

      <AiPipelinePanel />
    </div>
  );
}
