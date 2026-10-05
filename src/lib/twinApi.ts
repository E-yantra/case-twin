import { API_BASE } from "./api";
import type { CaseProfile } from "./caseProfileTypes";

/** One model step reported by the backend, shown in the UI's AI pipeline panels. */
export interface AiTraceStep {
  model: string;
  task: string;
  ms: number;
  output?: string[];
}

export interface ChannelScores {
  image?: number;
  crossmodal?: number;
  text?: number;
  rerank?: number;
}

export interface MatchItem {
  id: string;
  score: number;
  /** Per-channel similarity mapped to 0..1 against the library's typical range. */
  scores?: ChannelScores;
  /** Raw cosine similarities / reranker logits before calibration. */
  raw_scores?: ChannelScores;
  weights?: Record<string, number>;
  collection?: string;
  modality?: string;
  diagnosis: string;
  summary: string;
  facility: string;
  outcome: string;
  outcomeVariant: "success" | "warning" | "neutral";
  conclusion?: string | null;
  treatments?: string[];
  imaging_findings?: string[];
  image_url: string;
  asset_id?: string;
  related_image_urls?: string[];
  age?: number;
  gender?: string;
  pmc_id?: string;
  article_title?: string;
  journal?: string;
  year?: string;
  license?: string;
  source_url?: string;
  radiology_view?: string;
  case_text?: string;
  raw_payload?: Record<string, any>;
}

export interface RoutingScore {
  /** null when MedSigLIP's best label is a non-library image type (chart, ECG, …). */
  collection: string | null;
  label: string;
  score: number;
}

export interface SearchResult {
  matches: MatchItem[];
  collection: string | null;
  routing: { collection: string | null; scores: RoutingScore[] } | null;
  trace: AiTraceStep[];
}

async function errorMessage(response: Response, fallback: string): Promise<string> {
  try {
    const data = await response.json();
    return data.detail || data.error || JSON.stringify(data);
  } catch {
    return (await response.text().catch(() => "")) || fallback;
  }
}

type JsonRecord = Record<string, unknown>;

const PROFILE_ARRAY_PATHS = [
  ["patient", "comorbidities"], ["patient", "medications"],
  ["assessment", "suspected_primary"], ["assessment", "differential"],
  ["assessment", "diagnosis_secondary"],
  ["findings", "lungs", "consolidation_locations"],
  ["findings", "lungs", "atelectasis_locations"],
  ["findings", "devices", "device_list"], ["findings", "other"],
  ["summary", "key_points"], ["summary", "red_flags"],
  ["presentation", "differential_diagnosis"],
  ["plan", "immediate_interventions"], ["plan", "monitoring_recommendations"],
  ["provenance", "authors"],
  ["tags", "ml_labels"], ["tags", "gt_labels"], ["tags", "keywords"], ["tags", "mesh_terms"],
] as const;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Make cached/local-model payloads safe for list rendering in the UI. */
function normalizeMatchPayload(payload: unknown): Record<string, any> | undefined {
  if (!isRecord(payload)) return undefined;
  const normalized: JsonRecord = { ...payload };

  for (const path of PROFILE_ARRAY_PATHS) {
    let current: JsonRecord = normalized;
    let completePath = true;
    for (const key of path.slice(0, -1)) {
      const next = current[key];
      if (!isRecord(next)) {
        completePath = false;
        break;
      }
      current[key] = { ...next };
      current = current[key] as JsonRecord;
    }
    if (!completePath) continue;
    const field = path[path.length - 1];
    if (field in current && current[field] != null && !Array.isArray(current[field])) {
      current[field] = [current[field]];
    }
  }

  if ("related_images" in normalized && normalized.related_images != null && !Array.isArray(normalized.related_images)) {
    normalized.related_images = [normalized.related_images];
  }
  return normalized as Record<string, any>;
}

function normalizeMatch(match: MatchItem): MatchItem {
  return { ...match, raw_payload: normalizeMatchPayload(match.raw_payload) };
}

/**
 * Hybrid twin search. Either input is optional: with an image the backend uses
 * MedSigLIP visual similarity; with a profile it adds text similarity and
 * reranking; with only a profile it searches images cross-modally.
 */
export async function searchTwins(
  file: File | null, profile: CaseProfile | null, collection = "auto", limit = 10,
): Promise<SearchResult> {
  const formData = new FormData();
  if (file) formData.append("file", file);
  if (profile) formData.append("profile", JSON.stringify(profile));
  formData.append("collection", collection);

  const response = await fetch(`${API_BASE}/search?limit=${limit}`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, `Search failed (${response.status})`));

  const data = await response.json() as SearchResult;
  // The backend normally supplies canonical arrays. This guard keeps one old
  // or malformed cached enrichment from taking down the complete results view.
  return { ...data, matches: data.matches.map(normalizeMatch) };
}

export interface ComparisonInsights {
  insights_text: string;
  original_box: [number, number, number, number] | null;
  match_box: [number, number, number, number] | null;
  trace?: AiTraceStep[];
}

export async function compareInsights(originalImage: File, matchItem: MatchItem): Promise<ComparisonInsights> {
  const formData = new FormData();
  formData.append("original_image", originalImage);
  formData.append("match_diagnosis", matchItem.diagnosis);
  const caption = matchItem.raw_payload?.study?.caption;
  if (typeof caption === "string" && caption.trim()) {
    formData.append("match_caption", caption);
  }
  if (matchItem.asset_id) formData.append("match_asset_id", matchItem.asset_id);
  if (matchItem.collection) formData.append("match_collection", matchItem.collection);
  const response = await fetch(`${API_BASE}/compare_insights`, { method: "POST", body: formData });
  if (!response.ok) throw new Error(await errorMessage(response, `Comparison failed (${response.status})`));
  return response.json() as Promise<ComparisonInsights>;
}

export interface AiStatus {
  models: Array<{ id: string; role: string; available: boolean }>;
  library: { collection: string; points: number };
}

export async function fetchAiStatus(): Promise<AiStatus> {
  const response = await fetch(`${API_BASE}/ai_status`);
  if (!response.ok) throw new Error(await errorMessage(response, `Status failed (${response.status})`));
  return response.json() as Promise<AiStatus>;
}

export interface RouteCenter {
  name: string;
  url?: string;
  capability: string;
  travel: string;
  reason: string;
  lat: number;
  lng: number;
}

export async function findHospitalsRoute(
  diagnosis: string,
  location?: string,
  equipment?: Record<string, boolean>,
  maxTravelTime?: number,
  maxDistance?: string
): Promise<RouteCenter[]> {
  const formData = new FormData();
  formData.append("diagnosis", diagnosis);
  if (location) {
    formData.append("location", location);
  }

  // Format equipment into a comma separated list of the checked items
  if (equipment) {
    const requiredEq = Object.entries(equipment)
      .filter(([_, isChecked]) => isChecked)
      .map(([name]) => name)
      .join(", ");
    if (requiredEq) {
      formData.append("equipment", requiredEq);
    }
  }

  if (maxTravelTime) {
    formData.append("maxTravelTime", maxTravelTime.toString());
  }

  if (maxDistance) {
    formData.append("maxDistance", maxDistance);
  }

  const response = await fetch(`${API_BASE}/search_hospitals`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errMessage = `Routing fell back (${response.status})`;
    try {
      const data = await response.json();
      errMessage = data.detail || data.error || JSON.stringify(data);
    } catch {
      errMessage = await response.text() || errMessage;
    }
    console.error(errMessage);
    throw new Error(errMessage);
  }

  const data = await response.json() as any;
  let centers = data.centers || data.hospitals || data;

  if (!Array.isArray(centers)) {
    if (centers && Array.isArray(centers.centers)) {
      centers = centers.centers;
    } else if (centers && Array.isArray(centers.hospitals)) {
      centers = centers.hospitals;
    } else if (Array.isArray(data)) {
      centers = data;
    } else {
      console.warn("Could not find array in route response, falling back to empty.");
      centers = [];
    }
  }

  return centers as RouteCenter[];
}
