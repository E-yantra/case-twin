import { create } from 'zustand';
import { type CaseProfile } from '@/lib/caseProfileTypes';
import type { Step } from '@/components/CaseTopBar';
import { type OrchestratorState, createInitialState } from '@/lib/agenticOrchestrator';
import type { AiTraceStep, MatchItem, SearchResult } from '@/lib/twinApi';

/** One user-visible AI action (e.g. "Case report", "Twin search") and the model steps behind it. */
export interface AiTraceEntry {
    id: string;
    stage: string;
    at: number;
    steps: AiTraceStep[];
}

export interface Specialist {
    name: string;
    specialty: string;
    credentials?: string;
    context: string;
    url?: string;
    phone?: string;
}

interface DashboardStore {
    // Left panel state
    profile: CaseProfile | null;
    setProfile: (profile: CaseProfile | null) => void;

    // Right panel (Copilot) state
    orchestratorState: OrchestratorState;
    setOrchestratorState: (state: OrchestratorState | ((prev: OrchestratorState) => OrchestratorState)) => void;

    // Route page specialists state (persists across tab switches)
    extractedSpecialists: Record<string, Specialist[]>;
    setExtractedSpecialists: (hospitalName: string, specialists: Specialist[]) => void;

    // Wizard step (Upload / Matches), so returning from Chat or About lands where you were
    step: Step;
    setStep: (step: Step) => void;

    // Case image and twin search: kept here (not in page state) so switching
    // tabs or pages never drops the image or silently changes the twins.
    uploadedFile: File | null;
    setUploadedFile: (file: File | null) => void;
    matchResults: MatchItem[];
    searchMeta: Omit<SearchResult, "matches"> | null;
    selectedMatch: number | null;
    setSelectedMatch: (index: number | null) => void;
    isSearching: boolean;
    searchError: string | null;
    /** What the current results were computed from; a change means they are stale. */
    lastSearchKey: string | null;
    setSearchState: (patch: Partial<Pick<DashboardStore, "matchResults" | "searchMeta" | "selectedMatch" | "isSearching" | "searchError" | "lastSearchKey">>) => void;

    // MedGemma "Enhance Profile" output for the current case
    enhancedSynthesis: string | null;
    enhancedImaging: string | null;
    setEnhanced: (synthesis: string | null, imaging: string | null) => void;

    // AI pipeline log: every model call the session made, for the demo panel
    aiTrace: AiTraceEntry[];
    addTrace: (stage: string, steps: AiTraceStep[]) => void;

    // Reset function
    resetStore: () => void;
}

export const useDashboardStore = create<DashboardStore>((set) => ({
    profile: null,
    setProfile: (profile) => set({ profile }),

    orchestratorState: createInitialState(),
    setOrchestratorState: (state) => set((prev) => ({
        orchestratorState: typeof state === 'function' ? state(prev.orchestratorState) : state
    })),

    extractedSpecialists: {},
    setExtractedSpecialists: (hospitalName, specialists) => set((prev) => ({
        extractedSpecialists: {
            ...prev.extractedSpecialists,
            [hospitalName]: specialists
        }
    })),

    step: 0,
    setStep: (step) => set({ step }),
    uploadedFile: null,
    setUploadedFile: (uploadedFile) => set({ uploadedFile }),
    matchResults: [],
    searchMeta: null,
    selectedMatch: null,
    setSelectedMatch: (selectedMatch) => set({ selectedMatch }),
    isSearching: false,
    searchError: null,
    lastSearchKey: null,
    setSearchState: (patch) => set(patch),

    enhancedSynthesis: null,
    enhancedImaging: null,
    setEnhanced: (enhancedSynthesis, enhancedImaging) => set({ enhancedSynthesis, enhancedImaging }),

    aiTrace: [],
    addTrace: (stage, steps) => set((prev) => ({
        aiTrace: [...prev.aiTrace, { id: `${Date.now()}-${prev.aiTrace.length}`, stage, at: Date.now(), steps }].slice(-40)
    })),

    resetStore: () => set({
        profile: null,
        step: 0,
        uploadedFile: null,
        matchResults: [],
        searchMeta: null,
        selectedMatch: null,
        isSearching: false,
        searchError: null,
        lastSearchKey: null,
        enhancedSynthesis: null,
        enhancedImaging: null,
        aiTrace: [],
        orchestratorState: createInitialState(),
        extractedSpecialists: {}
    })
}));
