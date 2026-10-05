import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Cpu, X, CircleCheck, CircleAlert } from "lucide-react";
import { useDashboardStore } from "@/store/dashboardStore";
import { fetchAiStatus, type AiStatus } from "@/lib/twinApi";
import { cn } from "@/lib/utils";

/** Colour per model family so the same model is recognisable across steps. */
function modelTone(model: string): string {
    const m = model.toLowerCase();
    if (m.startsWith("gemma")) return "bg-violet-50 text-violet-700 border-violet-200";
    if (m.includes("medgemma-1.5")) return "bg-sky-50 text-sky-700 border-sky-200";
    if (m.includes("medgemma")) return "bg-blue-50 text-blue-700 border-blue-200";
    if (m.includes("siglip")) return "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (m.includes("qwen") || m.includes("rerank")) return "bg-amber-50 text-amber-700 border-amber-200";
    return "bg-zinc-50 text-zinc-600 border-zinc-200";
}

export function ModelChip({ model }: { model: string }) {
    return (
        <span className={cn("inline-flex shrink-0 items-center rounded-md border px-1.5 py-0.5 text-[11px] font-semibold", modelTone(model))}>
            {model}
        </span>
    );
}

/**
 * Floating "AI pipeline" drawer: which local model did what, in order, and how
 * long it took. This is the demo's window into the system — every number shown
 * comes from the backend's trace, not from the UI.
 */
export function AiPipelinePanel() {
    const trace = useDashboardStore((s) => s.aiTrace);
    const [open, setOpen] = useState(false);
    const [status, setStatus] = useState<AiStatus | null>(null);
    const [statusError, setStatusError] = useState<string | null>(null);

    useEffect(() => {
        if (!open) return;
        fetchAiStatus().then(setStatus).catch((e) => setStatusError(e instanceof Error ? e.message : "Unavailable"));
    }, [open]);

    return (
        <>
            <button
                type="button"
                onClick={() => setOpen(true)}
                className="fixed bottom-5 left-5 z-40 flex items-center gap-2 rounded-full border border-zinc-200 bg-white px-4 py-2.5 text-[13px] font-semibold text-zinc-800 shadow-lg hover:bg-zinc-50"
            >
                <Cpu className="h-4 w-4 text-[var(--mr-action)]" />
                AI pipeline
                {trace.length > 0 && (
                    <span className="rounded-full bg-[var(--mr-action)] px-1.5 text-[11px] text-white">{trace.length}</span>
                )}
            </button>

            {open && (
                <div className="fixed inset-0 z-50 flex justify-start bg-black/20" onClick={() => setOpen(false)}>
                    <aside
                        className="flex h-full w-full max-w-[460px] flex-col bg-white shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between border-b border-zinc-100 px-5 py-4">
                            <div>
                                <h2 className="text-[16px] font-semibold text-zinc-900">AI pipeline</h2>
                                <p className="text-[12px] text-zinc-500">Every model call this session made, all on local hardware.</p>
                            </div>
                            <button onClick={() => setOpen(false)} className="rounded-full p-1.5 hover:bg-zinc-100" aria-label="Close">
                                <X className="h-4 w-4" />
                            </button>
                        </div>

                        <div className="flex-1 space-y-6 overflow-y-auto px-5 py-4">
                            <section>
                                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">Models</h3>
                                {statusError && <p className="text-[12px] text-red-600">Status unavailable: {statusError}</p>}
                                {!status && !statusError && <p className="text-[12px] text-zinc-400">Checking…</p>}
                                {status && (
                                    <ul className="space-y-2">
                                        {status.models.map((m) => (
                                            <li key={m.id} className="flex items-start gap-2 text-[12px]">
                                                {m.available
                                                    ? <CircleCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
                                                    : <CircleAlert className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />}
                                                <ModelChip model={m.id} />
                                                <span className="text-zinc-600">{m.role}</span>
                                            </li>
                                        ))}
                                        <li className="pt-1 text-[12px] text-zinc-500">
                                            Twin library: <span className="font-semibold text-zinc-800">{status.library.points}</span> indexed case images
                                        </li>
                                    </ul>
                                )}
                            </section>

                            <section>
                                <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">This session</h3>
                                {trace.length === 0 ? (
                                    <p className="text-[12px] text-zinc-400">No AI calls yet. Add notes or an image to the copilot to start.</p>
                                ) : (
                                    <ol className="space-y-4">
                                        {[...trace].reverse().map((entry) => (
                                            <li key={entry.id} className="rounded-xl border border-zinc-200 p-3">
                                                <div className="mb-2 flex items-center justify-between">
                                                    <span className="text-[13px] font-semibold text-zinc-900">{entry.stage}</span>
                                                    <span className="text-[11px] text-zinc-400">{new Date(entry.at).toLocaleTimeString()}</span>
                                                </div>
                                                <ol className="space-y-1.5">
                                                    {entry.steps.map((step, i) => (
                                                        <li key={i} className="text-[12px]">
                                                            <div className="flex items-start gap-2">
                                                                <span className="mt-0.5 w-4 shrink-0 text-right text-zinc-400">{i + 1}.</span>
                                                                <ModelChip model={step.model} />
                                                                <span className="flex-1 text-zinc-700">{step.task}</span>
                                                                {step.ms > 0 && <span className="shrink-0 tabular-nums text-zinc-400">{(step.ms / 1000).toFixed(1)}s</span>}
                                                            </div>
                                                            {step.output && step.output.length > 0 && (
                                                                <div className="ml-6 mt-1 space-y-1 rounded-md bg-zinc-50 p-2 text-[11px] leading-relaxed text-zinc-600">
                                                                    {step.output.map((o, j) => <ReactMarkdown key={j}>{o}</ReactMarkdown>)}
                                                                </div>
                                                            )}
                                                        </li>
                                                    ))}
                                                </ol>
                                            </li>
                                        ))}
                                    </ol>
                                )}
                            </section>
                        </div>
                    </aside>
                </div>
            )}
        </>
    );
}
