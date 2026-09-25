import { Link } from "react-router-dom";
import { Check, FolderOpen, MessageSquare, Plus, Settings2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type Step = 0 | 1 | 2 | 3;

const stepLabels = ["Upload", "Matches" /*, "Route", "Memo" */] as const;

interface CaseTopBarProps {
  /** Which pill is highlighted. "chat" when on the Chat tab, otherwise the current wizard step. */
  active: "chat" | Step;
  /** Provided by the dashboard so Upload/Matches switch step in place instead of navigating. */
  onStepChange?: (next: Step) => void;
}

export function CaseTopBar({ active, onStepChange }: CaseTopBarProps) {
  return (
    <header className="relative z-40 shrink-0 border-b border-zinc-200/80 bg-white/80 shadow-[0_1px_3px_rgba(0,0,0,0.02)] backdrop-blur-xl supports-[backdrop-filter]:bg-white/60 print:hidden">
      <div className="mr-container flex h-16 items-center justify-between gap-4 py-3">
        <div className="flex items-center gap-2.5 cursor-pointer hover:opacity-90 transition-opacity">
          <div className="flex h-8 w-8 items-center justify-center rounded-[0.4rem] bg-gradient-to-tr from-zinc-900 to-zinc-800 text-white shadow-[0_1px_3px_rgba(0,0,0,0.1)] ring-1 ring-zinc-900/10 transition-transform duration-300 hover:scale-[1.03]">
            <span className="text-[13px] font-bold tracking-wider">CT</span>
          </div>
          <span className="text-[16px] font-semibold tracking-tight text-zinc-900">Case-Twin</span>
        </div>

        <div className="flex-1 flex justify-center">
          <ol className="flex flex-wrap items-center justify-center gap-2">
            <li>
              <Link
                to="/chat"
                aria-current={active === "chat" ? "step" : undefined}
                className={cn(
                  "inline-flex h-8 items-center gap-1.5 rounded-full px-4 text-xs leading-4 transition-colors",
                  active === "chat"
                    ? "bg-[var(--mr-action)] font-semibold text-[var(--mr-on-action)]"
                    : "bg-transparent text-[var(--mr-text-secondary)] hover:bg-[var(--mr-bg-subtle)] hover:text-[var(--mr-text)]"
                )}
              >
                <MessageSquare className="h-3 w-3" />
                <span>Chat</span>
              </Link>
            </li>
            {stepLabels.map((label, idx) => {
              const step = idx as Step;
              const state = active === "chat" ? "default" : idx < active ? "done" : idx === active ? "active" : "default";

              return (
                <li key={label}>
                  <Link
                    to="/"
                    onClick={(event) => {
                      if (onStepChange) {
                        event.preventDefault();
                        onStepChange(step);
                      }
                    }}
                    aria-current={state === "active" ? "step" : undefined}
                    className={cn(
                      "inline-flex h-8 items-center gap-1.5 rounded-full px-4 text-xs leading-4 transition-colors",
                      state === "default" &&
                      "bg-transparent text-[var(--mr-text-secondary)] hover:bg-[var(--mr-bg-subtle)] hover:text-[var(--mr-text)]",
                      state === "active" && "bg-[var(--mr-action)] font-semibold text-[var(--mr-on-action)]",
                      state === "done" && "bg-[var(--mr-bg-subtle)] text-[var(--mr-text)]"
                    )}
                  >
                    {state === "done" ? <Check className="h-3 w-3" /> : null}
                    <span>{label}</span>
                  </Link>
                </li>
              );
            })}
          </ol>
        </div>

        <div className="hidden lg:flex items-center gap-4">
          {/* <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] font-medium transition-all duration-200 text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50">
            <FolderOpen className="w-3.5 h-3.5" strokeWidth={2.5} /> My Cases
          </button>

          <button className="flex items-center gap-1.5 rounded-full bg-zinc-900 px-4 py-1.5 text-[13px] font-medium text-white shadow-md shadow-zinc-900/10 hover:bg-zinc-800 transition-all active:scale-[0.98]">
            <Plus className="h-4 w-4" strokeWidth={2.5} />
            New Case
          </button>

          <div className="w-px h-4 bg-zinc-200" />

          <button aria-label="Settings" className="flex h-8 w-8 items-center justify-center rounded-full text-zinc-400 hover:bg-zinc-100 hover:text-zinc-600 transition-colors">
            <Settings2 className="h-4 w-4" />
          </button> */}
        </div>
      </div>
    </header>
  );
}
