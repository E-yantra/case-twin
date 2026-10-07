import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Check, Info, MessageSquare, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";

export type Step = 0 | 1 | 2 | 3;

const stepLabels = ["Upload", "Matches" /*, "Route", "Memo" */] as const;

interface CaseTopBarProps {
  /** Which pill is highlighted. "chat"/"about" on those pages, otherwise the current wizard step. */
  active: "chat" | "about" | Step;
  /** Provided by the dashboard so Upload/Matches switch step in place instead of navigating. */
  onStepChange?: (next: Step) => void;
  /** Page-specific controls shown on the right of the bar (e.g. the Chat tab's refresh button). */
  actions?: ReactNode;
}

export function CaseTopBar({ active, onStepChange, actions }: CaseTopBarProps) {
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
              const state = typeof active !== "number" ? "default" : idx < active ? "done" : idx === active ? "active" : "default";

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

        <div className="flex items-center gap-3">
          {actions}
          <Link
            to="/about"
            className={cn(
              "hidden lg:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[13px] font-medium transition-colors",
              active === "about" ? "bg-zinc-100 text-zinc-900" : "text-zinc-500 hover:text-zinc-900 hover:bg-zinc-50"
            )}
          >
            <Info className="w-3.5 h-3.5" /> How it works
          </Link>
          {typeof active === "number" && (
            <button
              type="button"
              onClick={() => window.location.reload()}
              aria-label="Refresh Case-Twin"
              title="Refresh Case-Twin"
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-900"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
