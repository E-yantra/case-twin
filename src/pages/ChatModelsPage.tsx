import { useState } from "react";
import { RefreshCw } from "lucide-react";
import { CaseTopBar } from "@/components/CaseTopBar";

const OPENWEBUI_URL = import.meta.env.VITE_OPENWEBUI_URL as string | undefined;

export function ChatModelsPage() {
  // Changing the key re-mounts the iframe, which reloads Open WebUI only
  // (a cross-origin frame cannot be reloaded from here any other way).
  const [frameKey, setFrameKey] = useState(0);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--mr-page)] text-[var(--mr-text)]">
      <CaseTopBar
        active="chat"
        actions={OPENWEBUI_URL ? (
          <button
            type="button"
            onClick={() => setFrameKey((key) => key + 1)}
            aria-label="Refresh chat"
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-zinc-200 bg-white px-2.5 py-1.5 text-[13px] font-medium text-zinc-700 hover:bg-zinc-50 hover:text-zinc-900 sm:px-3"
            title="Reload the chat without reloading the app"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Refresh chat</span>
          </button>
        ) : null}
      />

      <main className="mr-container flex min-h-0 flex-1 flex-col overflow-hidden pb-6 pt-6">
        {OPENWEBUI_URL ? (
          <iframe
            key={frameKey}
            src={OPENWEBUI_URL}
            title="Chat"
            className="min-h-0 w-full flex-1 rounded-2xl border border-zinc-200/80"
            allow="clipboard-write; microphone"
          />
        ) : (
          <div className="flex min-h-0 flex-1 items-center justify-center text-sm text-zinc-500">
            Set VITE_OPENWEBUI_URL to enable the Chat tab.
          </div>
        )}
      </main>
    </div>
  );
}
