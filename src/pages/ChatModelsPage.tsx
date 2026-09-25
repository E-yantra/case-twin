import { CaseTopBar } from "@/components/CaseTopBar";

const OPENWEBUI_URL = import.meta.env.VITE_OPENWEBUI_URL as string | undefined;

export function ChatModelsPage() {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--mr-page)] text-[var(--mr-text)]">
      <CaseTopBar active="chat" />

      <main className="mr-container flex min-h-0 flex-1 flex-col overflow-hidden pb-6 pt-8">
        {OPENWEBUI_URL ? (
          <iframe
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
