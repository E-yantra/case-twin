import { CaseTopBar } from "@/components/CaseTopBar";

const OPENWEBUI_URL = import.meta.env.VITE_OPENWEBUI_URL as string | undefined;

export function ChatModelsPage() {
  return (
    <div className="h-screen overflow-hidden bg-[var(--mr-page)] text-[var(--mr-text)]">
      <CaseTopBar active="chat" />

      <main className="mr-container h-full overflow-hidden pb-6 pt-24">
        {OPENWEBUI_URL ? (
          <iframe
            src={OPENWEBUI_URL}
            title="Chat"
            className="h-full w-full rounded-2xl border border-zinc-200/80"
            allow="clipboard-write; microphone"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-zinc-500">
            Set VITE_OPENWEBUI_URL to enable the Chat tab.
          </div>
        )}
      </main>
    </div>
  );
}
