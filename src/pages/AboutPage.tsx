import { CaseTopBar } from "@/components/CaseTopBar";
import { ModelChip } from "@/components/AiPipelinePanel";

const PIPELINE = [
  {
    step: "1. Build the case report",
    models: ["MedSigLIP", "medgemma", "gemma-4"],
    what: "MedSigLIP labels the image type with zero-shot classification. MedGemma reads each image into findings text. Gemma 4 turns the clinician's free-text notes plus those image reads into a structured case report that fits a fixed JSON schema.",
    concept: "Structured extraction with schema-constrained decoding: the model can only produce JSON with the agreed fields, so free text becomes data the rest of the system can use.",
  },
  {
    step: "2. Find the case twins",
    models: ["MedSigLIP", "qwen3-embedding-8b", "bge-reranker-v2-m3"],
    what: "The image becomes a 1152-number MedSigLIP embedding, and the case report becomes a 4096-number Qwen3 embedding. Qdrant finds the nearest published cases on each signal. A cross-encoder then rereads the top 20 report pairs and rescores them.",
    concept: "Embeddings and vector search, hybrid retrieval (image + text) and reranking. Each match shows the score from every signal, so you can see why it was chosen.",
  },
  {
    step: "3. Search without an image",
    models: ["MedSigLIP"],
    what: "With only notes, MedSigLIP's text encoder places the findings in the same space as its image vectors, so text can retrieve images.",
    concept: "A shared image-text embedding space (contrastive training, as in CLIP and SigLIP).",
  },
  {
    step: "4. Explain it to the patient, in their language",
    models: ["medgemma", "gemma-4"],
    what: "Highlight any term. MedGemma explains it for a clinician, or in plain English. For हिंदी or मराठी, Gemma 4 then rewrites MedGemma's plain explanation in simple everyday Hindi or Marathi, keeping the English term in brackets and adding no new medical facts.",
    concept: "Model chaining: a specialist medical model ensures the content is correct, then a general multilingual model adapts the language for the audience.",
  },
  {
    step: "5. Learn from the twin",
    models: ["medgemma"],
    what: "Every twin was pre-processed offline into the same schema, so its management, outcome and the authors' conclusion are ready to show. MedGemma answers follow-up questions using only the two case reports, and explains any highlighted term for a clinician or in plain language.",
    concept: "Retrieval-augmented generation: answers are grounded in retrieved evidence, not the model's memory.",
  },
];

export function AboutPage() {
  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-[var(--mr-page)]">
      <CaseTopBar active="about" />
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-8 px-6 py-10">
          <header className="space-y-3">
            <p className="text-[12px] font-semibold uppercase tracking-wider text-[var(--mr-action)]">How Case-Twin works</p>
            <h1 className="text-[28px] font-semibold tracking-tight text-zinc-900">
              From messy notes to the closest published cases, using only local medical AI models
            </h1>
            <p className="text-[15px] leading-relaxed text-zinc-600">
              Case-Twin is a teaching demo. You give it unstructured notes and images. It builds a clean case
              report, finds the most similar published case reports in an open dataset, and shows what was found,
              done and concluded in those twin cases. All models run on local hardware.
            </p>
          </header>

          <ol className="space-y-4">
            {PIPELINE.map((item) => (
              <li key={item.step} className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="mr-2 text-[17px] font-semibold text-zinc-900">{item.step}</h2>
                  {item.models.map((m) => <ModelChip key={m} model={m} />)}
                </div>
                <p className="text-[14px] leading-relaxed text-zinc-700">{item.what}</p>
                <p className="rounded-lg bg-zinc-50 px-4 py-2.5 text-[13px] leading-relaxed text-zinc-600">
                  <span className="font-semibold text-zinc-800">AI concept: </span>{item.concept}
                </p>
              </li>
            ))}
          </ol>

          <section className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm space-y-2 text-[14px] leading-relaxed text-zinc-700">
            <h2 className="text-[17px] font-semibold text-zinc-900">The twin library</h2>
            <p>
              Cases come from <a className="text-blue-600 hover:underline" href="https://github.com/mauro-nievoff/MultiCaRe_Dataset" target="_blank" rel="noreferrer">MultiCaRe</a>,
              an open dataset of de-identified case reports from PubMed Central (76K+ articles, 139K+ images). The demo
              uses fixed, repeatable samples from five collections that match MedSigLIP's training domains: chest X-ray,
              chest CT, dermatology photos, fundus photos and H&amp;E histopathology. Each image keeps its article's
              licence (CC BY, CC BY-NC or CC BY-NC-SA) and a link to the source.
            </p>
            <p className="text-zinc-500">
              Not for clinical use. AI outputs can be wrong; every output links back to its source so it can be checked.
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
