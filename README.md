# CaseTwin — local medical AI case-twin demo

CaseTwin takes a clinician's unstructured notes and images, builds a clean
structured case report, then finds the most similar published cases ("twins")
in an open dataset and shows what was found, done and concluded in them. It is
a teaching demo for local medical AI models, not a clinical tool.

For how it works end to end (components, models, data contracts, request
flows, frontend state, guardrails), see [ARCHITECTURE.md](ARCHITECTURE.md).

## What each model does

| Step | Model | AI concept |
|---|---|---|
| Image type of an upload | MedSigLIP `/v1/classify` | Zero-shot classification |
| Image → findings text | `medgemma` (27B) | Vision-language reading |
| Notes + image reads → case report JSON | `gemma-4` | Schema-constrained structured extraction |
| Image similarity | MedSigLIP image embedding (1152-d) | Embeddings, vector search |
| Case-report similarity | `qwen3-embedding-8b` (4096-d) | Text embeddings, semantic search |
| Notes-only search | MedSigLIP text embedding | Shared image-text space (cross-modal) |
| Reorder the top 20 | `bge-reranker-v2-m3` | Cross-encoder reranking |
| Twin Q&A, term explanations, synthesis | `medgemma` (text-only) | Retrieval-augmented generation |
| Term explanation in simple Hindi / Marathi | `medgemma` → `gemma-4` | Model chaining (medical accuracy, then language) |

Every AI endpoint returns a `trace` (model, task, time). The UI's **AI pipeline**
drawer and the **How it works** page show it, so the audience can see which
model did what.

## Try it

[`demo_samples/`](demo_samples/README.md) has ten ready-made cases (an image plus
clinician notes each), taken from published case reports that are not in the twin
library. Every case returns twins with the same or a closely related diagnosis. Its
README covers how to run a case, a suggested 15-minute session, and the published
diagnosis for each case.

## Dataset

The twin library is a deterministic sample of
[MultiCaRe](https://github.com/mauro-nievoff/MultiCaRe_Dataset) v3.0.1
(Zenodo record 20416562; 76K+ open-access case reports, 139K+ images; CC BY /
CC BY-NC / CC BY-NC-SA per article). Five collections match MedSigLIP's training
domains, defined in `backend/collections_config.py`:
chest X-ray, chest CT, dermatology photo, fundus photo and H&E histopathology.

`data_pipeline/prepare_multicare.py`:

1. Samples N articles per collection (ranked by hash, so it is repeatable).
2. Drops images whose MultiCaRe type label is wrong: MedSigLIP must rank the
   collection's own label above distractors such as charts, endoscopy and ECG.
3. Has Gemma 4 turn each patient's case text into the same case-report schema
   the live intake uses (`backend/extraction.py`), including management, outcome
   and the authors' conclusion.
4. Copies images to content-addressed `dataset/assets` and writes `dataset/manifest.json`.

Extractions and zero-shot checks are cached in `dataset/enrichment-cache`, so
re-runs and larger slices reuse finished work.

## Setup

```bash
# 1. Source data (~3 GB download, ~5.6 GB unzipped) into medical_datasets/whole_multicare_dataset
#    files: captions_and_labels.csv, metadata.parquet, abstracts.parquet, cases.parquet, PMC1..9.zip (unzip in place)
#    https://zenodo.org/records/20416562

# 2. Backend environment
cp backend/.env.example backend/.env      # fill in the gateway and MedSigLIP keys
cd backend && uv venv --python 3.12 .venv && uv pip install -r requirements.txt pytest && cd ..

# 3. Qdrant
docker run -d --name casetwin-qdrant -p 127.0.0.1:6333:6333 -v casetwin_qdrant:/qdrant/storage qdrant/qdrant

# 4. Build and index the twin library
backend/.venv/bin/python data_pipeline/prepare_multicare.py --per-collection 100 --workers 2
backend/.venv/bin/python backend/validate_multicare.py --manifest dataset/manifest.json \
    --source medical_datasets/whole_multicare_dataset
(cd backend && .venv/bin/python index_multicare.py)
```

**Gemma 4 is a shared service.** Keep `--workers` at 2 (the default) so the
batch never takes over the gateway. Each patient takes about 10–25 s of Gemma
time; 100 articles per collection is about 460 patients. `--extract-only` fills
the extraction cache without copying images.

## Run

Development:

```bash
(cd backend && .venv/bin/uvicorn main:app --port 8005)
npm install && npm run dev          # http://localhost:5173, uses VITE_API_URL from .env
```

Docker Compose (backend, frontend and its own Qdrant; stop the standalone Qdrant first):

```bash
docker compose up --build           # UI http://localhost:8080, API http://localhost:8005
```

## Deploying to a server

The twin library is two things: the Qdrant collection (vectors and case
profiles) and the `dataset/` folder (images, `manifest.json`,
`score_calibration.json`). Move both; nothing needs re-embedding.

```bash
rsync -a dataset/ <server>:<path>/case-twin/dataset/
backend/.venv/bin/python backend/copy_collection.py \
    --source http://localhost:6333 --target http://<server>:6333
```

`copy_collection.py` streams points over the Qdrant API, so it works between
Qdrant versions (a snapshot from a newer Qdrant may not restore on an older one).
On the server, point the backend at that Qdrant and folder with `QDRANT_URL`,
`COLLECTION_NAME=multicare_cases` and `DATASET_DIR`.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q
npx tsc -b && npx vite build
```
