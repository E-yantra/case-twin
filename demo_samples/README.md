# Demo samples

Seven test cases from published case reports that are **not** in the twin library,
so the app has to find genuine twins rather than the case itself. Each was checked
end to end: every case returns twins with the same or a closely related diagnosis.

## Before you start

1. Open the app (`http://localhost:5173` in development, or port 8080 on the Docker deployment).
2. Open **AI pipeline** (bottom left) and check every model shows a green tick and the twin
   library shows **648** indexed case images.
3. Reload the page between cases, so each case starts fresh.

## Running a case

1. Drag the case's image into the **Clinical Copilot** chat on the right.
2. Paste the contents of its notes file into the message box and press **Enter**. After
   about 20 seconds the structured case report appears on the left.
3. Click **Matches** at the top. The "How these twins were found" panel lists each model
   step; click a twin to see what happened in that case.

The notes are written in clinician shorthand from the published case, as they would
have been at first assessment, before the diagnosis.

## The cases

| # | Image + notes | What the AI does for you | Twins found | Published diagnosis (keep for the reveal) |
|---|---|---|---|---|
| 1 | `case1_pulmonary_embolism_ct.webp` + `case1_notes.txt` | Turns a dense ICU note into a structured report and flags **RR 29, systolic BP 77 → emergent**. Flags that the notes say cardiomegaly but the image read doesn't | **5 of 5 pulmonary embolism** cases (top match 79%), with treatments and outcomes | Acute pulmonary embolism presenting with complete heart block (PMC10876825) |
| 2 | `case2_tuberculosis_chest_xray.webp` + `case2_notes.txt` | Flags **haemoptysis and weight loss**, records suspected TB, and notices that the image read disagrees with the documented cavities | **Top 2 are tuberculosis**, 3 of 5 overall | Tuberculosis presenting as immune thrombocytopenic purpura (PMC517508) |
| 3 | `case3_pneumothorax_chest_xray.webp` + `case3_notes.txt` | COVID patient with sudden chest pain. **SpO2 85% → emergent**, and a discrepancy flag: the notes suspect pneumothorax but the image read misses it | A **tension pneumothorax** twin and a COVID-19 twin in the top 5 | Bilateral pneumothoraces from COVID-related pneumatoceles (PMC7576439) |
| 4 | `case4_choroidal_metastasis_fundus.webp` + `case4_notes.txt` | Patient with known cancer and vision loss. The image read calls the fundus "relatively normal", and the system **flags the conflict** with the clinician's findings | 3 of 5 detachment or metastasis cases, incl. **metastatic choroidal melanoma** | Bilateral choroidal metastases from submandibular gland carcinoma (PMC2636054) |
| 5 | `case5_psoriasis_skin.webp` + `case5_notes.txt` | Reads the photo as skin (scaling, lichenification on the shins) and structures 20 years of history | **Top 2 are psoriasis** cases, one a paradoxical drug reaction | Psoriasis with coexisting bullous pemphigoid (PMC10460171) |
| 6 | `case6_retinal_vasculitis_fundus.webp` + `case6_notes.txt` | Child with a painful red eye and high pressure; structures a dense eye exam | Childhood retinal vascular diseases incl. **Coats' disease** | Bilateral occlusive retinal vasculitis with neovascular glaucoma (PMC11761241) |
| 7 | `case7_granulomas_skin_biopsy.webp` + `case7_notes.txt` | "?sarcoidosis vs TB": keeps both as suspected diagnoses instead of picking one | Granulomatous diseases: **orbital tuberculosis** (twice) and granulomatous uveitis | Pulmonary tuberculosis with simultaneous lung and skin sarcoidosis (PMC2822819) |

## A 15-minute session

1. **Case 1, pulmonary embolism (5 min).** The headline: a messy ICU note becomes a
   structured report with red flags and an emergent rating; five pulmonary-embolism twins
   come back, each with its treatment and outcome. Open one twin and ask the Copilot
   how it was treated.
2. **Case 3, pneumothorax (4 min).** The AI checks itself: the image read misses the
   pneumothorax, and the system flags the disagreement for the clinician instead of
   hiding it.
3. **Case 5, psoriasis (3 min).** A different kind of image (a skin photo) and the
   local-language explanation: highlight "lichenification" and click मराठी or हिंदी.
4. **AI pipeline drawer (3 min).** Walk through which model did what: Gemma 4,
   MedGemma, MedSigLIP, Qwen3 embeddings and the reranker.

Keep cases 2, 4, 6 and 7 for questions or a longer session.

## Good things to try on any case
- Highlight a term (e.g. "cavities", "RAPD", "exudative RD") and click **हिंदी** or **मराठी**: MedGemma explains it, then Gemma 4 rewrites it simply for the patient.
- **Enhance Profile** for MedGemma's differentials, risk factors and missing information.
- Open a twin and **Ask Copilot** "How was the twin treated?" or "What should I check next?"
- Open **AI pipeline** (bottom left) to show which model did each step and how long it took.

**One honest note to make during the demo:** the image reads are the weakest step.
MedGemma often misses findings on these journal figures. The system's value comes
from combining models (structured notes, image and text similarity, reranking) and
flagging disagreements for the clinician, not from any single model being perfect.
The only addition to the published facts is the clinician's "?pneumothorax" in
case 3's notes. Image reads run at temperature 0, so every case gives the same
output each time.

## Image credits

Each image is a figure from an open-access case report, distributed through the
[MultiCaRe dataset](https://github.com/mauro-nievoff/MultiCaRe_Dataset). Images are
unmodified apart from MultiCaRe's figure splitting and keep their article's licence.
CC BY-NC and CC BY-NC-SA images may only be used non-commercially; CC BY-NC-SA
copies must keep that licence.

| File | Source article | Licence |
|---|---|---|
| `case1_pulmonary_embolism_ct.webp` | Kim M, et al. *Case Report: Complete atrioventricular block in an elderly patient with acute pulmonary embolism.* Front Cardiovasc Med. 2024. [doi:10.3389/fcvm.2024.1355000](https://doi.org/10.3389/fcvm.2024.1355000) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case2_tuberculosis_chest_xray.webp` | Ozkalemkas F, et al. *Tuberculosis presenting as immune thrombocytopenic purpura.* Ann Clin Microbiol Antimicrob. 2004. [doi:10.1186/1476-0711-3-16](https://doi.org/10.1186/1476-0711-3-16) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case3_pneumothorax_chest_xray.webp` | Hameed M, et al. *Pneumothorax in Covid-19 pneumonia: a case series.* Respir Med Case Rep. 2020. [doi:10.1016/j.rmcr.2020.101265](https://doi.org/10.1016/j.rmcr.2020.101265) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case4_choroidal_metastasis_fundus.webp` | John SS, et al. *Bilateral choroidal metastasis from carcinoma of the submandibular gland.* Indian J Ophthalmol. 2008. [doi:10.4103/0301-4738.37608](https://doi.org/10.4103/0301-4738.37608) | [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/3.0/) |
| `case5_psoriasis_skin.webp` | Di Lernia V, et al. *Therapeutic management of a case of severe psoriasis coexistent with bullous pemphigoid in the elderly.* Psoriasis (Auckl). 2023. [doi:10.2147/PTT.S417427](https://doi.org/10.2147/PTT.S417427) | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
| `case6_retinal_vasculitis_fundus.webp` | Rakusiewicz-Krasnodębska K, et al. *Neovascular glaucoma as the first symptom of bilateral occlusive retinal vasculitis in a 4-year-old girl: a case report.* Biomedicines. 2025. [doi:10.3390/biomedicines13010148](https://doi.org/10.3390/biomedicines13010148) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case7_granulomas_skin_biopsy.webp` | Mise K, et al. *A rare case of pulmonary tuberculosis with simultaneous pulmonary and skin sarcoidosis: a case report.* Cases J. 2010. [doi:10.1186/1757-1626-3-24](https://doi.org/10.1186/1757-1626-3-24) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
