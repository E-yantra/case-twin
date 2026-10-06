# Demo samples

Seven test cases from published case reports that are **not** in the twin library,
so the app has to find genuine twins rather than the case itself. Each was checked
end to end: every case returns twins with the same or a closely related diagnosis.

**How to run one:** open the app, drop the image into the Clinical Copilot chat, paste
the matching `caseN_notes.txt`, press Enter, then click **Matches**. The notes are
written in clinician shorthand from the published case, as they would have been at
first assessment, before the diagnosis.

| # | Upload | What the AI does for you | Twins found | Published diagnosis (keep for the reveal) |
|---|---|---|---|---|
| 1 | `case1_pulmonary_embolism_ct.webp` | Turns a dense ICU note into a structured report and flags **RR 29, systolic BP 77 → emergent**. Flags that the notes say cardiomegaly but the image read doesn't | **5 of 5 pulmonary embolism** cases (top match 79%), with treatments and outcomes | Acute pulmonary embolism presenting with complete heart block (PMC10876825) |
| 2 | `case2_tuberculosis_chest_xray.webp` | Flags **haemoptysis and weight loss**, records suspected TB, and notices that the image read disagrees with the documented cavities | **Top 2 are tuberculosis**, 3 of 5 overall | Tuberculosis presenting as immune thrombocytopenic purpura (PMC517508) |
| 3 | `case3_pneumothorax_chest_xray.webp` | COVID patient with sudden chest pain. **SpO2 85% → emergent**, and a discrepancy flag: the notes suspect pneumothorax but the image read misses it | A **tension pneumothorax** twin and a COVID-19 twin in the top 5 | Bilateral pneumothoraces from COVID-related pneumatoceles (PMC7576439) |
| 4 | `case4_choroidal_metastasis_fundus.webp` | Patient with known cancer and vision loss. The image read calls the fundus "relatively normal", and the system **flags the conflict** with the clinician's findings | 3 of 5 detachment or metastasis cases, incl. **metastatic choroidal melanoma** | Bilateral choroidal metastases from submandibular gland carcinoma (PMC2636054) |
| 5 | `case5_psoriasis_skin.webp` | Reads the photo as skin (scaling, lichenification on the shins) and structures 20 years of history | **Top 2 are psoriasis** cases, one a paradoxical drug reaction | Psoriasis with coexisting bullous pemphigoid (PMC10460171) |
| 6 | `case6_retinal_vasculitis_fundus.webp` | Child with a painful red eye and high pressure; structures a dense eye exam | Childhood retinal vascular diseases incl. **Coats' disease** | Bilateral occlusive retinal vasculitis with neovascular glaucoma (PMC11761241) |
| 7 | `case7_granulomas_skin_biopsy.webp` | "?sarcoidosis vs TB": keeps both as suspected diagnoses instead of picking one | Granulomatous diseases: **orbital tuberculosis** (twice) and granulomatous uveitis | Pulmonary tuberculosis with simultaneous lung and skin sarcoidosis (PMC2822819) |

**Good things to try on any case**
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
