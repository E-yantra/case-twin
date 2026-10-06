# Demo samples

Five test cases from MultiCaRe articles that are **not** in the twin library, so the app
has to find genuine twins rather than the case itself. Notes are written in a busy
clinician's shorthand from the published case. In each one, drop the image into the
Copilot chat, paste the notes, press Enter, then open **Matches**.

| Case | Upload + notes | What to show | Published outcome (keep for the reveal) |
|---|---|---|---|
| 3 | `case3_chest_xray.webp` + `case3_notes.txt` | COVID patient with sudden chest pain. A **tension pneumothorax** twin appears in the top 3. Red flags show SpO2 85% and a **discrepancy**: the notes suspect pneumothorax, but MedGemma's image read misses it. AI cross-checking itself | Bilateral pneumothoraces from COVID-related pneumatoceles (PMC7576439) |
| 4 | `case4_fundus.webp` + `case4_notes.txt` | Child with a painful red eye. Twins are childhood retinal vascular diseases, including **Coats' disease**, a cause of exudative retinal detachment | Bilateral occlusive retinal vasculitis presenting as neovascular glaucoma, with exudative retinal detachment (PMC11761241) |
| 5 | `case5_skin_legs.webp` + `case5_notes.txt` | Long-standing psoriasis that is now worsening. The top 2 twins are **psoriasis** cases (one is a paradoxical drug reaction) | Bullous pemphigoid alongside psoriasis, both controlled on dimethyl fumarate (PMC10460171) |
| 1 | `case1_chest_xray.webp` + `case1_notes.txt` | Day-1 post-op breathlessness. The case report flags an **acute haemoglobin drop**. Twins match the large effusion, not the cause: AI finds look-alikes, and the clinician weighs the context | Haemothorax from a misplaced pedicle screw with diaphragmatic injury (PMC4012096) |
| 2 | `case2_skin_arms.webp` + `case2_notes.txt` | Painful red nodules on the arms. Twins are nodular skin infections (mycobacterial infection, leprosy), genuine look-alikes for a differential discussion | Erythema nodosum revealing sarcoidosis (PMC4909375) |

Cases 3–5 have twins that share the diagnosis; cases 1–2 show the limits of a
648-case library. Only one thing was added to the published facts: the
clinician's "?pneumothorax" in case 3's notes.

MedGemma's image reads run at temperature 0, so each case gives the same output
every time.

## Image credits

Each image is a figure from an open-access case report, distributed through the
[MultiCaRe dataset](https://github.com/mauro-nievoff/MultiCaRe_Dataset). Images are
unmodified apart from MultiCaRe's figure splitting and keep their article's licence.
CC BY-NC images may only be used non-commercially.

| File | Source article | Licence |
|---|---|---|
| `case1_chest_xray.webp` | Bini R, et al. *Repair of diaphragmatic hernia following spinal surgery by laparoscopic mesh application: a case report and review of the literature.* World J Emerg Surg. 2014. [doi:10.1186/1749-7922-9-34](https://doi.org/10.1186/1749-7922-9-34) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case2_skin_arms.webp` | Diernaes JEF, et al. *Unmasking sarcoidosis following surgery for Cushing disease.* Dermatoendocrinol. 2016. [doi:10.4161/derm.29855](https://doi.org/10.4161/derm.29855) | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
| `case3_chest_xray.webp` | Hameed M, et al. *Pneumothorax in Covid-19 pneumonia: a case series.* Respir Med Case Rep. 2020. [doi:10.1016/j.rmcr.2020.101265](https://doi.org/10.1016/j.rmcr.2020.101265) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case4_fundus.webp` | Rakusiewicz-Krasnodębska K, et al. *Neovascular glaucoma as the first symptom of bilateral occlusive retinal vasculitis in a 4-year-old girl: a case report.* Biomedicines. 2025. [doi:10.3390/biomedicines13010148](https://doi.org/10.3390/biomedicines13010148) | [CC BY](https://creativecommons.org/licenses/by/4.0/) |
| `case5_skin_legs.webp` | Di Lernia V, et al. *Therapeutic management of a case of severe psoriasis coexistent with bullous pemphigoid in the elderly.* Psoriasis (Auckl). 2023. [doi:10.2147/PTT.S417427](https://doi.org/10.2147/PTT.S417427) | [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) |
