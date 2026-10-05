# Demo samples

Five test cases from MultiCaRe articles that are **not** in the twin library, so the app
has to find genuine twins rather than the case itself. Notes are written in a busy
clinician's shorthand from the published case. In each one, drop the image into the
Copilot chat, paste the notes, press Enter, then open **Matches**.

| Case | Upload + notes | What to show | Published outcome (keep for the reveal) |
|---|---|---|---|
| 3 | `case3_chest_xray.webp` + `case3_notes.txt` | COVID patient with sudden chest pain. A **tension pneumothorax** twin appears in the top 3. Red flags show SpO2 85% and a **discrepancy**: the notes suspect pneumothorax, but MedGemma's image read misses it. AI cross-checking itself | Bilateral pneumothoraces from COVID-related pneumatoceles (PMC7576439) |
| 4 | `case4_fundus.webp` + `case4_notes.txt` | Child with a painful red eye. Twins are childhood retinal vascular diseases, including **Coats' disease**, a cause of exudative retinal detachment | Exudative retinal detachment with neovascular glaucoma (PMC11761241) |
| 5 | `case5_skin_legs.webp` + `case5_notes.txt` | Long-standing psoriasis that is now worsening. The top 2 twins are **psoriasis** cases (one is a paradoxical drug reaction) | Bullous pemphigoid alongside psoriasis, both controlled on dimethyl fumarate (PMC10460171) |
| 1 | `case1_chest_xray.webp` + `case1_notes.txt` | Day-1 post-op breathlessness. The case report flags an **acute haemoglobin drop**. Twins match the large effusion, not the cause: AI finds look-alikes, and the clinician weighs the context | Haemothorax from a misplaced pedicle screw with diaphragmatic injury (PMC4012096) |
| 2 | `case2_skin_arms.webp` + `case2_notes.txt` | Painful red nodules on the arms. Twins are nodular skin infections (mycobacterial infection, leprosy), genuine look-alikes for a differential discussion | Erythema nodosum revealing sarcoidosis (PMC4909375) |

Cases 3–5 have twins that share the diagnosis; cases 1–2 show the limits of a
648-case library. Only one thing was added to the published facts: the
clinician's "?pneumothorax" in case 3's notes.

MedGemma's image reads run at temperature 0, so each case gives the same output
every time. Images keep their source article's licence; see each PMC article.
