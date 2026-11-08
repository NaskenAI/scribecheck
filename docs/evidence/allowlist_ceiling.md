# Allow-list ceiling for rules-mode certain flags

Pass B's rules-mode certain flags were disabled after two adversarial reviews showed that a
deny-list of blocking cues cannot rule out false certain flags in open clinical language
(commit fd5ad82). The proposed replacement was an allow-list: a certain flag only when both
the transcript mention and the note mention match a small set of plain sentence shapes.
Before building it, the shapes and their ceilings were measured on the OmissionBench
development data (65 consultations, 65 clean notes, 265 evaluation pairs) on 2026-11-08.

**Result: the ceiling is zero.** On this data no pair has both mentions in an allowed shape,
so an allow-list would give no certain flags. The transcripts are conversation, not
statements.

## Method

- **Matching a shape.** A sentence (or note line) matches a shape only when every word is
  accounted for by a finder match (drug, dose, frequency, route, symptom, side, allergy) or
  by the shape's own fixed words, plus end punctuation and a list bullet.
- **Testing the matcher.** It was checked first on 28 synthetic sentences with known
  answers, negatives included, so a zero count means absent rather than a broken matcher.
- **Categories.** A removed or changed fact's category comes from running the existing
  finders over the pair's own `fact` text. The development copy has no fact sheets, and
  `target.mode` names no category except four change pairs marked `dose_value`.
- **Privacy.** The scripts read development data only and printed counts only.

## Ceilings

| Ceiling | Structured fact involved | Removed item in the transcript | Both mentions in an allowed shape |
|---|---|---|---|
| Changed facts (117 add-or-change pairs) | 45 (38.5%) | — | **0 (0%)** |
| Omissions (148 omission pairs) | 15 (10.1%) | 15 (10.1%) | **0 (0%)** |

- **Changed facts:** "structured fact involved" means the edit changes a drug, side or
  symptom item between the clean and errored note.
- **Omissions:** these are medicines (13), frequencies (2), dose (1), allergy (1) and side
  (1); a fact can be more than one. All 15 removed items appear in the transcript as the
  patient's own current mention; none in an allowed shape.

## Shape counts (65 transcripts, 65 clean notes)

| Shape | Mentions | Docs with one |
|---|---|---|
| T-D1 patient: "I take / I'm on / I'm taking *drug* [*dose*] [*freq*]" | 0 | 0 |
| T-D2 clinician read-back: "You're on *drug* *dose* *freq*?" + bare Yes, No or a correction | 0 | 0 |
| T-S1 patient: "I have / I've got / I get *symptom*" | 0 | 0 |
| T-S2 patient: "No *symptom*" / "I don't have *symptom*" | 6 | 4 |
| T-S3 dialogue: "Any *symptom*?" / "Do you have *symptom*?" + bare Yes or No | 2 | 2 |
| T-Si1 patient: "My *side* *part* hurts / is swollen" | 0 | 0 |
| T-A1 patient: "I'm allergic to *allergen*" | 1 | 1 |
| N-D1 note line: "*drug* *dose* [*route*] [*freq*]" | 8 | 2 |
| N-S1 note: "No *symptom*." / "Denies *symptom*." | 19 | 15 |
| N-S2 note: "*Symptom* for N days." / "Reports *symptom*." | 0 | 0 |
| N-Si1 note: "*Side* *part* pain / swelling / tenderness" | 3 | 2 |
| N-A1 note: "Allergies: *allergen*" / "Allergic to *allergen*" | 3 | 3 |

## Transcript analysis

| Speaker | Sentences | Naming a drug | Naming a drug with a dose | Naming a symptom |
|---|---|---|---|---|
| Patient | 3,917 | 74 | **1** | 237 (139 of them with 10 or more other words) |
| Clinician | 5,430 | 144 | 36 (only 4 with 5 or fewer other words) | — |

- **Notes are plain; transcripts aren't.** Notes do contain plain shapes, mostly "No
  *symptom*", but the transcript side of the same fact is almost never said plainly.
- **The read-back shape never occurs.**
- **Loosening won't help safely.** Allowing fillers or extra words would bring back the
  open-language problem the allow-list was meant to avoid.

## Decision

- **Dropped:** T-D1, T-D2, T-S1, T-Si1 and N-S2 (no mentions), and T-A1 and N-A1 (1 and 3
  mentions).
- **No allow-list is built.** Rules-mode certain flags stay disabled.
- **The rules layer's new role:** it nominates structured candidates for the model's
  quote-verified check, guards against false certain flags, and gives a no-model quick
  screen whose findings are warnings (`docs/spec.md`, "Deterministic rules layer").

## Caveat

These are ceilings on OmissionBench's development data, whose transcripts come from
ACI-Bench and PriMock57 consultations and authored scenarios. Vendor transcripts or
templated notes may be plainer. Only real samples would show whether some shapes become
common there.

Attribution: OmissionBench, Fox et al. (2026), CC BY 4.0, https://doi.org/10.57967/hf/10258.
The omission method is Fox et al. (arXiv:2608.31016), building on MED-OMIT.
