# scribecheck report: demo-003

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, changed doses, frequencies, routes, sides and symptom negation, and examination findings in telephone or video consultations; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 |

## Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | side | supporting | possible | left knee | "left knee" (turn 2) | - | History |
| F2 | changed_fact | frequency | critical | possible | three times a day -> twice daily | "ibuprofen 400 mg three times a day" (turn 4) | "Ibuprofen 400 mg twice daily" (Medications) | - |

## Unchecked items

No unchecked items.
