# scribecheck report: demo-001

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, and changed doses, frequencies, routes, sides and symptom negation; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 2 | 0 | 0 | 0 | 0 |

## Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | dose | supporting | possible | metformin 500 mg | "metformin 500 mg" (turn 4) | - | Medications |
| F2 | omission | medicine | supporting | possible | atorvastatin | "atorvastatin" (turn 4) | - | Medications |

## Unchecked items

No unchecked items.
