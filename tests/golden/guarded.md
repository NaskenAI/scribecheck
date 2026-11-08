# scribecheck report: guarded

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, and changed doses, frequencies, routes, sides and symptom negation; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 1 |

## Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | changed_fact | dose | critical | possible | 5 mg -> 10 mg | "ramipril 5 mg" (turn 2) | "Ramipril 10 mg" (Medications) | - |

## Unchecked items

| Primary reason | Items |
| --- | --- |
| planned_or_past | 1 |

| Category | Item | Reasons | Transcript evidence | Note evidence |
| --- | --- | --- | --- | --- |
| medicine | amlodipine | planned_or_past | "amlodipine" (turn 2) | - |
