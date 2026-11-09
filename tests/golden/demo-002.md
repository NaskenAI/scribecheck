# scribecheck report: demo-002

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, changed doses, frequencies, routes, sides and symptom negation, and examination findings in telephone or video consultations; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 4 |

## Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | allergy | critical | possible | allergy: penicillin | "allergic to penicillin" (turn 6) | - | Allergies |

## Unchecked items

| Primary reason | Items |
| --- | --- |
| planned_or_past | 2 |
| question_unanswered | 2 |

| Category | Item | Reasons | Transcript evidence | Note evidence |
| --- | --- | --- | --- | --- |
| dose | doxycycline | planned_or_past, certain_disabled_pending_allowlist | "doxycycline" (turn 7) | "Doxycycline" (Plan) |
| frequency | doxycycline | planned_or_past, certain_disabled_pending_allowlist | "doxycycline" (turn 7) | "Doxycycline" (Plan) |
| symptom | chest pain | question_unanswered, certain_disabled_pending_allowlist | "chest pain" (turn 3) | "chest pain" (History) |
| symptom | dyspnea | question_unanswered, certain_disabled_pending_allowlist | "shortness of breath" (turn 3) | "shortness of breath" (History) |
