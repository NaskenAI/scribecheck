# scribecheck batch report

Demo: synthetic consultations bundled with scribecheck, checked in rules mode. They describe no real patient and have not yet been reviewed by a clinician.

| Case | Consultation | Decision | Omissions | Changed facts | Unsupported | Unchecked |
| --- | --- | --- | --- | --- | --- | --- |
| demo-001 | demo-001 | warn | 2 | 0 | 0 | 0 |
| demo-002 | demo-002 | warn | 1 | 0 | 0 | 4 |
| demo-003 | demo-003 | warn | 1 | 1 | 0 | 0 |

## scribecheck report: demo-001

Case demo-001: `demo-001.transcript.json` and `demo-001.note.md`

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, and changed doses, frequencies, routes, sides and symptom negation; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 2 | 0 | 0 | 0 | 0 |

### Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | dose | supporting | possible | metformin 500 mg | "metformin 500 mg" (turn 4) | - | Medications |
| F2 | omission | medicine | supporting | possible | atorvastatin | "atorvastatin" (turn 4) | - | Medications |

### Unchecked items

No unchecked items.

## scribecheck report: demo-002

Case demo-002: `demo-002.transcript.json` and `demo-002.note.md`

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, and changed doses, frequencies, routes, sides and symptom negation; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | 4 |

### Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | allergy | critical | possible | allergy: penicillin | "allergic to penicillin" (turn 6) | - | Allergies |

### Unchecked items

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

## scribecheck report: demo-003

Case demo-003: `demo-003.transcript.json` and `demo-003.note.md`

**Decision: warn** (mode rules, standard clinical)

> Rules mode: findings are warnings at possible confidence and never fail a build. It checks omissions of medicines, doses, frequencies, allergies and sides, and changed doses, frequencies, routes, sides and symptom negation; every other check needs the model.

| Omissions | Changed facts | Unsupported | Not verifiable | Unchecked |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0 | 0 |

### Flags

| ID | Type | Category | Severity | Confidence | Finding | Transcript evidence | Note evidence | Expected section |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | omission | side | supporting | possible | left knee | "left knee" (turn 2) | - | History |
| F2 | changed_fact | frequency | critical | possible | three times a day -> twice daily | "ibuprofen 400 mg three times a day" (turn 4) | "Ibuprofen 400 mg twice daily" (Medications) | - |

### Unchecked items

No unchecked items.
