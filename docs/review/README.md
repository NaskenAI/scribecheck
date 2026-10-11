# Clinician review sheets

These sheets list the in-house additions to scribecheck's lexicons that need a clinician's check
before release. They are generated from the data files; please don't edit any column except the
last two.

| Sheet | What to check |
| --- | --- |
| `inn_usan_review.csv` | Each international (INN/BAN) drug name and the US name it is matched to. |
| `symptom_phrases_review.csv` | Lay phrasings matched to a symptom (`lay`), Symptom Ontology terms folded into another term (`merge`), and ontology synonyms deliberately not matched (`excluded_synonym`). |
| `extra_ingredients_review.csv` | Generic ingredients prescribed in India or the UK that RxNorm lacks, matched as drug names. |

## How to fill them in

For every row, fill in the `decision` column with one of:

- `keep`: correct as it stands.
- `change`: wrong, but fixable; write the correction in `clinician_comment`, for example the right
  US name, the right symptom term, or the reason an excluded synonym should be matched after all.
- `drop`: should not be matched at all; add a short reason in `clinician_comment` if useful.

Use `clinician_comment` for anything else worth knowing. Leave the other columns unchanged, so
the answers can be matched back to the data, and return the files as CSV.

The `notes` column explains why a row needs attention. One phrase, "white discharge", is already
withheld from matching because it maps to vaginal discharge and is unsafe without the patient's
sex; it is listed so you can confirm that decision.

## Regenerating

After any change to `src/scribecheck/data/inn_usan.json`, `symptoms.json` or
`ingredients_extra.json`, run:

```bash
python scripts/build_review_sheets.py
```

This overwrites the sheets with empty decision columns, so copy any returned answers elsewhere
first. A test fails while the committed sheets are out of date.
