# Severity agreement: claude-medium-v4-full

Our audit's severity for each removed fact we extracted, against the benchmark's
severity, over 148 omission notes with a Pass A trace. Development data only;
counts only. Each omission note counts once, so a fact removed from several notes of
one consultation counts once per note. 'Higher' means our audit graded the fact more
severe than the benchmark. Regenerate with `scripts/score_run.py severity`.

### Word matcher (score >= 0.5)

Removed facts matched and graded by the audit: 86. Cohen's kappa: 0.60. Exact agreement: 67/86; we grade higher: 6; we grade lower: 13.

| ours \ benchmark | critical | supporting | peripheral | total |
|---|---|---|---|---|
| critical | 41 | 5 | 0 | 46 |
| supporting | 9 | 20 | 1 | 30 |
| peripheral | 3 | 1 | 6 | 10 |
| total | 53 | 26 | 7 | 86 |
