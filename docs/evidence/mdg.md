# Minimum detectable gain (provisional)

This is the smallest improvement in omission detection over a published method that the
held-out comparison can reliably detect. It is written down before any held-out run of
scribecheck.

**Provisional.** It is computed against the published gpt-5.4 results. M4 recomputes it
against the Claude Sonnet 5.5 baselines, which are what scribecheck will be compared with,
using the discordance seen in development runs.

## Set-up

- **Notes:** the 131 errored notes that omit a fact, in the official 151-pair held-out
  subset. Each note is scored by both methods, so the comparison is paired.
- **Test:** exact McNemar, two-sided, alpha 0.05. This is the exact binomial test on the
  discordant notes with p = 0.5. Power counts only rejections in scribecheck's favour.
- **Power:** 80%.
- **Baselines:** the published gpt-5.4 strict-majority results, recomputed in Gate 1(a)
  (`docs/evidence/gate1a.md`), each at 1/47 false alarms on the clean notes:

| Baseline | Detection |
|---|---|
| Route one (B3 pipeline, critical fact judged absent), majority of 3 | 28/131 = 21.4% |
| Route two (`power-gepa-high`, aggregate below 10), majority of 3 | 42/131 = 32.1% |

## Discordance assumption

McNemar's power depends on how many notes the two methods disagree on, not only on their
detection rates. The discordance is unknown until both methods run on the same notes, so it
is bracketed:

- **Planning assumption: independence.** Whether scribecheck catches a note is unrelated to
  whether the baseline does. With detection rates p₀ (baseline) and p₁ (scribecheck):
  - P(scribecheck catches, baseline misses) = p₁(1 − p₀);
  - P(baseline catches, scribecheck misses) = p₀(1 − p₁).

  This gives the most disagreement for given rates, and so the largest gain needed: it is
  conservative.
- **Optimistic bound: nested.** scribecheck catches every note the baseline catches, so
  every discordant note favours scribecheck. This gives the least disagreement and the
  smallest gain needed.

Two methods applied to the same notes usually agree more than independent ones would, so
the true figure is likely between the two.

## Result

| Baseline | Assumption | Detection needed | Gain | Power | Expected discordant notes |
|---|---|---|---|---|---|
| Route one, 28/131 (21.4%) | **independence (planning)** | **50/131 (38.2%)** | **+22 notes, +16.8 points** | 0.816 | 56.6 (39.3 for, 17.3 against) |
| Route one, 28/131 (21.4%) | nested (optimistic) | 36/131 (27.5%) | +8 notes, +6.1 points | 0.817 | 8.0 (all for) |
| Route two, 42/131 (32.1%) | **independence (planning)** | **66/131 (50.4%)** | **+24 notes, +18.3 points** | 0.827 | 65.7 (44.8 for, 20.8 against) |
| Route two, 42/131 (32.1%) | nested (optimistic) | 50/131 (38.2%) | +8 notes, +6.1 points | 0.817 | 8.0 (all for) |

**What this means:**

- On 131 notes, against gpt-5.4's published results, scribecheck needs roughly **+17 to +18
  points** of detection to be detected reliably under the planning assumption.
- If its catches largely contain the baseline's, **+6 points** is enough.
- Gains between those figures may be real but would often not reach significance on this
  subset. Gate 2 allows for that by adding the 79 supplementary pairs when the headline
  result is borderline.

**False alarms.** These are not covered here. With 47 clean notes and a baseline of 1/47,
only a large rise in false alarms is detectable, so detection gains must be read beside
scribecheck's own false-alarm count, as the dataset card requires.

## What M4 changes

M4 recomputes this figure:

- **Baselines:** against the Claude Sonnet 5.5 baselines, which run both published methods
  once, route one as a single-model variant.
- **Discordance:** using the rate observed between scribecheck and those baselines on
  development data, in place of the bracketing above.

The baseline rates will change, so both the needed detection and the gain will move.

## Computation

Power is exact. It sums over the number of discordant notes d, which follows
Binomial(131, p₁₀ + p₀₁). For each d, it sums over the discordant notes b in scribecheck's
favour, which follow Binomial(d, p₁₀ / (p₁₀ + p₀₁)), counting the cases where the exact
two-sided McNemar test rejects with b > d − b. The needed detection is the smallest count
out of 131 whose power reaches 0.80.

```python
from math import comb

N, ALPHA, POWER = 131, 0.05, 0.80

def pmf(n, k, p):
    return comb(n, k) * p**k * (1 - p)**(n - k)

def rejects(b, d):  # exact two-sided McNemar, in favour of b
    if d == 0 or 2 * b <= d:
        return False
    return min(1.0, 2 * sum(comb(d, i) for i in range(b, d + 1)) / 2**d) <= ALPHA

def power(p10, p01):
    pd = p10 + p01
    q = p10 / pd
    return sum(pmf(N, d, pd) * sum(pmf(d, b, q) for b in range(d + 1) if rejects(b, d))
               for d in range(N + 1))

def needed(k0, nested=False):
    p0 = k0 / N
    for k1 in range(k0 + 1, N + 1):
        p1 = k1 / N
        p10, p01 = (p1 - p0, 0.0) if nested else (p1 * (1 - p0), p0 * (1 - p1))
        if power(p10, p01) >= POWER:
            return k1

print(needed(28), needed(28, nested=True), needed(42), needed(42, nested=True))  # 50 36 66 50
```

Attribution: OmissionBench, Fox et al. (2026), CC BY 4.0, https://doi.org/10.57967/hf/10258.
The omission method is Fox et al. (arXiv:2608.31016), building on MED-OMIT.
