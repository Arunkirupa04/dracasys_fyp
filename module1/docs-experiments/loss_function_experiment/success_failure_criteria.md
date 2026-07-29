# Success and Failure Criteria

## Decision framework

Hypothesis adjudication uses **pre-specified primary criteria on B0**. Condition A (if run) is confirmatory only. CPU metrics, R², **horizon-wise variance recovery**, and **energy recovery ratio (ERR)** do **not** determine accept/reject.

### Diagnostic metrics explicitly excluded from decision rules

| Metric | Role |
|--------|------|
| Horizon-band std ratio (bands 1–12 … 73–96) | Descriptive — thesis interpretation only |
| Energy recovery ratio (ERR) | Descriptive — complements std ratio; not an optimization target |
| Horizon-band ERR | Descriptive — optional secondary table |

See [diagnostic_analysis.md](diagnostic_analysis.md).

---

## Primary success criteria (support H₁)

All **three** must hold on B0 full cohort (n = 99):

### S1 — Dispersion recovery

| Criterion | Threshold |
|-----------|-----------|
| Cohort **mean std ratio** (DA-MSE) | **≥ 0.12** |
| Paired mean Δ std ratio (DA-MSE − MSE) | **> 0** with **95% bootstrap CI excluding 0** |
| Container win rate | **≥ 60%** containers with DA-MSE std ratio > MSE std ratio |

*Rationale:* 0.12 is ~1.7× MSE baseline (0.073) and ~55% of Ridge ceiling (0.22) — a meaningful non-trivial shift.

### S2 — Correlation improvement

| Criterion | Threshold |
|-----------|-----------|
| Paired mean Δr (DA-MSE − MSE) | **> 0** with **95% bootstrap CI excluding 0** |
| Cohort mean r (DA-MSE) | **≥ 0.10** (absolute floor above MSE ≈ 0.084) |

*Rationale:* Correlation must improve **as consequence** of dispersion fix, not alone (see inconclusive rules).

### S3 — MAE guardrail (no spurious inflation)

| Criterion | Threshold |
|-----------|-----------|
| Cohort mean residual MAE increase | **≤ 0.010** absolute (scaled space) vs MSE arm |
| Paired mean Δ MAE 95% CI upper bound | **≤ 0.015** |

*Rationale:* Ridge showed similar MAE with higher r — large MAE sacrifice implies a different failure mode (over-dispersion noise).

**If S1 + S2 + S3 all pass → SUPPORT H₁ (objective-primary variance collapse).**

---

## Primary failure criteria (reject H₁)

Any **one** of the following is sufficient to reject H₁ on B0:

### F1 — No dispersion change

Paired 95% CI for Δ std ratio **includes 0** AND cohort mean std ratio (DA-MSE) **< 0.10**.

### F2 — No correlation change

Paired 95% CI for Δr **includes 0** AND cohort mean r (DA-MSE) **< 0.09**.

### F3 — MAE guardrail violation

Cohort mean residual MAE (DA-MSE) exceeds MSE by **> 0.015** OR paired Δ MAE CI lower bound **> 0.010**.

### F4 — Control reproduction failure

LFHE-MSE-B0 fails Stage 2 reproduction gate (methodology.md) — experiment **invalid**; treat as **design/pipeline failure**, not H₀ support.

**If F1 or F2 (with F3 not violated) → REJECT H₁.**

---

## Inconclusive outcomes

### I1 — Dispersion without correlation

S1 passes (std ratio ↑) but S2 fails (r unchanged).

**Interpretation:** Objective increases amplitude but **not shape tracking** — partial mechanism; **inconclusive for full H₁**. Suggests dispersion penalty works but lag structure still not captured (architecture/linearity limit).

### I2 — Correlation without dispersion

S2 passes but S1 fails.

**Interpretation:** **Inconclusive / inconsistent with DA-MSE mechanism** — investigate bugs, metric computation, or non-monotonic scaling effects. Do not support H₁.

### I3 — Both pass with MAE violation (F3)

Dispersion and r improve but MAE guardrail fails.

**Interpretation:** **Inconclusive for deployment relevance** — supports objective effect on shape but at unacceptable error cost; hypothesis **partially supported** for mechanism, **rejected** for practical Hybrid improvement.

### I4 — Active injection vs full cohort conflict

Full cohort passes S1/S2 but α > 0 subset (n = 91) shows null effect.

**Interpretation:** **Inconclusive** — effect may be driven by α = 0 edge cases; report both; defer strong claim.

### I5 — B0 passes, Condition A fails (if run)

**Interpretation:** **Synthetic-specific objective gain** — supports limited H₁ on B0 only; real-data generalization **not established**.

---

## Secondary outcomes (report, do not override primary decision)

| Outcome | Reporting |
|---------|-----------|
| R² still negative | Expected; note direction only |
| CPU Hybrid − Prophet MAE | Report; no success weight |
| Horizon-band std ratio (1–12 … 73–96) | **Descriptive** — progressive collapse pattern |
| Energy recovery ratio (ERR) | **Descriptive** — signal mass reproduced |
| Horizon-band r (CSRLE bands) | Descriptive; short bands may show largest Δ |
| Training epochs to converge | DA-MSE may stop earlier/later — report, not criterion |

---

## Outcome interpretation matrix

| std ratio | Pearson r | MAE | Verdict |
|-----------|-------------|-----|---------|
| ↑ (S1) | ↑ (S2) | OK (S3) | **Support H₁** |
| ↑ (S1) | ↔ | OK | **Inconclusive (I1)** |
| ↔ | ↑ | OK | **Inconclusive (I2)** |
| ↑ | ↑ | Bad (F3) | **Partial / inconclusive (I3)** |
| ↔ | ↔ | OK | **Reject H₁ (F1+F2)** |
| ↓ | any | any | **Reject H₁ + investigate** |

---

## Reference anchors (not success thresholds)

| Benchmark | std ratio | r |
|-----------|-----------|---|
| Stage 2 MSE GRU B0 | 0.073 | 0.084 |
| Ridge B0 (frozen analysis) | 0.220 | 0.158 |
| Condition A MSE | 0.050 | 0.016 |

DA-MSE **need not match Ridge** to support H₁. Matching Ridge while MSE does not would be **strong** support.

---

## Thesis language guidance

### If H₁ supported

> “Replacing MSE alone with a dispersion-augmented objective recovered a significant fraction of residual amplitude and correlation on CSRLE B0 without architectural change, indicating that variance collapse is **substantially driven by the training objective** rather than exclusively by Hybrid GRU capacity.”

### If H₁ rejected

> “Dispersion-augmented MSE did not materially alter std ratio or correlation relative to MSE under identical Hybrid architecture, suggesting variance collapse reflects **architecture, optimization, or linear target structure** beyond objective choice alone.”

### If inconclusive

> “Dispersion penalties increased predicted variance without consistent correlation gains (or vice versa), indicating **partial** objective effects requiring further targeted study — not sufficient to attribute collapse primarily to MSE.”

---

## Checklist answers (15–17)

| # | Question | Answer |
|---|----------|--------|
| 15 | Outcomes supporting hypothesis | S1 + S2 + S3 all pass on B0 |
| 16 | Outcomes rejecting hypothesis | F1, F2, or invalid control F4 |
| 17 | Inconclusive | I1–I5 patterns above |
