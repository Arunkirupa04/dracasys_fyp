# LFHE MSE vs DA-MSE — Comparison Analysis

**Source notebook:** `notebooks/lfhe_mse_vs_damse_comparison.ipynb`  
**Authoritative experiment:** `experiments/loss_function_hypothesis_2026-07-26_130252/`  
**Publication figures:** `notebooks/outputs/lfhe_mse_vs_damse_comparison/`  
**Status:** Read-only analysis of completed LFHE run (no retraining)

---

## Executive summary

**Verdict: `inconclusive_I1`**

DA-MSE **strongly fixes variance collapse** (dispersion and energy recovery) but **does not fix temporal alignment or forecasting accuracy**. The experiment confirms that the **loss function is implicated in variance collapse**, but the frozen DA-MSE formulation (λ=1.0) is **not sufficient** for a usable hybrid forecaster.

| Criterion | Result | Meaning |
|-----------|--------|---------|
| **S1 — Dispersion recovery** | PASS | Std ratio 0.10 → 0.83; 100% of containers improved |
| **S2 — Correlation improvement** | FAIL | Pearson r 0.113 → 0.087; paired bootstrap CI excludes gain |
| **S3 — MAE guardrail** | FAIL | Residual MAE +0.011 (limit 0.01) |

---

## 1. What the experiment confirmed (mechanism)

Parity checks in the comparison notebook verify both arms used the **same** CSRLE B0-LC data, GRU architecture, Prophet models, and 99 containers — **only the loss function changed**.

That isolation supports a causal reading:

- **MSE** → predicted residual std ≈ **10%** of actual (variance collapse)
- **DA-MSE** → predicted residual std ≈ **83%** of actual (near recovery)
- **Ridge reference** (CSRLE) ≈ **22%** std ratio

**Conclusion:** CSRLE’s central suspicion is supported — **MSE loss contributes to variance collapse**, and changing the loss can recover amplitude without changing architecture or data.

---

## 2. Training behavior

| | MSE | DA-MSE |
|--|-----|--------|
| Epochs run | 13 | 17 |
| Best epoch | **3** | 7 |
| Best val loss | **1.70** | 2.44 |
| Final val loss | 1.71 | 2.60 |

**Insights:**

- MSE converges **faster and to lower validation loss** — expected because DA-MSE adds a dispersion penalty term.
- DA-MSE requires more epochs; the penalty makes optimization harder.
- Higher DA-MSE loss values do **not** imply worse MSE fit alone — part of the objective is explicit variance matching.

**Takeaway:** DA-MSE trades easy convergence for explicit pressure on σ_ŷ.

---

## 3. Overall metric comparison (99 containers, cohort mean)

| Metric | MSE | DA-MSE | Δ (DA−MSE) | % change |
|--------|-----|--------|------------|----------|
| Residual MAE (scaled) | 0.093 | 0.105 | +0.011 | +12% |
| Residual RMSE | 0.137 | 0.149 | +0.012 | +9% |
| Pearson r | **0.113** | 0.087 | −0.026 | −23% |
| R² | −0.32 | −2.39 | — | worse |
| Std ratio | 0.101 | **0.827** | +0.726 | +726% |
| Energy recovery (ERR) | 0.011 | **0.823** | +0.812 | +7400% |
| CPU day-1 MAE | 1.76 | 2.01 | +0.25 | +14% |
| CPU day-1 MAPE (%) | 117 | 151 | +33 | +28% |
| Peak MAE | 3.47 | 3.63 | +0.16 | +5% |

**Paired bootstrap (DA − MSE, 99 pairs):**

| Metric | Mean diff | 95% CI | Fraction DA better |
|--------|-----------|--------|-------------------|
| Std ratio | +0.726 | [+0.529, +0.988] | **100%** |
| Pearson r | −0.026 | [−0.055, +0.003] | 43% |
| Residual MAE | +0.011 | [+0.010, +0.013] | 10% |

**Insights:**

- DA-MSE is a **scale fix**, not an **accuracy fix**.
- R² collapses further because larger-amplitude wrong predictions inflate squared error.
- CPU metrics worsen slightly — recovering residual variance **without correct timing** hurts the hybrid CPU forecast.

---

## 4. Trajectory and CPU plots (qualitative)

### MSE trajectories

- Residual predictions appear **flat/smoothed** relative to actual swings.
- CPU hybrid tracks Prophet closely; GRU adds little structured residual correction.

### DA-MSE trajectories

- Predictions show **much larger amplitude**.
- Peaks and troughs often **do not align** with actual residual timing.
- CPU plots: Hybrid DA-MSE can overshoot or move at wrong times → higher MAE/MAPE.

### Auto-selected representative containers

| Role | Container | Key numbers |
|------|-----------|-------------|
| Best std-ratio gain | `c_13308` | MSE σ=0.002, DA σ=0.044; both r negative at lag 0 |
| Worst Pearson delta | `c_13389` | MSE r=**0.349** → DA r=**−0.235**; sign agreement 56% → 33% |
| Median DA correlation | `c_15035` | MSE nearly flat (σ=0.002); DA σ=0.048, r≈0.09 |

**Insight:** DA-MSE can **hurt** containers where MSE had modest positive correlation — amplification without alignment is actively harmful.

---

## 5. Residual distribution analysis

Histograms, KDE, boxplots, and violin plots (Section 7 of notebook) show:

- **Actual residuals:** wide, centered near zero.
- **MSE predictions:** narrow, concentrated near zero (collapsed).
- **DA-MSE predictions:** wider, sometimes **wider than actual** (over-dispersion).

Cohort sign agreement ≈ **53%** for both arms — barely above chance for a zero-mean oscillating signal. DA-MSE does not fix **which direction** the residual moves at each timestep.

---

## 6. Variance recovery (full trajectory and horizon bands)

### Cohort-level σ

- MSE: σ_ŷ ≈ **10%** of σ_y
- DA-MSE: σ_ŷ ≈ **83%** of σ_y
- **100%** of containers have higher std ratio under DA-MSE

### Horizon-wise std ratio (cohort mean)

| Band | Steps | MSE | DA-MSE |
|------|-------|-----|--------|
| h01_12 | 1–12 | 0.19 | **2.43** |
| h13_24 | 13–24 | 0.15 | 2.00 |
| h25_48 | 25–48 | 0.07 | 0.83 |
| h49_72 | 49–72 | 0.10 | 1.02 |
| h73_96 | 73–96 | 0.18 | 1.79 |

**Insights:**

- MSE collapse is **uniformly low** across all bands (~7–19%) — not only an early-horizon artifact.
- DA-MSE **overcompensates** in early bands (ratio > 1, especially steps 1–12).
- Mid-horizon (25–48) is closest to balanced recovery (~0.83).
- The per-sequence dispersion penalty can inflate variance **unevenly across the 96-step horizon**.

---

## 7. Residual energy recovery

| | MSE | DA-MSE |
|--|-----|--------|
| Mean ERR (Σŷ²/Σy²) | 0.011 | 0.823 |
| Median ERR | 0.006 | 0.311 |
| 95% CI (cohort) | 0.009–0.013 | 0.47–1.27 |

**Insight:** Mean ERR rises dramatically, but the **median is much lower than the mean** — a subset of containers drives strong over-recovery (ERR > 1). DA-MSE fixes energy on average but **inconsistently**, with overshooting in some cases.

---

## 8. Correlation analysis

- Cohort mean r: **0.113 → 0.087**
- Only **~43%** of containers improve Pearson r under DA-MSE
- Paired bootstrap mean Δr = **−0.026**, CI **[−0.055, +0.003]** — excludes meaningful improvement

Paired scatter (MSE r vs DA r): most points lie at or **below** the diagonal — DA-MSE rarely improves correlation even when it improves std ratio.

**Why DA-MSE increased variance but reduced correlation:** The model learns **how large** to predict, not **when** to predict up vs down. Pearson r requires correct **phase and sign**, not just amplitude.

---

## 9. Extra analysis — amplitude vs temporal alignment

Measured across all 99 containers (Section 19 of notebook):

| Diagnostic | MSE | DA-MSE | Interpretation |
|------------|-----|--------|----------------|
| Zero-lag Pearson r | 0.113 | 0.087 | Alignment did not improve |
| Sign agreement | 53.5% | 52.3% | ~coin flip; no gain |
| Median best lag (max \|r\|) | 0 | 1 | Near zero — **not** large phase shift |

### Supported causes (evidence-based)

- **Wrong sign / mis-timed oscillations at zero lag** — sign agreement ~52%, zero-lag r flat or worse
- **Amplitude–alignment decoupling** — std ratio up, r down

### Not supported as primary cause

- **Large systematic phase shift** — best lag remains near 0–1 for both arms
- **Single uniform peak-timing error** — peak-shift histograms show mixed shifts, not one global offset

### Illustrative case: `c_13389`

MSE had meaningful r (0.35); DA-MSE increased σ but **flipped** correlation to −0.24 and cut sign agreement from 56% to 33%. This is **destructive amplification**, not phase correction.

**Primary measurable conclusion:** DA-MSE increases residual scale (variance/energy) without improving sign or zero-lag correlation. Large systematic phase shift is not the dominant failure mode.

---

## 10. Hypothesis evaluation (S1 / S2 / S3)

| Question | Answer |
|----------|--------|
| Did DA-MSE reduce variance collapse? | **Yes** (S1 pass) |
| Did it improve temporal alignment? | **No** (S2 fail) |
| Did it stay within MAE tolerance? | **No** (S3 fail, +0.011 > 0.01) |
| Did it improve CPU forecasting? | **No** |

### Why S1 passed

DA-MSE restored cohort std ratio (~0.83 vs ~0.10) with significant paired gain (bootstrap CI entirely positive; 100% of containers improved).

### Why S2 failed

Pearson r did not improve; paired bootstrap CI for DA−MSE is negative at the lower bound (−0.055).

### Why S3 failed

Residual MAE increased by +0.0114, exceeding the frozen guardrail of 0.01.

### Why `inconclusive_I1` (not full support or rejection)

- **Not full support:** S1 alone is insufficient; frozen criteria require dispersion **and** correlation **and** MAE guardrail.
- **Not full rejection:** Variance collapse **is** loss-driven; DA-MSE proves the mechanism. The hypothesis is partially validated at the mechanism level but fails on downstream forecasting utility.

---

## 11. Thesis-ready narrative

### What improved

- Residual **dispersion** (std ratio, σ scatter, histogram width)
- Residual **energy** (ERR)
- Visual **amplitude** of GRU corrections

### What did not improve

- **Temporal Pearson correlation** (primary alignment metric)
- **Sign agreement** and turning-point timing
- **Residual MAE** (guardrail breached)
- **CPU day-1 and peak metrics**

### What remains open

- Why the dispersion penalty increases σ without learning **conditional structure** (sign/phase)
- Whether **λ=1.0** over-penalizes early horizons (band h01_12 mean ratio 2.43)
- Whether a **different** dispersion-aware loss (correlation-aware, horizon-weighted, or lower λ) could recover both scale and alignment — **out of scope for frozen LFHE v1.1**

### One-sentence conclusion

> **MSE causes variance collapse; DA-MSE recovers residual scale and energy but not temporal structure — fixing amplitude alone is insufficient for hybrid CPU forecasting under the frozen LFHE protocol.**

---

## 12. Publication figures (thesis / demo)

Exported from Section 17 of the comparison notebook to  
`notebooks/outputs/lfhe_mse_vs_damse_comparison/` (PNG 300 dpi + PDF):

| File | Recommended use |
|------|-----------------|
| `01_training_val_loss` | Methods — training convergence |
| `02_std_ratio_distribution` | **Main result** — variance collapse vs recovery |
| `03_horizon_std_ratio` | Horizon analysis — early-band overcompensation |
| `04_energy_recovery` | Diagnostic — ERR distribution |
| `05_pearson_scatter` | Negative result — correlation did not improve |
| `06_case_study_trajectory` | Qualitative — amplitude without alignment |

For interactive demos, use **Section 12** (container explorer) or **Section 13** case studies (`c_13308`, `c_13389`, `c_15035`).

---

## 13. Related artifacts

| Item | Path |
|------|------|
| Comparison notebook | `notebooks/lfhe_mse_vs_damse_comparison.ipynb` |
| Visualization dashboard | `notebooks/lfhe_visualization.ipynb` |
| Final report (JSON) | `experiments/.../reports/final_report.json` |
| Final report (MD) | `experiments/.../reports/final_report.md` |
| Cohort summary CSV | `experiments/.../tables/lfhe_summary.csv` |
| Paired bootstrap | `experiments/.../comparisons/paired_mse_vs_da_mse.json` |
| Frozen protocol | `docs/loss_function_experiment/` |

---

*Generated from executed notebook outputs and frozen LFHE artifacts. No retraining or artifact modification.*
