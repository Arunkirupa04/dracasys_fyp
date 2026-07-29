# Results

**Analysis run:** `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/`  
**Synthesis:** `results/evidence_synthesis.json`

---

## 1. Confirmed Stage 2 gap (B0 cohort, n=99)

| Metric | GRU | Ridge |
|--------|-----|-------|
| Mean residual Pearson r | 0.084 | **0.158** |
| Mean std ratio (pred/actual) | 0.073 | **0.220** |
| Mean MAE | 0.094 | 0.093 |
| Mean R² | −0.332 | −0.390 |
| Containers with Ridge r > GRU r | — | **62.6%** |

Ridge wins on **correlation and variance recovery**, not MAE.

---

## Analysis 1 — B0 residual linearity (pooled validation)

| Metric | Value |
|--------|-------|
| AR(1) R² | **0.201** |
| AR(2) R² | **0.213** |
| Linear trend R² | 0.003 |
| Poly(2) R² | 0.004 |
| Nonlinear gap (poly2 − AR2) | **−0.210** |
| Mean \|ACF\| lags 1–10 | **0.259** |

**Finding:** After Prophet, B0 residuals exhibit **strong short-lag linear dependence**. Polynomial trend adds negligible explanatory power beyond AR(2).

Plots: `plots/analysis1_b0_acf_pacf.{png,pdf}`, `analysis1_b0_welch_spectrum.{png,pdf}`

---

## Analysis 2 — Complexity (pooled Day-1 residuals)

| Condition | Permutation entropy | Sample entropy | Std |
|-----------|--------------------:|---------------:|----:|
| A (control) | 1.690 | 0.608 | 0.177 |
| B0 | 1.692 | 0.635 | 0.175 |
| C0 | 1.757 | 0.700 | 0.175 |

**Finding:** B0 residuals are **not substantially simpler** than A or C0 on entropy metrics.

---

## Analysis 3 — Frequency domain (cohort mean)

| Model | Mean spectral correlation with actual |
|-------|--------------------------------------:|
| GRU | 0.318 |
| Ridge | 0.317 |

**Finding:** Mean spectral shape similarity is **nearly identical**. Ridge advantage is **not** primarily better frequency preservation in the cohort mean.

---

## Analysis 4 — Variance recovery

| Model | Mean amplitude ratio |
|-------|---------------------:|
| GRU | 0.073 |
| Ridge | 0.220 |

Plot: `plots/analysis4_variance_recovery_scatter.{png,pdf}`

**Finding:** GRU predictions use ~7% of actual residual std; Ridge ~22%. GRU **collapses toward near-constant** corrections.

---

## Analysis 6 — Error decomposition (cohort mean)

| Component | GRU | Ridge |
|-----------|-----|-------|
| Amplitude ratio | 0.073 | 0.220 |
| Pearson r | 0.084 | 0.158 |
| \|Bias\| | 0.015 | 0.018 |
| MAE | 0.094 | 0.093 |

**Finding:** Primary GRU failure mode is **low amplitude (variance)**, not large bias or phase (phase lag available in `error_decomposition.csv`).

---

## Analysis 7 — Horizons (cohort mean)

| h | GRU r | Ridge r | GRU std ratio | Ridge std ratio |
|---|------:|--------:|--------------:|----------------:|
| 4 | 0.151 | **0.194** | 0.385 | **0.989** |
| 12 | 0.108 | **0.204** | 0.147 | 0.482 |
| 96 | 0.084 | **0.158** | 0.073 | 0.220 |

Ridge leads at **all reported horizons** on correlation; short horizons show largest std-ratio gap.

Plot: `plots/analysis7_horizon_comparison.{png,pdf}`

---

## Analysis 8 — Container ranking

- Top Ridge r example: `c_15179` (see `results/top10_ridge_containers.csv`)
- Representative (median Δr): `c_11461`
- Scatter: `plots/analysis8_container_scatter.{png,pdf}`

---

## Analysis 9 — Generator properties

See `results/generator_property_correlations.csv`. α correlates weakly with residual std; **no strong α-driven Ridge−GRU gap** across cohort.

---

## Analysis 10 — Synthesis JSON

All key aggregates in `results/evidence_synthesis.json`.
