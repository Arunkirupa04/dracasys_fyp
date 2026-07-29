# Diagnostic Analysis Protocol

**Status:** Frozen — part of LFHE v1.1 protocol  
**Role:** Descriptive interpretability only — **not** used for hypothesis accept/reject, model selection, or early stopping

---

## Purpose

Extend LFHE reporting beyond the single full-trajectory std ratio with two **diagnostic** analyses:

1. **Horizon-wise variance recovery** — whether dispersion collapse worsens at longer forecast horizons.
2. **Residual energy recovery** — how much squared residual signal the model reproduces.

These complement primary metrics in [evaluation_protocol.md](evaluation_protocol.md) and [success_failure_criteria.md](success_failure_criteria.md) without changing them.

---

## 1. Horizon-wise variance recovery

### Motivation

The overall 96-step std ratio may mask **horizon-dependent collapse** — e.g. strong short-horizon amplitude with progressive flattening at longer lags. This analysis is **descriptive** and answers:

> Does variance collapse become progressively worse toward longer forecasting horizons, or remain approximately constant across the Day-1 window?

### Horizon bands (frozen)

| Band ID | Step range (1-indexed) | Steps |
|---------|------------------------|-------|
| `h01_12` | 1–12 | 12 |
| `h13_24` | 13–24 | 12 |
| `h25_48` | 25–48 | 24 |
| `h49_72` | 49–72 | 24 |
| `h73_96` | 73–96 | 24 |

Bands partition the full 96-step Day-1 validation trajectory without overlap.

### Per-container, per-band computation

For container \(c\), band \(b\), with actual residual vector \(\mathbf{y}_{c,b}\) and predicted \(\hat{\mathbf{y}}_{c,b}\) over steps in band \(b\) (scaled residual space):

| Quantity | Symbol | Formula |
|----------|--------|---------|
| Actual std | \(\sigma_y\) | \(\text{std}(\mathbf{y}_{c,b})\), ddof = 0 |
| Predicted std | \(\sigma_{\hat{y}}\) | \(\text{std}(\hat{\mathbf{y}}_{c,b})\), ddof = 0 |
| Std ratio | \(R^{\sigma}_{c,b}\) | \(\sigma_{\hat{y}} / \sigma_y\) |

**Edge case:** If \(\sigma_y < 10^{-8}\), set ratio to NaN and flag container-band; report count of flagged cells. Primary tables use available values; sensitivity analysis may exclude flagged bands.

### Cohort statistics (per arm, per band)

Across containers \(c = 1,\ldots,99\):

| Statistic | Definition |
|-----------|------------|
| Cohort mean | \(\text{mean}_c(R^{\sigma}_{c,b})\) |
| Cohort median | \(\text{median}_c(R^{\sigma}_{c,b})\) |
| 95% CI | Percentile bootstrap on \(\{R^{\sigma}_{c,b}\}\), 10,000 resamples, fixed seed |

### Paired comparison (DA-MSE vs MSE)

Per band \(b\), paired bootstrap on \(\Delta R^{\sigma}_{c,b} = R^{\sigma,\,\text{DA-MSE}}_{c,b} - R^{\sigma,\,\text{MSE}}_{c,b}\):

- Mean, median, 95% CI
- Fraction \(\Delta > 0\)

**Descriptive only** — not a success criterion.

### Outputs

| Artifact | Path (per experiment run) |
|----------|---------------------------|
| Per-container table | `evaluation/horizon_variance_recovery_per_container.csv` |
| Cohort summary | `evaluation/horizon_variance_recovery_cohort.csv` |
| Paired comparison | `comparisons/horizon_variance_recovery_paired_bootstrap.json` |
| Line plot (mean ± CI by band) | `plots/horizon_variance_recovery_mean_ci.{png,pdf}` |
| Boxplot by band | `plots/horizon_variance_recovery_boxplot.{png,pdf}` |
| MSE vs DA-MSE overlay | `plots/horizon_variance_recovery_mse_vs_da_mse.{png,pdf}` |

**CSV schema (per-container):**

`container_id, band, sigma_actual, sigma_predicted, std_ratio, arm`

**CSV schema (cohort):**

`arm, band, n, mean_std_ratio, median_std_ratio, ci_low, ci_high`

### Interpretation guide

| Pattern | Interpretation |
|---------|----------------|
| Std ratio ↓ monotonically with band index (both arms) | Progressive horizon collapse — error accumulation or autoregressive flattening |
| Std ratio ≈ flat across bands | Global amplitude suppression, not horizon-specific |
| DA-MSE ↑ vs MSE uniformly across bands | Objective effect on dispersion is **horizon-general** |
| DA-MSE ↑ only in early bands | Short-horizon dispersion recovery only — limited objective benefit at long lags |
| Late bands → NaN flags increase | Low residual variance at long horizons in subset of containers — report separately |

---

## 2. Residual energy recovery

### Motivation

Std ratio captures **relative amplitude**; **energy recovery** quantifies how much **squared residual mass** the model reproduces:

\[
\text{Energy} = \sum_{h=1}^{96} r_h^2
\]

Complements std ratio: two sequences can share std ratio but differ in energy if sample size or outliers differ (here band length varies in horizon analysis; full-trajectory energy uses all 96 steps).

### Definition

For container \(c\), full Day-1 trajectory:

\[
E_y = \sum_{h=1}^{96} y_h^2, \quad E_{\hat{y}} = \sum_{h=1}^{96} \hat{y}_h^2
\]

\[
\text{ERR}_c = \frac{E_{\hat{y}}}{E_y}
\]

**Energy Recovery Ratio (ERR)** — values near 0 indicate near-zero predicted energy (severe collapse); 1.0 indicates matched energy; >1.0 indicates over-energetic predictions.

### Per-container and cohort

| Statistic | Scope |
|-----------|-------|
| ERR per container | Both arms |
| Cohort mean, median | Both arms |
| 95% bootstrap CI | Cohort mean ERR per arm |
| Paired ΔERR | DA-MSE − MSE, bootstrap CI |

### Horizon-band energy (optional secondary table)

Compute band-wise energy ratio using the same five bands as Section 1:

\[
\text{ERR}_{c,b} = \frac{\sum_{h \in b} \hat{y}_h^2}{\sum_{h \in b} y_h^2}
\]

Report in `evaluation/horizon_energy_recovery_per_container.csv` — same interpretability as horizon std ratio.

### Outputs

| Artifact | Path |
|----------|------|
| Per-container ERR | `evaluation/energy_recovery_per_container.csv` |
| Cohort summary | `evaluation/energy_recovery_cohort.csv` |
| Paired bootstrap | `comparisons/energy_recovery_paired_bootstrap.json` |
| Distribution histogram | `plots/energy_recovery_distribution.{png,pdf}` |
| MSE vs DA-MSE scatter | `plots/energy_recovery_mse_vs_da_mse_scatter.{png,pdf}` |
| Horizon-band ERR (optional) | `evaluation/horizon_energy_recovery_cohort.csv` |

**CSV schema (per-container):**

`container_id, energy_actual, energy_predicted, energy_recovery_ratio, arm`

### Interpretation guide

| Pattern | Interpretation |
|---------|----------------|
| ERR ≪ 1 (MSE arm) | Confirms variance collapse in energy terms |
| ERR(DA-MSE) > ERR(MSE) with stable MAE | Model reproduces more signal energy without large point-error penalty |
| ERR ↑ but r unchanged | Amplitude gain without shape alignment |
| ERR > 1 for some containers | Over-energetic predictions — check MAE guardrail on those containers |

**ERR is not an optimization objective** and must not be used for model selection or hypothesis adjudication.

---

## 3. Comparison between MSE and DA-MSE (diagnostic)

All diagnostic comparisons are **side-by-side descriptive**:

| View | Content |
|------|---------|
| Horizon std ratio | Line plot: cohort mean ± 95% CI by band, both arms |
| Horizon std ratio | Paired Δ by band (bar + CI) |
| Energy recovery | Overlaid histograms or KDE |
| Energy recovery | Paired scatter (MSE ERR vs DA-MSE ERR) |
| Joint | Table correlating Δ std ratio (full) vs Δ ERR |

Primary hypothesis decision remains on **full-trajectory** std ratio and Pearson r (see success_failure_criteria.md).

---

## 4. Notebook sections (implementation phase)

When implementation begins, `notebooks/lfhe_visualization.ipynb` must include:

| Section | Content |
|---------|---------|
| Horizon-wise variance recovery | Load cohort CSV; interactive line/box plots; band table |
| Residual energy recovery | Distribution + scatter; cohort summary |
| Horizon collapse interpretation | Markdown + computed pattern (monotonic vs flat) |
| MSE vs DA-MSE comparison | Overlay plots for both diagnostics |

**Figure policy:** All figures render **inline** in the notebook **and** export to `experiments/loss_function_hypothesis_<timestamp>/plots/` as **PNG and PDF** (publication quality per project visualization standards: titles, axis labels, legends, readable fonts).

---

## 5. Explicit exclusions

| Use | Allowed? |
|-----|----------|
| Hypothesis accept/reject | **No** |
| Early stopping | **No** |
| Loss function / λ tuning | **No** |
| Model selection between arms | **No** |
| Thesis discussion / interpretation | **Yes** |
| Supplementary figures | **Yes** |
