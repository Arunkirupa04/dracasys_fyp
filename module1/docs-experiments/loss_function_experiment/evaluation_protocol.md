# Evaluation Protocol

## Principle

**Primary evaluation identical to CSRLE Stage 2** residual-learning and CPU evaluation on the full 96-step trajectory. LFHE compares MSE vs DA-MSE arms on **pre-specified primary metrics** (std ratio, Pearson r, MAE guardrail) — unchanged from v1.0.

**v1.1 addition:** [diagnostic_analysis.md](diagnostic_analysis.md) defines **descriptive-only** horizon-wise variance recovery and residual energy recovery. These **do not** enter hypothesis accept/reject or model selection.

## Evaluation phases

### Phase 0 — Control reproduction gate

Before treatment analysis, LFHE-MSE-B0 must pass reproduction vs frozen Stage 2 B0 (see methodology.md).

### Phase 1 — Within-trajectory residual metrics (primary)

Computed per container on Day-1 validation (96 steps), **scaled residual space** unless noted.

| Metric | Symbol | Definition | CSRLE source |
|--------|--------|------------|--------------|
| Pearson r | `residual_pearson_r` | corr(y_actual, ŷ_gru) within trajectory | `extended_metrics.csv` |
| R² | `residual_r2` | 1 − SS_res/SS_tot | same |
| MAE | `residual_mae` | mean \|y − ŷ\| | same |
| Std ratio | `residual_std_ratio` | std(ŷ) / std(y) | same |
| Horizon-band r | `r_band_*` | r within horizon bands 1–4, 5–12, … | Stage 2 horizon tables |

### Phase 2 — Cross-horizon cohort correlation

Per horizon h ∈ {1,…,96}: Pearson r between stacked container predictions and actuals at step h (`cross_horizon_correlation.csv` protocol).

### Phase 3 — CPU-level metrics (secondary / exploratory)

| Metric | Definition |
|--------|------------|
| `prophet_day1_mae` | MAE(Prophet, actual CPU %) |
| `hybrid_day1_mae` | MAE(Prophet + GRU, actual CPU %) |
| `hybrid_minus_prophet_mae` | Δ MAE |

**Not used for hypothesis accept/reject** — CSRLE showed these are insensitive; reported for completeness.

### Phase 4 — Baselines (B0 only, descriptive)

Recompute **Ridge, zero, persistence** from frozen Stage 2 protocol on same predictions — **no retraining**. Serves as static reference lines on plots (Ridge std ratio ≈ 0.22).

Do **not** retrain Ridge inside LFHE.

### Phase 5 — Diagnostic analyses (descriptive only)

See [diagnostic_analysis.md](diagnostic_analysis.md). **Not used for hypothesis decision.**

| Diagnostic | Summary |
|------------|---------|
| Horizon-wise variance recovery | Std ratio in bands 1–12, 13–24, 25–48, 49–72, 73–96 |
| Residual energy recovery | ERR = Σ(ŷ²) / Σ(y²) per container |

## Variance collapse quantification

### Primary index

**Within-trajectory std ratio** (per container):

\[
\text{std\_ratio}_c = \frac{\text{std}(\hat{\mathbf{r}}_c)}{\text{std}(\mathbf{r}_c)}
\]

where **r** is the scaled residual vector over 96 validation steps.

### Cohort summaries

| Statistic | Description |
|-----------|-------------|
| Mean std ratio | Primary dispersion index |
| Median std ratio | Robust central tendency |
| frac(std_ratio > 0.15) | Fraction achieving meaningful amplitude |
| frac(std_ratio > MSE arm) | Per-container win rate (paired) |

### Secondary dispersion diagnostics

| Diagnostic | Purpose |
|------------|---------|
| Mean(\|ŷ\|) / Mean(\|y\|) | Amplitude ratio (supplementary) |
| var(ŷ) / var(y) | Equivalent to std ratio squared |
| Horizon-band std ratio (bands 1–12 … 73–96) | **Descriptive** — progressive collapse pattern; see diagnostic_analysis.md |
| Energy recovery ratio (ERR) | **Descriptive** — squared-signal mass reproduced |

### Reference benchmarks (frozen, not targets)

| Source | Cohort mean std ratio |
|--------|----------------------|
| Stage 2 GRU B0 (MSE) | 0.073 |
| Ridge B0 (analysis) | 0.220 |
| Condition A GRU | 0.050 |

**Movement of DA-MSE toward 0.15–0.22 band** supports hypothesis; **remaining at 0.07** rejects.

## Correlation improvement measurement

### Primary

**Within-trajectory Pearson r** per container — same as CSRLE.

### Paired comparison (DA-MSE − MSE)

For each container c:

\[
\Delta r_c = r_c^{\text{DA-MSE}} - r_c^{\text{MSE}}
\]

**Cohort inference:** Paired bootstrap (10,000 resamples, seed fixed) on {Δr_c}, report mean, median, 95% CI, fraction Δr_c > 0.

Mirror Stage 2 `paired_bootstrap_results.json` format for B0 vs C0 comparisons.

### Secondary correlation views

- Horizon-band cohort mean r (bands 1–4, 5–12, 13–24, 25–48, 49–96)
- Cross-horizon cohort r at h = 1, 4, 12, 24, 48, 96
- R² (expect negative for both; directional change only)

**Do not** use R² as primary success metric (CSRLE established persistent negativity).

## Statistical tests

| Comparison | Method | Primary? |
|------------|--------|----------|
| Δ std ratio (paired) | Bootstrap 95% CI | **Yes** |
| Δ Pearson r (paired) | Bootstrap 95% CI | **Yes** |
| Δ MAE (paired) | Bootstrap 95% CI | Guardrail |
| Independent t-test | Not primary | Avoid — use bootstrap consistent with CSRLE |

## Reporting tables (minimum)

1. Cohort summary: MSE vs DA-MSE (r, std ratio, MAE, R²) — side by side.
2. Paired bootstrap: Δr, Δ std ratio with CIs.
3. Per-container scatter: MSE r vs DA-MSE r; MSE std ratio vs DA-MSE std ratio.
4. Training: best epoch, final train/val loss curves per arm.
5. Reference lines: Stage 2 Ridge r and std ratio (horizontal/ diagonal guides).
6. **Diagnostic:** Horizon variance recovery cohort CSV + plots (descriptive).
7. **Diagnostic:** Energy recovery cohort CSV + distribution plots (descriptive).

## Evaluation integrity checks

| Check | Criterion |
|-------|-----------|
| Container count | 99 both arms |
| No train/val leakage | Train scalers only |
| Prediction alignment | 96 steps, same timestamps as Stage 2 |
| Metric reproduction | MSE arm ≈ Stage 2 B0 |

## Answers to design checklist (evaluation)

| # | Question | Answer |
|---|----------|--------|
| 11 | Success metrics | Δ std ratio > 0 (CI excludes 0), mean std ratio ≥ 0.12, Δr > 0 (CI excludes 0), MAE guardrail |
| 12 | Failure metrics | std ratio ≈ MSE (CI includes 0), r ≈ MSE, or MAE guardrail violated |
| 13 | Variance collapse quantification | Full-trajectory std ratio (primary); horizon-band std ratio + ERR (diagnostic) |
| 14 | Correlation measurement | Within-trajectory Pearson r; paired Δr bootstrap |
| 15–17 | Outcomes | See success_failure_criteria.md |
