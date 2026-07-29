# RRA Diagnostics

## Per-container metrics

### Ridge performance

| Column | Definition |
|--------|------------|
| `ridge_pearson_r` | Pearson r(prophet residual, ridge prediction) on validation |
| `ridge_mae_scaled` | MAE in scaled residual space |
| `remaining_std_ratio` | std(remaining) / std(prophet residual) |
| `var_reduction_fraction` | 1 − var(remaining)/var(prophet) |

### Temporal structure (`temporal_structure_per_container.csv`)

| Column | Definition |
|--------|------------|
| `prophet_avg_abs_acf_1_10` | Mean \|ACF(l)\|, l=1..10 on Prophet val residual |
| `remaining_avg_abs_acf_1_10` | Same on Ridge remaining |
| `acf_reduction` | prophet − remaining (positive = Ridge removed ACF) |
| `prophet_lb_reject_lag_20` | 1 if Ljung–Box p < 0.05 at lag 20 |
| `remaining_lb_reject_lag_20` | Same on remaining |
| `prophet_dominant_freq` | Argmax power (excl. DC) on Prophet residual |
| `remaining_dominant_freq` | Same on remaining |

## Cohort summaries (`white_noise_comparison.json`)

- Mean |ACF| and |PACF| lags 1–10 (Prophet vs remaining)
- Ljung–Box reject fractions (lag 20)
- Bootstrap 95% CI for paired ACF/PACF reduction (2000 resamples, seed 12345)

## Case study selection

Automatic by validation `ridge_pearson_r`:

- **Best:** max r
- **Median:** closest to cohort median r
- **Worst:** min r

Authoritative run: best `c_15179`, median `c_10032`, worst `c_12231`.

## Justification thresholds (pre-registered in `report.py`)

| Question | Criterion |
|----------|-----------|
| Q1 Ridge removes structure | ACF reduction > 0.01 OR Ljung reject reduction > 5 pp |
| Q2 Closer to white noise | remaining ACF < prophet AND remaining Ljung < prophet |
| Q3 Still rejects white noise | remaining Ljung reject fraction > 30% |
| Q4 Significant ACF remains | remaining mean \|ACF\| 1–10 > 0.05 |
| Q5 Nonlinear justified | Q3 OR Q4 |
| Q6 Ridge→GRU future | Same as Q5 |

## Plots

| File | Content |
|------|---------|
| `residual_distribution_comparison` | Pooled histogram Prophet vs remaining |
| `acf_boxplot_comparison` | Per-container \|ACF\| 1–10 |
| `pacf_boxplot_comparison` | Per-container \|PACF\| 1–10 |
| `cohort_acf_paired` | Cohort mean bar chart |
| `case_study_{best,median,worst}_*` | Four-panel case studies |
| `acf_*`, `pacf_*`, `fft_*` | Median container detail plots |
