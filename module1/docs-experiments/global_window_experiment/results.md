# GGTCE Results

**Run:** `experiments/ggtce_2026-07-26_190616`

## Cohort metrics

| variant_id | n | mean_day1_mae | mean_day1_rmse | mean_forecast_pearson_r | mean_forecast_std_ratio |
|------------|---|---------------|----------------|-------------------------|-------------------------|
| g96 | 99 | 1.924 | 2.611 | 0.416 | 0.517 |
| g192 | 99 | 2.255 | 2.947 | 0.278 | 0.498 |
| g288 | 99 | 2.099 | 2.843 | 0.377 | 0.537 |

## Window transitions (G96 reference)

| Transition | Mean Δ MAE | Interpretation |
|------------|------------|----------------|
| G96 → G192 | +0.331 | Substantial degradation |
| G192 → G288 | −0.156 | Partial recovery (still worse than G96) |
| G96 → G288 | +0.175 | Significant degradation |

## G96 replication gate

PASSED — identical mean MAE (1.924) and Spearman rank correlation ≈ 1.0 vs frozen baseline `experiments/global_gru_baseline_2026-07-17_121748`.

## Memory-aware co-primary analysis

- Memory score vs Δ MAE (G96→G288): Pearson r = −0.14 (p = 0.17, not significant)
- High-memory subgroup mean Δ MAE = +0.250 (worse with G288)
- Low-memory subgroup mean Δ MAE = +0.026 (near neutral)
- **No support** for the hypothesis that high-memory workloads gain more from longer windows

## Artifacts

- Plots: `experiments/ggtce_2026-07-26_190616/plots/` (PNG + PDF)
- Per-container deltas: `window_comparison/deltas_*.csv`
- Case studies: `case_studies/case_studies.json`
- Final report: `reports/final_report.md`, `reports/final_report.json`
