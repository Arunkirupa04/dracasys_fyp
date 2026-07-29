# Evaluation Protocol

## Scope

Everything identical to Global GRU baseline **except** `input_window` at train and inference time.

## Primary metrics (frozen — real CPU %)

| Metric | Definition | Direction |
|--------|------------|-----------|
| Day-1 MAE | Mean absolute error over 96 validation steps | Lower better |
| Day-1 RMSE | Root mean squared error | Lower better |
| Day-1 MAPE | Epsilon-floor MAPE (ε = 0.01 pp) | Lower better |

**Aggregation:** Per-container → cohort mean, std, min, max (match `summarize_evaluation_metrics` pattern).

## Per-container outputs

- `evaluation_df.csv` — one row per container per variant
- `inference_cache.pkl` — `{container_id: {actual_day1_real, day1_final_real}}`
- Failures / skipped lists (same rules as baseline: missing scaler, insufficient train steps)

## Statistical comparisons (paired, per-container)

For each transition (G96→G192, G192→G288, G96→G288):

| Statistic | Specification |
|-----------|---------------|
| Mean Δ MAE | `MAE_variant - MAE_G96` |
| Bootstrap 95% CI | Paired container bootstrap, seed **12345**, 10,000 resamples |
| Wilcoxon signed-rank | Two-sided, α = 0.05 |
| Cohen's d | Mean(Δ) / SD(Δ) on paired differences |
| Fraction improved | % containers with Δ MAE < 0 |

Apply same framework to RMSE; report Pearson correlation of errors only if needed for diagnostics.

## TMA-linked analyses (recommended co-primary)

Import **read-only** per-container TMA statistics from  
`experiments/temporal_memory_analysis_2026-07-26_170052/tables/` (container-level CPU memory table):

| TMA metric | GGTCE use |
|------------|-----------|
| `pct_capture_within_96` | Predict who benefits from G192/G288 |
| `lag96_acf`, `lag192_acf` | Long-memory severity |
| `integrated_autocorr_time` | Memory length proxy |
| Memory tertile (low/med/high) | **Subgroup analysis** |

**Analyses:**

1. Scatter: TMA `capture_96` vs Δ MAE (G96 − G288)
2. Subgroup cohort means: low / medium / high memory containers
3. Test interaction: improvement concentrated in high-memory tertile?

## Additional recommended analyses

| Analysis | Rationale |
|----------|-----------|
| Horizon-band MAE (1–12, 13–24, …, 73–96) | Long window may help early vs late Day-1 differently |
| Training curve comparison | Overfitting diagnostic (val_loss divergence) |
| Sequence-count normalised metrics | Rule out trivial sample-size artefacts |
| G96 replication gate | Confirm G96 matches frozen baseline within tolerance |

## G96 replication tolerance (recommended)

Compare GGTCE G96 to `experiments/global_gru_baseline_2026-07-17_121748/`:

| Check | Tolerance |
|-------|-----------|
| Mean MAE delta | ≤ 0.05 pp (retrain variance) |
| Per-container MAE rank correlation | ≥ 0.95 |

If G96 fails gate, diagnose before interpreting G192/G288.

## Optional (not required for primary verdict)

- Peak-subset MAE using frozen `peak_thresholds.pkl`
- Comparison table vs Hybrid baseline (read-only reference — not GGTCE primary)

## Reporting artifacts

```
experiments/ggtce_{timestamp}/
  variants/g96|g192|g288/
    evaluation/evaluation_df.csv
    evaluation/cohort_summary.json
  ablation/paired_tests.json
  ablation/subgroup_analysis.json
  plots/  (PNG + PDF)
  reports/final_report.md
  reports/final_report.json
```
