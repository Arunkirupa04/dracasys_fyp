# Results

**Authoritative run:** `experiments/hcerl_2026-07-26_180649/`

## Cohort comparison (99 containers)

| Variant | Mean Day-1 MAE | Mean RMSE | Mean Pearson r | Mean std ratio | Mean energy recovery |
|---------|----------------|-----------|----------------|----------------|----------------------|
| **V0** (baseline) | **1.732** | **2.372** | **0.062** | 0.043 | 0.002 |
| V1 (+ prophet) | 1.740 | 2.380 | 0.059 | 0.066 | 0.005 |
| V2 (+ vol) | 1.746 | 2.387 | 0.046 | 0.065 | 0.005 |
| V3 (+ cpu_std) | 1.757 | 2.398 | 0.022 | 0.078 | 0.007 |
| V4 (+ peak) | 1.752 | 2.388 | 0.041 | 0.063 | 0.007 |

**Best CPU MAE:** V0 baseline. **Best Pearson r:** V0 baseline.

## Ablation transitions (mean Δ, right − left)

| Transition | Added feature | Δ MAE | Δ Pearson r | Δ std ratio | Significant? |
|------------|---------------|-------|-------------|-------------|--------------|
| V0→V1 | prophet_yhat_scaled | +0.008 | −0.004 | +0.023 | No |
| V1→V2 | res_roll_std_12 | +0.006 | −0.013 | −0.001 | No |
| V2→V3 | cpu_std | +0.011 | −0.024 | +0.013 | No |
| V3→V4 | is_peak_p90 | **−0.005** | **+0.020** | −0.015 | No |

No transition reached bootstrap significance at 95% for MAE or Pearson r.

## Verdict

**Keep V0 baseline.** Context channels do not improve Day-1 CPU MAE; additional features tend to **hurt** residual correlation. Peak flag (V3→V4) shows the only MAE improvement but is not statistically significant and does not recover Pearson r vs V0.

Full report: `experiments/hcerl_2026-07-26_180649/reports/final_report.md`
