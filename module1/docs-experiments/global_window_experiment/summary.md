# GGTCE Summary

**Authoritative run:** `experiments/ggtce_2026-07-26_190616`

## Headline finding

**G96 remains the best Global GRU configuration.** Extending the input window to 192 or 288 steps did not improve cohort Day-1 MAE. G192 degraded forecasting (+0.331 pp mean MAE); G288 partially recovered but still underperformed G96 (+0.175 pp).

## Cohort Day-1 MAE

| Variant | Mean MAE | Mean Pearson r | Sequences |
|---------|----------|----------------|-----------|
| **G96** | **1.924** | **0.416** | 41,650 |
| G192 | 2.255 | 0.278 | 32,146 |
| G288 | 2.099 | 0.377 | 22,642 |

## Key conclusions

1. Longer input windows **do not** improve Global GRU forecasting on this cohort.
2. G96 replication matched frozen baseline exactly (Δ MAE = 0.0).
3. G96→G288 degradation is **statistically significant** (bootstrap CI excludes zero).
4. Memory-aware analysis found **no evidence** that high-memory workloads benefit more (Pearson r = −0.14, p = 0.17).
5. **Keep G96** as the production Global GRU baseline.

Full answers: `experiments/ggtce_2026-07-26_190616/reports/final_report.md`
