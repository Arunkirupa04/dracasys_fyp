# Ablation Protocol

## Ladder design

Cumulative channels — each step adds exactly one feature family:

```
V0: residual
V1: V0 + prophet level
V2: V1 + volatility
V3: V2 + container scale
V4: V3 + peak regime
```

## Transitions analyzed

| Transition | Added feature |
|------------|---------------|
| V0 → V1 | `prophet_yhat_scaled` |
| V1 → V2 | `res_roll_std_12` |
| V2 → V3 | `cpu_std` |
| V3 → V4 | `is_peak_p90` |

## Delta metrics (per container)

For each transition, compute `metric(V_next) - metric(V_prev)` for:

- `day1_mae`, `day1_rmse`
- `residual_pearson_r`, `residual_r2`, `residual_std_ratio`
- `energy_recovery_ratio`

## Significance

| Metric direction | Improvement when |
|------------------|------------------|
| MAE, RMSE | Δ < 0 |
| Pearson r, R², std ratio, energy | Δ > 0 |

Bootstrap CI excluding zero → `significant_bootstrap_95`  
Wilcoxon p < 0.05 → `significant_wilcoxon_005`

## Parsimony rule

Recommend the **simplest variant** within +0.01 pp cohort MAE of the best MAE.
