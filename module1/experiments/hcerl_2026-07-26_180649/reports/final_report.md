# HCERL Final Report

**Protocol:** hcerl_v1.0
**Timestamp:** 2026-07-26_180649

## Recommendation

**v0_baseline** — V0 — Residual only (96×1) achieves near-best cohort MAE (1.732) with the fewest context channels among variants within 0.01 pp of the best.

## Explicit Answers

### Does contextual information improve Hybrid forecasting?

**No clear improvement**

### Which contextual feature contributes the most?

**is_peak_p90**

### Which contextual feature contributes the least?

**cpu_std**

### Is Prophet forecast context more useful than statistical context?

**Prophet level context**

### Do peak-aware features improve residual learning?

**Yes**

### Are gains practically meaningful?

**No (MAE Δ=0.00%)**

### Should Hybrid v2 replace the baseline model?

**Keep baseline — simpler model preferred**

## Cohort Comparison

```
   variant_id  n  mean_day1_mae  mean_day1_rmse  mean_day1_mape  mean_peak_mae  mean_peak_rmse  mean_residual_mae_scaled  mean_residual_rmse_scaled  mean_residual_pearson_r  mean_residual_r2  mean_residual_std_ratio  mean_energy_recovery_ratio
  v0_baseline 99       1.732136        2.372467      112.135833       3.309097        3.921800                  0.094368                   0.139422                 0.062456         -0.341932                 0.043290                    0.002100
   v1_prophet 99       1.740448        2.379958      112.009513       3.317670        3.926186                  0.094517                   0.139551                 0.058526         -0.337480                 0.066205                    0.004923
v2_volatility 99       1.745970        2.386703      112.463413       3.324981        3.937264                  0.094655                   0.139771                 0.045845         -0.368952                 0.065377                    0.004850
   v3_cpu_std 99       1.757092        2.398176      111.869813       3.332375        3.947194                  0.095023                   0.140192                 0.021572         -0.348996                 0.077927                    0.007498
      v4_full 99       1.751930        2.387844      113.534455       3.282644        3.896924                  0.094879                   0.139797                 0.041081         -0.352386                 0.063195                    0.007195
```

## Feature Contribution Ranking

```
                 transition       added_feature  delta_mae_mean  delta_rmse_mean  delta_pearson_r_mean  delta_std_ratio_mean  delta_r2_mean  mae_improved_fraction  pearson_improved_fraction  mae_significant  pearson_significant  rank_by_composite
      v3_cpu_std -> v4_full         is_peak_p90       -0.005162        -0.010332              0.019508             -0.014732      -0.003389               0.444444                   0.494949            False                False                  1
  v0_baseline -> v1_prophet prophet_yhat_scaled        0.008311         0.007492             -0.003930              0.022915       0.004452               0.474747                   0.515152            False                False                  2
v1_prophet -> v2_volatility     res_roll_std_12        0.005522         0.006745             -0.012680             -0.000828      -0.031472               0.424242                   0.484848            False                False                  3
v2_volatility -> v3_cpu_std             cpu_std        0.011122         0.011472             -0.024273              0.012550       0.019956               0.363636                   0.424242            False                False                  4
```