# RRAA Verification

## Task 1 — Residual definition

| Check | Result |
|-------|--------|
| Sign | **Correct:** `remaining = prophet − ridge` (no inversion) |
| Identity (element-wise) | **Holds** for both original and corrected formulas |
| Scaling | **BUG in RRA v1.0:** raw Prophet residual minus z-scored Ridge prediction |
| Same space for subtraction | **Failed in RRA; passes after correction** |

Evidence: `verification/task1_scaling_mismatch.csv` — for all 99 containers, `mean_abs_diff_raw_vs_z_pred` > `mean_abs_diff_raw_vs_unzscored_pred`.

Example (c_10032): prophet std = 0.111, ridge pred z std = 0.101, ridge pred unzscored std = 0.011.

## Task 2 — Alignment

| Check | Result |
|-------|--------|
| Day-1 input window | Last 96 train z-scored residuals |
| Day-1 target | First 96 val z-scored residuals |
| Off-by-one | **None** for first block (matches Hybrid day-1 convention) |
| Multi-block extension | RRA uses non-overlapping 96-step blocks; history extended with **actual** val residuals (correct in z-space after fix) |

Sample: `verification/task2_day1_alignment_sample.json` (c_10032).

## Task 3 — Data splits

| Check | Result |
|-------|--------|
| Ridge training data | Train B0-LC only |
| Val in Ridge train | **0 rows** |
| Train/val timestamp overlap | **0** |
| Leakage | **None detected** |

## Task 4 — ACF computation

| Check | Result |
|-------|--------|
| Same implementation | `utils.csrle.diagnostics.avg_abs_acf` lags 1–10, fft=True |
| RRA CSV replication | Matches reported cohort means exactly |
| Corrected remaining | Recomputed on z-space remaining only |

| Metric | Prophet | RRA remaining | Corrected remaining |
|--------|---------|---------------|---------------------|
| Mean \|ACF\| 1–10 | 0.199 | 0.266 | **0.201** |
| Ljung reject lag 20 | 75.8% | 96.0% | **73.7%** |

**Conclusion:** ACF/Ljung increase in RRA is **not reproducible** after scaling correction.
