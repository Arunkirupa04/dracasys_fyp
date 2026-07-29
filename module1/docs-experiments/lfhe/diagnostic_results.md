# LFHE Diagnostic Results

**Descriptive only** — not used for hypothesis accept/reject.

## A. Horizon-wise variance recovery

Bands: 1–12, 13–24, 25–48, 49–72, 73–96.

### MSE arm (cohort mean std ratio by band)

| Band | Mean std ratio |
|------|----------------|
| h01_12 | 0.194 |
| h13_24 | 0.151 |
| h25_48 | 0.073 |
| h49_72 | 0.105 |
| h73_96 | 0.181 |

### DA-MSE arm (cohort mean std ratio by band)

| Band | Mean std ratio |
|------|----------------|
| h01_12 | 2.431 |
| h13_24 | 1.995 |
| h25_48 | 0.826 |
| h49_72 | 1.020 |
| h73_96 | 1.795 |

### Interpretation

- MSE shows **non-monotonic** horizon pattern (dip at 25–48, recovery at 73–96) — not uniformly progressive collapse.
- DA-MSE **over-restores** dispersion at all bands (many band ratios >> 1.0), consistent with one-sided penalty allowing over-dispersion.

Artifacts: `diagnostics/horizon_variance_recovery_*.csv`, `plots/horizon_variance_recovery_*.{png,pdf}`

## B. Residual energy recovery

ERR = Σ(ŷ²) / Σ(y²)

| Arm | Cohort mean ERR |
|-----|-----------------|
| MSE | 0.011 |
| DA-MSE | 0.823 |

Paired bootstrap: DA-MSE > MSE on ERR for all 99 containers.

Artifacts: `diagnostics/energy_recovery_*.csv`, `plots/energy_recovery_*.{png,pdf}`

## C. Variance collapse interpretation

| Quantity | Value |
|----------|-------|
| Absolute std ratio improvement (DA−MSE) | +0.726 |
| Relative improvement | ~721% |
| Remaining gap to Ridge reference (0.22) | DA-MSE **exceeds** Ridge (over-dispersed) |

See `diagnostics/variance_collapse_interpretation.json`.
