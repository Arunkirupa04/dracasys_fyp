# Phase 1 — Freeze the Baseline (Complete)

[← Overview](README.md)

---

## 2. Phase 1 — Freeze the Baseline (Complete)

### 2.1 Objective

Establish an immutable control model and control metrics for all subsequent Peak-Aware comparisons.

### 2.2 Methodology

The production Hybrid model (`models/hybrid_gru.keras`) was copied to a dedicated reference folder. A fresh Day 1 evaluation was executed on 99 evaluable containers using the frozen evaluation pipeline. Full configuration and verification results were recorded.

### 2.3 Assumptions

- The Hybrid implementation in `utils/hybrid_*.py` and `notebooks/hybrid_model.ipynb` is finalized.
- `input_window = 96` is locked following the H1.2 ablation recommendation.
- Prophet models are not persisted; they are refit per container at inference.

### 2.4 Control Model

| Parameter | Value |
|-----------|-------|
| Architecture | Hybrid Prophet + GRU |
| Version | `hybrid_prophet_gru_v1` |
| Input window | 96 timesteps (24 h) |
| Forecast horizon (Day 1) | 96 timesteps (24 h) |
| Prophet | `daily_seasonality=True`, `weekly_seasonality=False` |
| GRU target | Globally normalized Prophet residuals |

### 2.5 Control Metrics (Day 1, real CPU %, 99 containers)

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 1.745882 | 2.486804 |
| RMSE | 2.387813 | 3.219141 |
| MAPE | 111.940318 | 615.299100 |

- **Skipped container:** `c_14674` (missing from frozen train/val data)

### 2.6 Outputs

| Artifact | Location |
|----------|----------|
| Reference model | `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` |
| Residual statistics | `experiments/baseline_reference_2026-07-14/models/residual_stats.pkl` |
| Per-container evaluation | `experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv` |
| Aggregate evaluation | `experiments/baseline_reference_2026-07-14/evaluation/evaluation_summary.csv` |
| Configuration record | `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json` |
| Verification log | `experiments/baseline_reference_2026-07-14/verification/verification_notes.md` |
| Freeze statement | `experiments/baseline_reference_2026-07-14/FREEZE_STATEMENT.md` |

### 2.7 Verification

All five baseline verification scripts passed on 2026-07-14.

### 2.8 Conclusion

Phase 1 is complete. The baseline is frozen and serves as the permanent control for Peak-Aware research.
