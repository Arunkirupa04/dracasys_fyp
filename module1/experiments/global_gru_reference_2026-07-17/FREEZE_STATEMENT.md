# Global GRU Baseline Freeze Statement

> **SUPERSEDED (2026-07-17):** This reference used `EarlyStopping(patience=3)`, an implementation oversight. The official Global GRU baseline is now `experiments/global_gru_baseline_2026-07-17_121748/` (patience=10). This directory is preserved as a historical artifact and must not be used as the active control reference.

**Baseline name:** Global GRU v1 (`global_gru_v1`)  
**Implementation freeze date:** 2026-07-17  
**Evaluation freeze date:** 2026-07-17  
**Methodology comparison date:** 2026-07-17  
**Status:** FROZEN — implementation, training, Phase 5 cohort evaluation, and Phase 6 methodology comparison complete

---

## Experiment Identifiers

| Role | Identifier |
|------|------------|
| Source training run | `global_gru_baseline_2026-07-17_074743` |
| Source evaluation run | `global_gru_evaluation_2026-07-17_092120` |
| Immutable reference | `experiments/global_gru_reference_2026-07-17/` |
| Methodology comparison | `experiments/hybrid_vs_global_2026-07-17_095147/` |

This reference folder captures the verified Phase 4 implementation, trained model, and **official Phase 5 cohort evaluation** under the shared 99-container Day 1 protocol. Phase 6 produced a read-only Hybrid vs Global GRU comparison saved under the comparison experiment directory above.

---

## Specification Authority

[Phase 0.5 — Global GRU Baseline Specification](../../docs/global-gru-baseline/phase-03-5-baseline-specification.md)

---

## Random Seed

**42** (Python, NumPy, TensorFlow)

---

## Architecture

```
GRU(128, return_sequences=True)
Dropout(0.2)
GRU(64)
Dense(64, ReLU)
Dense(96)
```

---

## Input Features

| Feature | Role |
|---------|------|
| `cpu_scaled` | Input feature and prediction target |
| `cpu_mean` | Static train-period container context |
| `cpu_std` | Static train-period container context |

---

## Forecast Horizon

- **Input window:** 96 timesteps (24 hours)
- **Forecast horizon:** 96 timesteps (Day 1 primary)

---

## Training Methodology

| Item | Value |
|------|-------|
| Sequence source | `global_train` only — never `concat(global_train, global_val)` |
| Loss | MSE |
| Optimizer | Adam |
| Training metric | MAE |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | `val_loss`, patience 3, `restore_best_weights=True` |
| Shuffle | `False` |
| Training val split | 80/20 index on pooled train sequences (early stopping only) |
| Epochs run | 9 |

---

## Official Cohort Evaluation (Phase 5 — Frozen)

| Item | Value |
|------|-------|
| Selected containers | 100 |
| Evaluated containers | **99** |
| Skipped containers | **1** (`c_14674` — missing from frozen train/val data) |
| Metric unit | Real CPU % |
| MAPE epsilon | 0.01 |

### Aggregate Day 1 metrics

| Metric | Mean | Std |
|--------|------|-----|
| MAE | 2.0627 | 2.4701 |
| RMSE | 2.7559 | 3.1039 |
| MAPE | 118.9320 | 659.0400 |

**Source:** `evaluation/evaluation_summary.csv` in this reference folder.

**Evaluation immutability:** These evaluation files must not be overwritten. Re-runs require a new `global_gru_evaluation_*` directory.

---

## Methodology Comparison (Phase 6 — Complete)

Read-only comparison against frozen Hybrid Prophet + GRU control (`experiments/baseline_reference_2026-07-14/`).

| Methodology | MAE (mean) | RMSE (mean) |
|-------------|------------|-------------|
| Hybrid Prophet + GRU | 1.7459 | 2.3878 |
| Global GRU v1 | 2.0627 | 2.7559 |
| **Δ (Global − Hybrid)** | **+0.3168** | **+0.3681** |

**Per-container MAE:** Global better on 30 containers; Hybrid better on 69.

**Verification:** `scripts/verify_hybrid_vs_global_comparison.py` — **PASS**

**Executive summary:** `experiments/hybrid_vs_global_2026-07-17_095147/FINAL_COMPARISON_SUMMARY.md`

**Scope:** Conclusions apply only to the implemented configurations in this study.

---

## Production Artifact Locations

| Artifact | Path |
|----------|------|
| Model | `models/global_gru.keras` |
| Metadata | `models/global_gru_metadata.json` |
| Reference copy | `experiments/global_gru_reference_2026-07-17/models/` |

---

## Immutable Reference Location

```
experiments/global_gru_reference_2026-07-17/
├── FREEZE_STATEMENT.md
├── config/              baseline_metadata.json
├── models/              global_gru.keras, global_gru_metadata.json
├── training/            training_metadata.json
├── evaluation/        evaluation_df.csv, evaluation_summary.csv, evaluation_metadata.json
├── verification/        verification_notes.md (Phase 4 + Phase 5)
└── plots/               Phase 5 publication figures (PNG + PDF)

experiments/hybrid_vs_global_2026-07-17_095147/   ← Phase 6 comparison (read-only)
├── comparison_table.csv, per_container_delta.csv
├── phase6_comparison.json, comparison_metadata.json
├── FINAL_COMPARISON_SUMMARY.md
├── plots/               cross-methodology figures (PNG + PDF)
└── verification/        comparison verification record
```

This folder is the **permanent Global GRU control reference**. Future experiments must not overwrite it.

---

## Frozen Components

The following must remain unchanged during Peak-Aware Global GRU and all subsequent research:

- `data/` preprocessing artifacts
- `models/global_gru.keras` and `models/global_gru_metadata.json` (production path)
- `experiments/global_gru_reference_2026-07-17/` evaluation and plots (this reference)
- Phase 0.5 locked training configuration and model form
- Phase 5 evaluation artifacts in this reference
- Phase 6 comparison experiment `experiments/hybrid_vs_global_2026-07-17_095147/` (read-only record)

Shared Hybrid baseline artifacts remain independently frozen under `experiments/baseline_reference_2026-07-14/`.

---

## Future Experiment Policy

All future experiments — including **Peak-Aware Global GRU** — must:

1. Compare against this frozen baseline as the control.
2. Implement changes only in **parallel modules** (e.g. `utils/global_training_peak_aware.py`).
3. Save outputs only under new `experiments/` directories.
4. **Never** retrain, overwrite, or modify this frozen baseline or its evaluation files.

---

## Baseline Governance

> The Global GRU v1 baseline is frozen for implementation, training, cohort evaluation, and methodology comparison. Any future modifications — including Peak-Aware Learning, architecture changes, preprocessing changes, feature engineering, or hyperparameter optimisation — must be implemented as separate experiments using this frozen baseline as the control. The baseline itself must remain unchanged to preserve reproducibility and ensure fair experimental comparison.

---

## Verification

| Phase | Status | Log |
|-------|--------|-----|
| Phase 4 implementation | **PASS** (2026-07-17) | `verification/verification_notes.md` |
| Phase 5 evaluation (5 scripts) | **PASS** (2026-07-17) | `experiments/global_gru_evaluation_2026-07-17_092120/verification/` |
| Phase 6 methodology comparison | **PASS** (2026-07-17) | `experiments/hybrid_vs_global_2026-07-17_095147/verification/` |

Phase 5 full evaluation summary: `experiments/global_gru_evaluation_2026-07-17_092120/FINAL_EVALUATION_SUMMARY.md`  
Phase 6 comparison summary: `experiments/hybrid_vs_global_2026-07-17_095147/FINAL_COMPARISON_SUMMARY.md`

---

## Study Track Status

**Global GRU baseline research track: CLOSED** (Phases 1–6 complete).

Documentation: [Phase 6 — Baseline Freeze & Methodology Comparison](../../docs/global-gru-baseline/phase-06-baseline-freeze-and-comparison.md)

**Hybrid control reference (read-only):** `experiments/baseline_reference_2026-07-14/`
