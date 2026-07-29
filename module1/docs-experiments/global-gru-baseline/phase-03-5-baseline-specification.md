# Phase 0.5 — Global GRU Baseline Specification

[← Overview](README.md) · [Phase 3 Pipeline Design](phase-03-pipeline-design.md) · [Phase 4 Implementation](phase-04-implementation.md)

---

## 4.5 Phase 0.5 — Global GRU Baseline Specification

### 4.5.1 Objective

Formally lock the **complete experimental configuration** for Global GRU v1 as a research contract before any implementation code is written.

This phase converts the design decisions from Phases 1–3 into an immutable specification. Every component listed below remains frozen throughout the baseline experiment. Later studies — including Peak-Aware Global GRU — may change **only** an explicitly approved experimental variable (e.g. the training loss function) while all other components stay identical.

This phase is **specification only**: no training, no notebook execution, no modification of frozen Hybrid or preprocessing artifacts.

### 4.5.2 Methodology

1. Consolidated design decisions from [Phase 3 — Pipeline & Architecture Design](phase-03-pipeline-design.md).
2. Mapped each specification item to **frozen**, **shared (reuse)**, or **Global GRU–specific (new code)**.
3. Defined the fairness contract: what is identical vs what differs between forecasting methodologies.
4. Recorded the specification as the authoritative reference for Phases 4–6 and future Peak-Aware Global GRU work.

### 4.5.3 Assumptions

- The existing `notebooks/global_gru_model.ipynb` prototype supplies the starting model form (128→64 GRU, Dropout(0.2) after first GRU, three input features, existing hyperparameters) — not an optimised configuration.
- No hyperparameter tuning, feature selection, or model-structure search is permitted during baseline establishment.
- The Hybrid control in `experiments/baseline_reference_2026-07-14/` defines the shared evaluation protocol; Global GRU adopts that protocol without modification.

### 4.5.4 Implementation Philosophy (Baseline Experiment)

During the Global GRU baseline experiment, the following are **strictly out of scope**:

| Prohibited activity | Rationale |
|---------------------|-----------|
| Peak-Aware Learning | Separate study track after baseline freeze |
| Model-structure optimisation | Would confound the baseline with tuning |
| Hyperparameter tuning | Existing notebook values are locked as-is |
| Additional input features | Would change the experimental configuration |
| Preprocessing changes | Shared pipeline must remain identical |
| Changes to evaluation protocol | Fair comparison requires fixed reporting |

**Objective:** Establish a scientifically valid, reproducible baseline — not maximise accuracy.

Any future enhancement (Peak-Aware Learning, architecture changes, feature engineering, hyperparameter optimisation, unseen-container splits) must be treated as a **separate experiment** after this baseline has been verified, evaluated, and frozen.

---

## Locked Baseline Specification — `global_gru_v1`

### Data & Preprocessing (Frozen — Identical to Hybrid)

| Item | Specification |
|------|---------------|
| Dataset | Alibaba Cluster Trace — `container_usage.csv` |
| Target variable | `cpu_util_percent` → `cpu_scaled` (MinMax, train-fit only) |
| Preprocessing artifacts | `data/train_df.parquet`, `data/val_df.parquet`, `data/scalers.pkl`, `data/selected_containers.npy` |
| Temporal split | Per-container 80/20 chronological split from frozen preprocessing |
| Container cohort | 100 selected → 99 evaluable (`c_14674` skipped) |
| Resampling | 15-minute intervals |

### Input Features (Locked)

| Feature | Role |
|---------|------|
| `cpu_scaled` | Input feature **and** prediction target |
| `cpu_mean` | Input feature (static train-period container context) |
| `cpu_std` | Input feature (static train-period container context) |

**Excluded:** `hour`, `cpu_max`, `cpu_min` — not part of v1 baseline.

### Sequence Configuration (Locked)

| Parameter | Value |
|-----------|-------|
| Input window | **96** timesteps (24 hours) |
| Forecast horizon | **96** timesteps (24 hours — Day 1 primary) |
| Sequence source | `global_train` only — **never** concatenate with `global_val` |
| Ordering | Per-container chronological; pooled across containers |
| Shuffle | `False` |
| Training val split | Last 20% of pooled **train** sequences — early stopping only |
| Sequence builder | `create_longterm_sequences()` in `utils/sequence_utils.py` |

### Model (Locked — Existing Notebook Form)

| Parameter | Value |
|-----------|-------|
| Version | `global_gru_v1` |
| Layers | GRU(128, return_sequences=True) → Dropout(0.2) → GRU(64) → Dense(64, relu) → Dense(96) |
| Input shape | `(96, 3)` |
| Output shape | `(96,)` — scaled CPU values |
| Dropout | 0.2 (after first GRU layer) |
| Output activation | Linear |

### Training (Locked — Existing Notebook Values)

| Parameter | Value |
|-----------|-------|
| Loss function | MSE |
| Optimizer | Adam |
| Training metric | MAE |
| Batch size | **256** |
| Maximum epochs | **50** |
| Early stopping | `monitor="val_loss"`, `patience=3`, `restore_best_weights=True` |
| Validation during training | Unweighted `val_loss` on train-sequence holdout only |
| Random seed | **42** |

### Evaluation (Locked — Identical Protocol to Hybrid)

| Parameter | Value |
|-----------|-------|
| Evaluation period | Preprocessing validation period (`global_val`) |
| Inference context | Last 96 train timesteps per container |
| Primary horizon | Day 1 — first 96 validation timesteps |
| Forecasts per container | One canonical Day 1 forecast (not sliding windows over val) |
| Metrics | MAE, RMSE, MAPE |
| Metric unit | **Real CPU %** after per-container inverse MinMax |
| MAPE epsilon | 0.01 percentage points (same as Hybrid) |
| Aggregation | Per-container → cohort mean ± std |
| Demo container | `c_11461` (same as Hybrid) |

**MAPE formula:**

```
MAPE = mean(|actual - predicted| / max(|actual|, 0.01)) × 100
```

### Temporal Evaluation Protocol (Locked)

```
Layer 1 — Preprocessing (frozen):
  Per-container 80/20 chronological split → train_df / val_df

Layer 2 — GRU training:
  Sequences from train period only
  80/20 index split on pooled train sequences → early stopping only

Layer 3 — Final evaluation:
  Per-container temporal holdout on val_df
  Last 96 train steps → predict first 96 val steps → inverse MinMax → metrics
```

### Saved Artifacts (Locked)

| Artifact | Location |
|----------|----------|
| Production model | `models/global_gru.keras` |
| Metadata record | `models/global_gru_metadata.json` |
| Experiment directory | `experiments/global_gru_baseline_YYYY-MM-DD_HHMMSS/` |
| Frozen reference (after Phase 6) | `experiments/global_gru_reference_YYYY-MM-DD/` |

**Experiment directory must include:**

- `config/baseline_metadata.json` — full specification record
- `models/global_gru.keras` + metadata copy
- `training/training_metadata.json`
- `evaluation/evaluation_df.csv`, `evaluation_summary.csv`
- `verification/verification_notes.md`
- `plots/` — PNG + PDF figures

---

## Forecasting Methodology Comparison (Research Contract)

This baseline study compares **forecasting methodologies** under an identical experimental protocol — not neural network architectures in isolation.

| Component | Status across methodologies |
|-----------|----------------------------|
| Preprocessing pipeline | **Identical** |
| Temporal data split | **Identical** |
| Evaluation procedure | **Identical** |
| Metrics and reporting | **Identical** |
| Container cohort | **Identical** |
| Primary forecast horizon | **Identical** |
| **Forecasting methodology** | **Different by design** |

| Forecasting methodology | Hybrid Prophet + GRU (control) | Global GRU (baseline) |
|-------------------------|----------------------------------|----------------------|
| Approach | Decomposed: Prophet trend/seasonality + GRU residual correction | End-to-end: single shared GRU direct CPU forecast |
| Per-container statistical model | Prophet refit at inference | None |
| GRU role | Residual learner | Sole forecaster |
| Final prediction | `prophet + gru_residual` → inverse MinMax | `gru_prediction` → inverse MinMax |

Internal model differences (layer sizes, feature count, training hyperparameters) are **consequences of each forecasting methodology**, not independent variables under test. The research question is which **methodology** performs better under fair conditions — not which GRU layer configuration is optimal.

---

## Code Reuse Contract (Locked)

Implementation must follow a **reuse-first** philosophy. Do not mirror every Hybrid module.

| Layer | Approach |
|-------|----------|
| **Shared utilities (reuse or extend)** | `utils/sequence_utils.py` — add `create_longterm_sequences()`; shared constants from `utils/hybrid_config.py` (read-only import for `DAY1_HORIZON`, `DEFAULT_INPUT_WINDOW`); shared metric functions from `utils/hybrid_evaluation.py` (read-only import for `compute_day1_metrics`, `compute_day1_mape`, `_evaluable_container_ids`) |
| **Global GRU–specific (new code only)** | `utils/global_config.py` — Global-only paths and feature lists; `utils/global_training.py` — model build and training pipeline; `utils/global_inference.py` — direct CPU inference; `utils/global_artifacts.py` — save/load Global GRU model + metadata; thin `utils/global_evaluation.py` — orchestration only, delegates metrics to shared functions |
| **Frozen (do not modify)** | All `utils/hybrid_*.py`, `notebooks/hybrid_model.ipynb`, `experiments/baseline_reference_2026-07-14/` |

**Rule:** If logic is identical between methodologies (metrics, container pre-filtering, sequence sliding-window pattern), **reuse** it. Only implement new modules for behaviour that is genuinely specific to Global GRU direct forecasting.

---

## Future Peak-Aware Global GRU — Single-Variable Contract

When Peak-Aware Global GRU is implemented (after this baseline is frozen):

| Component | Rule |
|-----------|------|
| Everything in this specification | **Identical** to frozen `global_gru_v1` |
| **Single approved change** | Training loss function (or another explicitly approved peak-aware mechanism) |
| Control reference | `experiments/global_gru_reference_YYYY-MM-DD/` |
| Implementation | Parallel module only (e.g. `utils/global_training_peak_aware.py`) |
| Outputs | `experiments/peak_aware_global_*/` only |

---

## Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Consolidate Phase 1–3 decisions into single specification | **Complete** |
| 2 | Document implementation philosophy and prohibitions | **Complete** |
| 3 | Define code reuse contract | **Complete** |
| 4 | Define future Peak-Aware single-variable contract | **Complete** |
| 5 | Review and approve specification before Phase 4 | **Pending approval** |

---

## Exit Criteria (Gate Before Phase 4)

- [x] Complete baseline specification documented (features, target, window, horizon, model, training, evaluation, artifacts)
- [x] Implementation philosophy and prohibitions recorded
- [x] Forecasting methodology comparison contract written
- [x] Code reuse contract defined (reuse-first, not mirror-all)
- [x] Future Peak-Aware single-variable contract defined
- [ ] Specification reviewed and approved
- [ ] Phase 4 implementation **not** started until approval

---

## Conclusion

Phase 0.5 documents the formal research contract for Global GRU v1. This specification is the authoritative reference for all baseline work in Phases 4–6 and for future Peak-Aware Global GRU comparisons.

**Next phase (after approval):** [Phase 4 — Implementation](phase-04-implementation.md)
