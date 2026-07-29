# Phase 3 — Pipeline & Architecture Design

[← Overview](README.md) · [Phase 2 Methodology Audit](phase-02-methodology-audit.md)

---

## 4. Phase 3 — Pipeline & Architecture Design

### 4.1 Objective

Lock all design decisions for **Global GRU v1** before implementation: data flow, sequence generation, model architecture, training hyperparameters, inference protocol, evaluation methodology, artifact structure, and module layout.

This phase is **design only**: no implementation, no training, no modification of frozen Hybrid or preprocessing artifacts.

### 4.2 Inputs from Phase 2 (Locked)

| Item | Decision |
|------|----------|
| Sequence source | `global_train` only — no concat with `global_val` |
| Training val split | 80/20 index on pooled train sequences — early stopping only |
| Final evaluation | Temporal holdout on `global_val` — last 96 train → first 96 val |
| Hybrid control | `experiments/baseline_reference_2026-07-14/` |
| Peak-Aware | **Out of scope** for Global GRU baseline |

### 4.3 Phase 3 Constraints

| Constraint | Rule |
|------------|------|
| Frozen Hybrid | Do **not** modify `utils/hybrid_*.py`, `notebooks/hybrid_model.ipynb`, `models/hybrid_*`, or `baseline_reference/` |
| Preprocessing | Do **not** change frozen `data/` artifacts or `preprocessing.ipynb` |
| Peak-Aware | Do **not** import or use `utils/peak_*.py` or `hybrid_training_peak_aware.py` |
| Model / hyperparameter changes | Do **not** change layer sizes, features, or hyperparameters from existing notebook during v1 baseline |
| Optimisation or tuning | Do **not** perform any hyperparameter search or model-structure optimisation |
| Implementation | Deferred to Phase 4; formal lock in [Phase 0.5](phase-03-5-baseline-specification.md) |

---

## 4.4 Locked Forecast Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Resampling interval | 15 minutes | Frozen preprocessing |
| Input window | 96 steps (24 h) | Research spec; matches locked Hybrid |
| Forecast horizon (Day 1) | 96 steps (24 h) | Research spec |
| Target metric | `cpu_util_percent` → `cpu_scaled` | Module 1 scope |
| Random seed | 42 | Match Hybrid baseline metadata |

---

## 4.5 Data Sources (Frozen — Reuse Without Modification)

| Artifact | Path | Role |
|----------|------|------|
| Train data | `data/train_df.parquet` | GRU training sequences (~613 steps/container) |
| Validation data | `data/val_df.parquet` | Temporal holdout evaluation (~154 steps/container) |
| Scalers | `data/scalers.pkl` | Per-container MinMax inverse transform |
| Container cohort | `data/selected_containers.npy` | 100-container evaluation subset |
| Preprocessing record | `data/preprocessing_summary.json` | Reproducibility metadata |

**Filtering pattern (identical to Hybrid):**

```python
global_train = train_df[train_df["container_id"].isin(selected_containers)]
global_val   = val_df[val_df["container_id"].isin(selected_containers)]
```

### Locked Input Features

| Feature | Used | Rationale |
|---------|------|-----------|
| `cpu_scaled` | ✓ (input + target) | Primary time-series signal |
| `cpu_mean` | ✓ (input) | Static container context for generalization |
| `cpu_std` | ✓ (input) | Static container context for generalization |
| `hour` | ✗ | Excluded — commented out in existing notebook; no justification to add |
| `cpu_max`, `cpu_min` | ✗ | Unused in existing design |

---

## 4.6 Sequence Generation Design

### Function

**Name:** `create_longterm_sequences()`  
**Location:** `utils/sequence_utils.py` (new function, parallel to `create_residual_sequences()`)

### Training Sequences

| Parameter | Value |
|-----------|-------|
| Source dataframe | `global_train` only |
| `input_window` | 96 |
| `forecast_horizon` | 96 |
| Features | `["cpu_scaled", "cpu_mean", "cpu_std"]` |
| Target | `cpu_scaled` |
| Ordering | Per-container chronological; pooled across containers |
| Shuffle | `False` |

**Sliding window logic (per container):**

```python
for i in range(len(group) - input_window - forecast_horizon):
    X = values[i : i + input_window]                    # shape: (96, 3)
    y = target_values[i + input_window : i + input_window + forecast_horizon]  # shape: (96,)
```

**Expected sequence count:** ~42,000 from 100 containers (~517 valid windows per container).

### Training-Time Validation Split (Early Stopping Only)

```python
split_idx = int(len(X_all) * 0.8)
X_train, y_train = X_all[:split_idx], y_all[:split_idx]
X_val_es, y_val_es = X_all[split_idx:], y_all[split_idx:]
```

**Must never** be used for final reported metrics.

### Inference Sequence (Evaluation)

For each container — **one canonical Day 1 forecast**:

```
Input  = last 96 train timesteps of [cpu_scaled, cpu_mean, cpu_std]
Target = first 96 val timesteps of cpu_scaled (ground truth)
Output = model.predict(input) → inverse MinMax → real CPU %
```

Do **not** evaluate on all sliding windows within the val period for primary metrics.

---

## 4.7 Model Architecture Design (Locked)

Preserve existing notebook design — no changes during v1 baseline establishment.

```
Input (96, 3)
    │
    ▼
GRU(128, return_sequences=True)
    │
    ▼
Dropout(0.2)
    │
    ▼
GRU(64)
    │
    ▼
Dense(64, relu)
    │
    ▼
Dense(96)          ← DAY1_HORIZON outputs (cpu_scaled, linear)
```

| Property | Value |
|----------|-------|
| Version | `global_gru_v1` |
| Output activation | Linear |
| Dropout | 0.2 after first GRU layer (existing notebook — do not change for v1) |
| Loss | MSE |
| Optimizer | Adam |
| Training metric | MAE |

### Rationale for Differences vs Hybrid GRU

| Aspect | Global GRU | Hybrid GRU | Note |
|--------|------------|------------|------|
| Task | Direct CPU forecasting | Residual correction | Different roles by design |
| Input features | 3 | 1 | Global model needs container context |
| Layer sizes | 128→64 | 256→128→64 | Document, do not align for v1 |
| Dropout | 0.2 (after first GRU) | 0.2 (after each GRU) | Existing notebook values |

---

## 4.8 Training Hyperparameters (Locked)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Match Hybrid family |
| Loss | MSE | Standard continuous forecasting |
| Training metric | MAE | Match Hybrid |
| Epochs | 50 max | Existing notebook — do not change |
| Batch size | 256 | Existing notebook — do not change |
| Shuffle | `False` | Mandatory |
| Early stopping | `monitor="val_loss"`, `patience=3`, `restore_best_weights=True` | Existing notebook |
| Random seed | 42 | Match Hybrid baseline |

---

## 4.9 Module Layout Design (Reuse-First)

Do **not** mirror every Hybrid module. Reuse shared utilities; implement new modules only for Global GRU–specific behaviour.

### Shared Utilities (Reuse or Extend — No Duplication)

| Module | Role |
|--------|------|
| `utils/sequence_utils.py` | Extend with `create_longterm_sequences()` — shared sliding-window logic |
| `utils/hybrid_config.py` | Read-only import of `DAY1_HORIZON`, `DEFAULT_INPUT_WINDOW` |
| `utils/hybrid_evaluation.py` | Read-only import of `compute_day1_metrics()`, `compute_day1_mape()`, `_evaluable_container_ids()` |

### Global GRU–Specific Modules (New Code Only)

| Module | Responsibility |
|--------|----------------|
| `utils/global_config.py` | Global-only constants: feature list, model paths, `RANDOM_SEED` |
| `utils/global_training.py` | `build_global_gru_model()`, `train_global_gru()` |
| `utils/global_inference.py` | `GlobalInferenceResult`, `run_global_inference()` |
| `utils/global_artifacts.py` | `save_global_artifacts()`, `load_global_artifacts()` |
| `utils/global_evaluation.py` | Thin orchestration: `evaluate_selected_containers()` — delegates metrics to shared functions |
| `notebooks/global_gru_model.ipynb` | Two-section notebook with `RUN_TRAINING` guard |

**Rule:** If logic is identical across methodologies (metrics, container pre-filtering, window constants), import or reuse it. Do not copy Hybrid module structure for its own sake.

### Notebook Workflow (Mirror `hybrid_model.ipynb`)

```
Section 1 — Training (RUN_TRAINING = True)
  ├─ Load frozen data artifacts
  ├─ Filter to selected containers
  ├─ train_global_gru(global_train)
  └─ save_global_artifacts() → models/global_gru.keras + metadata

Section 2 — Evaluation (always runs)
  ├─ load_global_artifacts()
  ├─ evaluate_selected_containers() on global_val
  └─ evaluation_df / evaluation_summary + demo plots
```

**Rule:** Section 2 never calls `model.fit()`.

---

## 4.10 Evaluation Design

### Primary Evaluation (Day 1)

**Per container:**

1. Extract train and val slices
2. Build input: last 96 train steps of features
3. `model.predict()` → 96 scaled CPU values
4. Inverse MinMax via per-container scaler → real CPU %
5. Compare against first 96 val actuals (real CPU %)
6. Compute MAE, RMSE, MAPE

**MAPE formula (match Hybrid):**

```
MAPE = mean(|actual - predicted| / max(|actual|, 0.01)) × 100
```

### Multi-Container Evaluation

Import `_evaluable_container_ids()` from `utils/hybrid_evaluation.py` (read-only) — do not duplicate:

- Expected: **99 evaluable**, **1 skipped** (`c_14674`)

### `evaluation_df` Schema (Match Hybrid)

| Column | Type | Description |
|--------|------|-------------|
| `container_id` | str | Container identifier |
| `day1_mae` | float | Day 1 MAE (real CPU %) |
| `day1_rmse` | float | Day 1 RMSE (real CPU %) |
| `day1_mape` | float | Day 1 MAPE (%) |
| `train_steps` | int | Train-period length |
| `validation_steps` | int | Validation-period length |

### Supplementary Outputs (Not Primary Metrics)

| Output | Scope |
|--------|-------|
| Demo container plot | `c_11461` (same as Hybrid) |
| Sample Actual vs Predicted | 3–5 containers, PNG + PDF |
| Error distribution | Cohort-level histogram |
| Day 2 recursive forecast | **Deferred** |

---

## 4.11 Artifact Structure Design

### Production Path

| File | Contents |
|------|----------|
| `models/global_gru.keras` | Trained Keras model |
| `models/global_gru_metadata.json` | Hyperparameters, features, seed, date |

### Experiment Directory (Timestamped)

```
experiments/global_gru_baseline_YYYY-MM-DD_HHMMSS/
├── config/
│   └── baseline_metadata.json
├── models/
│   ├── global_gru.keras
│   └── global_gru_metadata.json
├── training/
│   └── training_metadata.json
├── evaluation/
│   ├── evaluation_df.csv
│   ├── evaluation_summary.csv
│   └── inference_cache.pkl          # optional
├── verification/
│   └── verification_notes.md
└── plots/
    ├── actual_vs_predicted/
    └── error_distribution.png
```

### Frozen Reference (After Phase 6)

```
experiments/global_gru_reference_YYYY-MM-DD/
├── FREEZE_STATEMENT.md
├── models/
├── evaluation/
├── config/
└── verification/
```

---

## 4.12 Verification Scripts Design (Phase 5)

| Script | Asserts |
|--------|---------|
| `verify_global_gru_data_split.py` | No val-period targets in training sequences |
| `verify_global_gru_inference.py` | Single-container inference matches manual calculation |
| `verify_global_gru_evaluation.py` | 99 evaluated, 1 skipped; schema correct; MAPE computed; no duplicate IDs; no NaN metrics; non-negative MAE/RMSE/MAPE; RMSE ≥ MAE per container |
| `verify_global_gru_train_eval_split.py` | Loaded model = in-memory model; no retrain in eval |
| `verify_global_gru_hybrid_parity.py` | Same cohort, period, metrics, MAPE ε as Hybrid |

---

## 4.13 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Lock forecast configuration | **Complete** |
| 2 | Lock data sources and features | **Complete** |
| 3 | Design sequence generation and splits | **Complete** |
| 4 | Lock model architecture and hyperparameters | **Complete** |
| 5 | Design module layout and notebook workflow | **Complete** |
| 6 | Design evaluation protocol and schema | **Complete** |
| 7 | Design artifact and verification structure | **Complete** |
| 8 | Lock Phase 3 design document | **Complete** |

---

## 4.14 Exit Criteria (Before Phase 4)

- [x] All design decisions documented and locked
- [x] No open architecture or hyperparameter decisions for v1
- [x] Fair comparison contract reflected in evaluation design
- [x] Module layout defined (reuse-first, not mirror-all)
- [x] Artifact structure defined
- [x] Verification script list defined
- [x] Peak-Aware explicitly excluded
- [ ] Phase 4 implementation **not** started (gate)

### 4.15 Conclusion

Phase 3 is complete. Global GRU v1 design decisions are documented: 96/96 window, 3-feature direct CPU forecasting, 128→64 GRU with Dropout(0.2) after first GRU, train-period sequences only, temporal-holdout Day 1 evaluation on 99 containers, real CPU % metrics, reuse-first module layout.

The formal research contract is recorded in [Phase 0.5 — Baseline Specification](phase-03-5-baseline-specification.md). Implementation may proceed in Phase 4 only after Phase 0.5 is approved.

**Next phase:** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)
