# Phase 4 — Implementation

[← Overview](README.md) · [Phase 3 Design](phase-03-learning-design.md) · [Design lock JSON](../experiments/peak_aware_design_2026-07-14/phase3_design.json)

---

## 5. Phase 4 — Implementation

### 5.1 Objective

Implement the **locked Phase 3 design** as parallel modules and scripts. Train the Peak-Aware Hybrid GRU model, save experiment artifacts, and pass fairness verification — without modifying the frozen baseline or running Phase 5 evaluation.

**Design authority:** `experiments/peak_aware_design_2026-07-14/phase3_design.json`

### 5.2 Locked Inputs (From Phase 3)

| Item | Value |
|------|-------|
| Mechanism | Timestep-weighted MSE |
| Peak definition | P90 per-container, train-only, real CPU % |
| λ | 5 |
| Early stopping | Unweighted `val_loss`, patience 10 |
| Seed | 42 |
| Input window / horizon | 96 / 96 |
| Baseline control | `experiments/baseline_reference_2026-07-14/` |

### 5.3 Phase 4 Constraints

| Constraint | Rule |
|------------|------|
| Frozen baseline | Do **not** modify `utils/hybrid_*.py`, `sequence_utils.py`, notebooks, `models/`, or `baseline_reference/` |
| Imports | May import **read-only** from frozen modules (`generate_prophet_residuals`, `build_hybrid_gru_model`, `create_residual_sequences`) |
| Outputs | Save only under `experiments/peak_aware_<timestamp>/` |
| Evaluation | Do **not** run Phase 5 comparison metrics in this phase |
| Design changes | Do **not** change λ, P90, mechanism, or architecture without explicit approval |

### 5.4 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Prepare workspace + `peak_config.py` | **Complete** |
| 2 | Implement `peak_detection.py` | **Complete** |
| 3 | Implement `hybrid_training_peak_aware.py` | **Complete** |
| 4 | Implement + run experiment script (train model) | **Complete** |
| 5 | Implement + run fairness verification | **Complete** |
| 6 | Lock Phase 4 implementation record | **Complete** |

### 5.5 Exit Criteria (Before Phase 5)

- [x] All 5 new files implemented
- [x] Peak-aware model trained and saved to experiment dir
- [x] `peak_thresholds.pkl`, `peak_config.json`, `training_metadata.json` saved
- [x] `res_mean` / `res_std` match baseline computation on same data
- [x] All 8 fairness checks PASS
- [x] Frozen baseline files unmodified
- [x] `docs/peak-aware-hybrid/phase-04-implementation.md` updated task-by-task
- [x] Phase 5 evaluation **not** started

---

### 5.6 Implementation Summary

**Status:** **Complete**  
**Date:** 2026-07-14  
**Training run:** `experiments/peak_aware_2026-07-14_164518/`  
**Lock record:** `experiments/peak_aware_2026-07-14_164518/phase4_implementation.json`

This section consolidates the Phase 4 implementation outcome. Per-task methodology and decisions remain in the task sections below.

#### What Was Built

| Component | Path | Role |
|-----------|------|------|
| Configuration | `utils/peak_config.py` | Locked constants (P90, λ=5, seed=42, paths, N=41650) |
| Peak detection | `utils/peak_detection.py` | P90 thresholds, `W_all` weight matrix, save/load, alignment checks |
| Peak-aware training | `utils/hybrid_training_peak_aware.py` | Weighted GRU pipeline; custom `train_step` for timestep MSE |
| Experiment runner | `scripts/run_peak_aware_hybrid_experiment.py` | Full training + artifact save to timestamped dir |
| Fairness gate | `scripts/verify_peak_aware_fairness.py` | 8 pre-evaluation checks |

All five new files are implemented. Frozen baseline modules (`utils/hybrid_*.py`, `sequence_utils.py`, `models/`) were imported read-only only.

#### Single Experimental Variable

Everything except GRU training loss is identical to the frozen baseline:

| Unchanged | Changed |
|-----------|---------|
| Prophet residual generation | Train loss: timestep-weighted MSE |
| `create_residual_sequences()` → N=41,650 | Peak timesteps weighted λ=5, others 1.0 |
| GRU architecture (96→96) | Custom `train_step` (Keras 2D `sample_weight` workaround) |
| 80/20 pooled split | `sample_weight=W_train` on train only |
| Early stopping on unweighted `val_loss` | — |
| `res_mean`, `res_std` computation | — |

#### Training Outcome

| Item | Value |
|------|-------|
| Model | `experiments/peak_aware_2026-07-14_164518/models/hybrid_gru_peak_aware.keras` |
| Residual stats | `experiments/peak_aware_2026-07-14_164518/models/residual_stats.pkl` |
| Epochs run | **20** / 100 max (early stopping, patience 10) |
| Final train loss | **1.668** (weighted MSE) |
| Final val loss | **1.724** (unweighted) |
| `res_mean` | **1.536e-05** (matches baseline) |
| `res_std` | **0.111239** (matches baseline) |
| Peak-weight fraction | **16.94%** of forecast timesteps |
| Runtime | ~7.4 min (M1 Pro, Prophet + GRU) |

#### Fairness Verification

All 8 checks **PASS** — record at `experiments/peak_aware_2026-07-14_164518/verification/fairness_check.json`.

| # | Check | Result |
|---|-------|--------|
| 1 | Frozen file integrity | PASS |
| 2 | Identical `res_mean` / `res_std` | PASS |
| 3 | Identical `X_all`, `y_all` | PASS |
| 4 | Identical architecture | PASS |
| 5 | Early stopping config | PASS |
| 6 | Unweighted validation | PASS |
| 7 | Shared inference/evaluation path | PASS |
| 8 | Valid P90 threshold artifact | PASS |

#### Artifact Layout (Training Run)

```
experiments/peak_aware_2026-07-14_164518/
├── experiment_metadata.json
├── models/
│   ├── hybrid_gru_peak_aware.keras      ← trained model
│   └── residual_stats.pkl
├── peak/
│   ├── peak_thresholds.pkl
│   └── peak_config.json
├── training/
│   └── training_metadata.json
└── verification/
    └── fairness_check.json
```

**Workspace metadata:** `experiments/peak_aware_implementation_2026-07-14/` (task progress, smoke tests, sanity checks).

#### Baseline Integrity

| Item | Status |
|------|--------|
| `models/hybrid_gru.keras` | Unmodified |
| `models/residual_stats.pkl` | Unmodified |
| `experiments/baseline_reference_2026-07-14/` | Unmodified |
| Frozen `utils/hybrid_*.py` | Unmodified |

#### Key Implementation Decision

Keras default MSE cannot apply 2D `sample_weight` `(batch, 96)` to vector targets. Phase 4 uses a bound custom `train_step`:

```
loss = sum(w · (y − ŷ)²) / sum(w)
```

Validation remains unweighted via the default `test_step`. This is mathematically equivalent to the locked Phase 3 protocol.

#### Phase 4 Status

**Phase 4 is complete.** All six tasks finished; fairness gate passed; ready for Phase 5 evaluation.

Phase 4 implementation is **cleared for evaluation**. Phase 5 should:

1. Load peak-aware artifacts from `experiments/peak_aware_2026-07-14_164518/models/`
2. Run `evaluate_selected_containers()` via unchanged `utils/hybrid_evaluation.py`
3. Compare Day 1 MAE / RMSE / MAPE against `experiments/baseline_reference_2026-07-14/`
4. Report peak-subset metrics using saved `peak_thresholds.pkl`

Phase 5 evaluation has **not** been started.

---

### Task 1 — Prepare Workspace + `peak_config.py`

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Create the Peak-Aware configuration module and initialize the Phase 4 implementation workspace. No training, no modification to frozen baseline files.

#### Methodology

1. Created `utils/peak_config.py` with all locked constants from `phase3_design.json`.
2. Imported `INPUT_WINDOW` and `FORECAST_HORIZON` from frozen `utils/hybrid_config.py` to prevent drift from baseline geometry.
3. Recorded verified sequence counts — see **Pre-Implementation Sequence Count Verification** below and `sequence_count_verification.json`.
4. Created `experiments/peak_aware_implementation_2026-07-14/experiment_metadata.json` to track Phase 4 task progress.
5. Confirmed no frozen baseline files were modified.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Import window/horizon from `hybrid_config` | Guarantees peak-aware geometry matches frozen baseline constants |
| `EXPECTED_N_SEQUENCES = 41_650` in config | Enables assert-before-train check in Tasks 2–4 |
| Separate `peak_aware_implementation_2026-07-14` workspace | Tracks Phase 4 progress; timestamped training run dir set in Task 4 |
| Path constants via `pathlib.Path` | Matches repository Python standards |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Peak-aware constants module | `utils/peak_config.py` |
| Phase 4 workspace metadata | `experiments/peak_aware_implementation_2026-07-14/experiment_metadata.json` |
| Sequence count verification | `experiments/peak_aware_implementation_2026-07-14/sequence_count_verification.json` |

#### `peak_config.py` Constants Summary

| Category | Key constants |
|----------|---------------|
| Peak definition | `PEAK_PERCENTILE=90`, `PEAK_RULE`, `PEAK_VALUE_SPACE` |
| Weighting | `PEAK_WEIGHT=5`, `NON_PEAK_WEIGHT=1` |
| Geometry | `INPUT_WINDOW=96`, `FORECAST_HORIZON=96` (from `hybrid_config`) |
| Training | `RANDOM_SEED=42`, `EPOCHS_MAX=100`, `BATCH_SIZE=64`, `SEQUENCE_VAL_SPLIT=0.8` |
| Early stopping | `val_loss`, patience `10`, restore best weights |
| Sequence counts | `EXPECTED_N_SEQUENCES=41650`, `N_train=33320`, `N_val=8330` |
| References | `BASELINE_REFERENCE_DIR`, `PHASE3_DESIGN_REFERENCE`, data paths |

#### Pre-Implementation Sequence Count Verification

Before Phase 4 coding began, a user-requested check confirmed that the Peak-Aware pipeline will use the **same number of training samples (N)** as the frozen baseline.

##### Verification methodology

1. Loaded frozen `data/train_df.parquet` and `data/selected_containers.npy`.
2. Built `global_train` with the same filter as baseline training: `container_id ∈ selected_containers`.
3. Counted sequences three ways:
   - **Method A:** Baseline window formula — `max(0, L − 96 − 96)` per container, summed.
   - **Method B:** Phase 2 Task 3 loop — identical formula (used in peak exploration).
   - **Method C:** Actual `create_residual_sequences()` from frozen `utils/sequence_utils.py`.
4. Compared results to Phase 2 documented total (**41,650**).

**Record:** `experiments/peak_aware_implementation_2026-07-14/sequence_count_verification.json`

##### Cohort

| Item | Value |
|------|-------|
| Selected containers | 100 |
| Containers in `global_train` | 99 |
| Excluded | `c_14674` (missing from train split) |

##### Sequence count results

| Method | N |
|--------|---|
| Baseline window formula | **41,650** |
| Phase 2 Task 3 loop | **41,650** |
| `create_residual_sequences()` | **41,650** |
| Phase 2 documented (Task 3) | **41,650** |
| **All match** | **Yes** |

##### Train / sequence-validation split (80/20 pooled)

| Quantity | Value | Rule |
|----------|-------|------|
| `N` (total sequences) | **41,650** | `len(X_all)` |
| `split_idx` | **33,320** | `int(N × 0.8)` |
| `N_train` | **33,320** | `X_all[:split_idx]` |
| `N_val` | **8,330** | `X_all[split_idx:]` (early stopping only) |

##### Tensor alignment matrix (baseline vs peak-aware)

Peak-aware training adds a weight matrix only; **all other tensors are identical**.

| Tensor | Baseline shape | Peak-aware shape | Same as baseline? |
|--------|----------------|------------------|-------------------|
| `X_all` | `(41650, 96, 1)` | `(41650, 96, 1)` | **Yes** |
| `y_all` | `(41650, 96)` | `(41650, 96)` | **Yes** |
| `sample_cids` | `(41650,)` | `(41650,)` | **Yes** |
| `W_all` | N/A (uniform implicit) | `(41650, 96)` | **New** — one row per sequence |
| `W_train` | N/A | `(33320, 96)` | First 80% of `W_all` |
| `W_val` | N/A | **Not used** | Validation unweighted |

##### Per-container sequence formula

For each container with `L` chronologically sorted train timesteps:

```
sequences_per_container = L - input_window - forecast_horizon
                        = L - 96 - 96
                        = L - 192
```

`create_residual_sequences()` implements this as:

```python
max_idx = len(group) - input_window - forecast_horizon
for i in range(max_idx):   # generates exactly max_idx sequences
```

Peak-aware `W_all` must assign **exactly one weight row per sequence** — no filtering.

##### Why N cannot change in peak-aware training

| Action | Effect on N |
|--------|-------------|
| Reuse `create_residual_sequences()` on same `train_residual_df` | **N unchanged** |
| Build `W_all` as one row per `sample_cids` entry | **N unchanged** |
| Filter non-peak sequences | **Would reduce N — forbidden** |
| Oversample peak sequences | **Would increase N — forbidden** |
| Different container filter or window size | **Would change N — forbidden** |

##### Phase 4 implementation assertions (Tasks 2–4)

```python
assert len(X_all) == len(y_all) == len(sample_cids) == EXPECTED_N_SEQUENCES
assert W_all.shape == y_all.shape == (EXPECTED_N_SEQUENCES, FORECAST_HORIZON)
assert W_train.shape == (EXPECTED_N_TRAIN, FORECAST_HORIZON)
```

**Re-verification in Task 5:** `verify_peak_aware_fairness.py` check #3 asserts `X_all` and `y_all` are identical to baseline recomputation on the same data.

#### Findings

No training performed. Pre-implementation verification confirmed:

1. **N = 41,650** — all three counting methods match Phase 2.
2. **Split** — `N_train = 33,320`, `N_val = 8,330` (same as baseline).
3. **Weight matrix** — `W_all` shape `(41650, 96)` adds emphasis only; sample count unchanged.

#### Conclusions

Task 1 is complete. Configuration and workspace are ready for `peak_detection.py` (Task 2).

#### Limitations

- `training_run_dir` in workspace metadata is `null` until Task 4 executes training.
- `peak_detection.py` and remaining modules not yet created.

#### Next Task

**Task 2 — Implement `peak_detection.py`:** Threshold computation, weight matrix construction, save/load helpers; sanity check on peak rate ~17%; assert `W_all.shape == (41650, 96)`.

---

## Phase 4 Task Details (Plan — Tasks 2–6)

### Task 2 — Implement `peak_detection.py`

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Implement peak threshold computation, horizon peak labelling, and the sequence weight matrix `W_all` aligned with frozen `create_residual_sequences()`.

#### Methodology

1. Created `utils/peak_detection.py` with threshold, labelling, weight-matrix, save/load, and alignment helpers.
2. Mirrored `create_residual_sequences()` window iteration (`groupby` → sort → `range(max_idx)`).
3. Labelled peaks on **real CPU %** in the forecast horizon only; assigned `λ=5` or `1.0`.
4. Ran sanity check on frozen `data/` including Prophet residual path and baseline `sample_cids` alignment.
5. Saved results to `peak_detection_sanity.json`.

#### Functions Implemented

| Function | Purpose |
|----------|---------|
| `cpu_real_series()` | Inverse MinMax → real CPU % |
| `compute_peak_thresholds()` | Per-container P90 on train period |
| `label_peak_timesteps()` | Boolean peak mask (`>=` threshold) |
| `build_sequence_weight_matrix()` | `W_all` shape `(N, 96)` + `sample_cids` |
| `summarize_weight_matrix()` | Peak-weight fraction for sanity checks |
| `assert_binary_weight_values()` | Assert only `1.0` and `λ` present in `W_all` |
| `verify_binary_weight_values()` | Report unique values and counts |
| `save_peak_thresholds()` / `load_peak_thresholds()` | Threshold artifact I/O |
| `align_with_residual_sequences()` | Verify `sample_cids` match baseline |

#### Sanity Check Results

| Check | Result |
|-------|--------|
| Thresholds computed | **99** (= containers in train) |
| `W_all` shape | **(41650, 96)** |
| `y_all` shape | **(41650, 96)** |
| Shapes match | **Yes** |
| `sample_cids` == baseline | **Yes** (`cid_arrays_equal: true`) |
| Peak-weighted timestep fraction | **16.94%** (Phase 2 reference: **17.36%**) |
| `N_train` / `N_val` | **33320 / 8330** |
| **Binary weight values only** | **Yes** — unique values `{1.0, 5.0}` only |
| Peak weight count (λ=5) | **677,184** |
| Non-peak weight count (1.0) | **3,321,216** |
| Total entries | **3,998,400** (= 41650 × 96) |

**Binary weight verification:** `verify_binary_weight_values()` and `assert_binary_weight_values()` — all entries are exactly `1.0` or `5.0`; no other values present.

**Record:** `experiments/peak_aware_implementation_2026-07-14/peak_detection_sanity.json`

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Same `groupby` loop as `sequence_utils` | Guarantees row order matches baseline `sample_cids` |
| Return `sample_cids` from weight builder | Enables explicit alignment verification |
| Weights on `global_train` CPU, not residuals | Phase 2 peak definition is on real CPU % |
| `float32` weight rows | Sufficient precision; matches typical Keras `sample_weight` |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Peak detection module | `utils/peak_detection.py` |
| Sanity check results | `experiments/peak_aware_implementation_2026-07-14/peak_detection_sanity.json` |

#### Conclusions

Task 2 exit criterion met: `W_all` aligns with baseline sequences; peak rate ~17%; `N=41650` verified.

#### Next Task

**Task 3 — Implement `hybrid_training_peak_aware.py`:** Full weighted training pipeline per Phase 3 Task 4 protocol.

---

### Task 2 — Implement `peak_detection.py` *(plan reference)*

**Deliverables:**
- `utils/peak_detection.py`:
  - `compute_peak_thresholds()`
  - `build_sequence_weight_matrix()` → `W_all` shape `(N, 96)`
  - `save_peak_thresholds()` / `load_peak_thresholds()`
- Lightweight sanity check: threshold count = train containers; mean peak weight rate ~17% at P90
- Update documentation

**Exit criterion:** Weight matrix aligns with `create_residual_sequences()` indices; sanity check passes.

---

### Task 3 — Implement `hybrid_training_peak_aware.py`

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Implement the full peak-aware training pipeline per Phase 3 Task 4 protocol: identical Prophet residual generation and sequence construction as baseline, with per-timestep weighted MSE on `y_train` only.

#### Methodology

1. Created `utils/hybrid_training_peak_aware.py` with `set_random_seeds()` and `train_hybrid_gru_peak_aware()`.
2. Reused frozen imports: `generate_prophet_residuals`, `build_hybrid_gru_model`, `create_residual_sequences`.
3. Integrated `compute_peak_thresholds()` and `build_sequence_weight_matrix()` from Task 2.
4. Added pre-train assertions: `N=41650`, binary weights, `sample_cids` alignment, `W_all.shape == y_all.shape`.
5. Applied 80/20 pooled split; `W_train` on train only; validation unweighted.
6. Installed custom `train_step` for timestep-weighted MSE (Keras default MSE rejects 2D `sample_weight`).
7. Ran 1-epoch smoke test on frozen `data/`; saved `peak_aware_training_smoke.json`.

#### Functions Implemented

| Function | Purpose |
|----------|---------|
| `set_random_seeds()` | Python, NumPy, TensorFlow seed = 42 |
| `_weighted_train_step()` | Custom train step: `loss = sum(w·(y−ŷ)²) / sum(w)` |
| `_install_weighted_train_step()` | Bind weighted step via `types.MethodType` |
| `train_hybrid_gru_peak_aware()` | Full 13-step pipeline; returns model, stats, thresholds, `W_all` |

#### Weighted Loss Implementation

Keras default MSE with `sample_weight` shape `(batch, 96)` raised `Incompatible shapes: [64,96] vs. [64]`. Phase 4 uses a custom `train_step` that computes:

```
loss = sum(sq_error * sample_weight) / sum(sample_weight)
```

- **Train:** weighted MSE via `sample_weight=W_train`
- **Validation:** default unweighted `val_loss` (no `sample_weight` passed)
- **MAE metric:** unweighted, for logging parity with baseline

#### Smoke Test Results

| Check | Result |
|-------|--------|
| Epochs run | **1** |
| `final_train_loss` | **1.608** |
| `final_val_loss` | **1.856** (unweighted) |
| `res_mean` | **1.54e-05** |
| `res_std` | **0.1112** |
| `W_all` shape | **(41650, 96)** |
| Binary weights | **Yes** — `{1.0, 5.0}` only |
| Peak-weight fraction | **16.94%** |
| Pipeline status | **PASS** |

**Record:** `experiments/peak_aware_implementation_2026-07-14/peak_aware_training_smoke.json`

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Custom `train_step` not custom loss class | Keras 3 passes 2D `sample_weight` incorrectly to default MSE; train_step gives exact control |
| `types.MethodType` binding | Ensures metric `update_state` / `result()` work with compiled metrics |
| Unweighted MAE in train_step | Matches baseline `metrics=["mae"]` for training logs |
| Compile after model build, before fit | Standard Keras flow; weighted step installed before compile |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Peak-aware training module | `utils/hybrid_training_peak_aware.py` |
| Smoke test results | `experiments/peak_aware_implementation_2026-07-14/peak_aware_training_smoke.json` |

#### Conclusions

Task 3 exit criterion met: `train_hybrid_gru_peak_aware()` runs end-to-end with weighted train loss and unweighted validation.

#### Next Task

**Task 4 — Implement + run experiment script:** `scripts/run_peak_aware_hybrid_experiment.py`; full training (100 epochs, early stopping); save artifacts to `experiments/peak_aware_<timestamp>/`.

---

### Task 3 — Implement `hybrid_training_peak_aware.py` *(plan reference)*

**Objective:** Implement the full 13-step peak-aware training pipeline (Phase 3 Task 4).

**Deliverables:**
- `utils/hybrid_training_peak_aware.py`:
  - `set_random_seeds()`
  - `train_hybrid_gru_peak_aware()` — mirrors baseline flow + `sample_weight=W_train`
  - Custom weighted MSE loss if Keras default `sample_weight` insufficient
- Reuse frozen `generate_prophet_residuals`, `build_hybrid_gru_model`, `create_residual_sequences`
- Update documentation

**Exit criterion:** Function runs end-to-end in isolation (can use short dry-run or full train in Task 4).

---

### Task 4 — Implement + Run Experiment Script

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Orchestrate full peak-aware training and persist all experiment artifacts to a timestamped directory without touching frozen baseline `models/`.

#### Methodology

1. Created `scripts/run_peak_aware_hybrid_experiment.py` (CLI: `--seed`, `--epochs`, `--output-dir`).
2. Loaded frozen `data/train_df.parquet`, `scalers.pkl`, `selected_containers.npy`.
3. Called `train_hybrid_gru_peak_aware()` with `epochs=100`, `seed=42`, `verbose=1`.
4. Saved model, residual stats, peak thresholds, peak config, and metadata per Phase 3 layout.
5. Created empty `verification/` and `evaluation/` directories for Tasks 5 and Phase 5.
6. Updated implementation workspace with `training_run_dir`.

#### Training Run

| Item | Value |
|------|-------|
| Experiment directory | `experiments/peak_aware_2026-07-14_164518` |
| Runtime | ~7.4 min (Prophet + GRU on M1 Pro) |
| Epochs max | 100 |
| Epochs run | **20** (early stopping, patience 10) |
| Final train loss | **1.668** (weighted) |
| Final val loss | **1.724** (unweighted) |
| Final train MAE | **0.792** |
| Final val MAE | **0.807** |
| `res_mean` | **1.54e-05** |
| `res_std` | **0.1112** |
| Peak thresholds | **99** containers |
| Baseline `models/` | **Untouched** |

#### Saved Artifacts

```
experiments/peak_aware_2026-07-14_164518/
├── experiment_metadata.json
├── models/
│   ├── hybrid_gru_peak_aware.keras
│   └── residual_stats.pkl
├── peak/
│   ├── peak_thresholds.pkl
│   └── peak_config.json
├── training/
│   └── training_metadata.json
├── verification/          (empty — Task 5)
└── evaluation/          (empty — Phase 5)
```

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Inline artifact save in script | Avoids writing to production `models/` via `save_hybrid_artifacts` defaults |
| `residual_stats.pkl` includes `model_variant` | Distinguishes peak-aware artifacts at load time in Phase 5 |
| UTC `trained_at` in training metadata | Reproducible timestamp independent of local timezone |
| No Phase 5 evaluation in script | Phase 4 scope ends at training + artifact save |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Experiment runner | `scripts/run_peak_aware_hybrid_experiment.py` |
| Trained model + stats | `experiments/peak_aware_2026-07-14_164518/models/` |
| Peak artifacts | `experiments/peak_aware_2026-07-14_164518/peak/` |
| Training metadata | `experiments/peak_aware_2026-07-14_164518/training/training_metadata.json` |
| Experiment metadata | `experiments/peak_aware_2026-07-14_164518/experiment_metadata.json` |

#### Conclusions

Task 4 exit criterion met: full training completed with early stopping; all artifacts saved; frozen baseline `models/hybrid_gru.keras` unchanged.

#### Next Task

**Task 5 — Implement + run fairness verification:** `scripts/verify_peak_aware_fairness.py`; 8 checks against `experiments/peak_aware_2026-07-14_164518/`.

---

### Task 4 — Implement + Run Experiment Script *(plan reference)*

**Objective:** Orchestrate training, save all artifacts to timestamped experiment directory.

**Deliverables:**
- `scripts/run_peak_aware_hybrid_experiment.py`
- Execute training → `experiments/peak_aware_<timestamp>/`
  - `models/hybrid_gru_peak_aware.keras`
  - `models/residual_stats.pkl`
  - `peak/peak_thresholds.pkl`, `peak/peak_config.json`
  - `training/training_metadata.json`
  - `experiment_metadata.json`
- Record training metadata (epochs run, `res_mean`, `res_std`, sequence counts)
- Update documentation

**Exit criterion:** Trained model and all artifacts saved; `models/` (baseline) untouched.

---

### Task 5 — Implement + Run Fairness Verification

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Assert the peak-aware implementation satisfies the Phase 3 fairness contract before Phase 5 evaluation.

#### Methodology

1. Created `scripts/verify_peak_aware_fairness.py` with 8 automated checks.
2. Ran against `experiments/peak_aware_2026-07-14_164518/`.
3. Saved results to `verification/fairness_check.json`.
4. Updated experiment and workspace metadata.

#### Fairness Check Results

| # | Check | Result |
|---|-------|--------|
| 1 | Frozen file integrity | **PASS** |
| 2 | Identical `res_mean` / `res_std` | **PASS** |
| 3 | Identical `X_all`, `y_all` | **PASS** |
| 4 | Identical architecture | **PASS** |
| 5 | Early stopping config | **PASS** |
| 6 | Unweighted validation | **PASS** |
| 7 | Shared inference/evaluation path | **PASS** |
| 8 | Valid P90 threshold artifact | **PASS** |

**Overall:** **8/8 PASS**

**Record:** `experiments/peak_aware_2026-07-14_164518/verification/fairness_check.json`

#### Key Verifications

| Item | Result |
|------|--------|
| `res_mean` (saved vs recomputed) | **1.536e-05** — exact match |
| `res_std` (saved vs recomputed) | **0.111239** — exact match |
| Sequence shapes | **(41650, 96, 1)** / **(41650, 96)** |
| Architecture layers | **8 layers** — matches `build_hybrid_gru_model(96, 1, 96)` |
| Early stopping | `val_loss`, patience **10**, restore best weights |
| Inference smoke (`c_11461`) | Day 1 prediction **96 steps**, eval **1 container** |
| Peak thresholds | **99** containers, values match recomputation |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Fairness verification script | `scripts/verify_peak_aware_fairness.py` |
| Fairness check report | `experiments/peak_aware_2026-07-14_164518/verification/fairness_check.json` |

#### Conclusions

Task 5 exit criterion met: all 8 fairness checks PASS. Implementation is cleared for Phase 5 evaluation.

#### Next Task

**Task 6 — Lock Phase 4 record:** consolidate implementation results and mark Phase 4 complete.

---

### Task 5 — Implement + Run Fairness Verification *(plan reference)*

**Objective:** Assert the implementation satisfies the Task 1 fairness contract.

**Deliverables:**
- `scripts/verify_peak_aware_fairness.py` — 8 checks from Phase 3 Task 5
- Run against Task 4 experiment output
- Save `verification/fairness_check.json`
- Update documentation

**Checks:**
1. Frozen file integrity
2. Identical `res_mean` / `res_std`
3. Identical `X_all`, `y_all`
4. Identical architecture
5. Early stopping config
6. Unweighted validation
7. Shared inference/evaluation import path
8. Valid P90 threshold artifact

**Exit criterion:** All checks PASS (or documented FAIL with fix before Phase 5).

---

### Task 6 — Lock Phase 4 Implementation Record

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Consolidate implementation results into a locked record and mark Phase 4 complete for Phase 5 handoff.

#### Methodology

1. Created `experiments/peak_aware_2026-07-14_164518/phase4_implementation.json` with training, verification, and artifact references.
2. Updated implementation workspace metadata to `status: complete`.
3. Updated experiment metadata with Phase 4 lock reference.
4. Marked Phase 4 **Complete** in overview README.
5. Finalized implementation summary (§5.6) and handoff note for Phase 5.

#### Lock Record Contents

| Section | Summary |
|---------|---------|
| `implemented_modules` | All 5 new files — complete |
| `training_run` | 20 epochs, `res_mean`/`res_std` match baseline |
| `verification` | 8/8 fairness checks PASS |
| `baseline_integrity` | Production `models/` and frozen utils unmodified |
| `next_phase` | Phase 5 evaluation handoff actions |

**Record:** `experiments/peak_aware_2026-07-14_164518/phase4_implementation.json`

#### Conclusions

Phase 4 exit criteria met. Peak-Aware Hybrid implementation is locked and cleared for Phase 5 evaluation.

#### Handoff to Phase 5

See [Phase 5 — Evaluation](phase-05-evaluation.md). Treatment artifacts:

- Model: `experiments/peak_aware_2026-07-14_164518/models/hybrid_gru_peak_aware.keras`
- Stats: `experiments/peak_aware_2026-07-14_164518/models/residual_stats.pkl`
- Control: `experiments/baseline_reference_2026-07-14/`

---

### Task 6 — Lock Phase 4 Implementation Record *(plan reference)*

**Objective:** Consolidate implementation results and mark Phase 4 complete.

**Deliverables:**
- `experiments/peak_aware_<timestamp>/phase4_implementation.json` (or symlink to design run dir)
- Final Phase 4 summary in `phase-04-implementation.md`
- Update overview README — Phase 4 complete
- Handoff note for Phase 5 evaluation

**Exit criterion:** Phase 4 marked complete; ready for Phase 5.

---

## Estimated Effort

| Task | Effort |
|------|--------|
| Task 1 — Workspace + config | ~30 min |
| Task 2 — Peak detection | ~1–2 hours |
| Task 3 — Training module | ~1–2 hours |
| Task 4 — Run script + train | ~1–2 hours (+ training time) |
| Task 5 — Fairness verify | ~1 hour |
| Task 6 — Lock record | ~30 min |

**Total:** ~1–2 days including GRU training runtime.

---

## What Phase 4 Does NOT Include

- Phase 5 evaluation (baseline vs peak-aware metrics)
- Peak-subset MAE/RMSE reporting
- Modifying frozen baseline code or artifacts
- λ tuning or methodology changes
- Global GRU comparison (Phase 6)

---

*Phase 4 locked: 2026-07-14 — ready for Phase 5 evaluation*
