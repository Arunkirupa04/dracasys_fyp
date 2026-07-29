# Phase 4 — Implementation

[← Overview](README.md) · [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md) · [Phase 3 Pipeline Design](phase-03-pipeline-design.md)

---

## 5. Phase 4 — Implementation

### 5.1 Objective

Implement the **locked [Phase 0.5 specification](phase-03-5-baseline-specification.md)** using a reuse-first module layout and a refactored notebook. Fix validation-target leakage, train the Global GRU v1 model, and save experiment artifacts — **without** modifying the frozen Hybrid baseline or running Phase 5 evaluation metrics.

**Specification authority:** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)

### 5.2 Locked Inputs (From Phase 0.5)

| Item | Value |
|------|-------|
| Forecasting methodology | Global GRU v1 — direct CPU forecasting |
| Model form | 128→64 GRU, Dropout(0.2), Dense(64)→Dense(96) |
| Features | `cpu_scaled`, `cpu_mean`, `cpu_std` |
| Target | `cpu_scaled` |
| Input window / horizon | 96 / 96 |
| Sequence source | `global_train` only |
| Early stopping | Unweighted `val_loss`, patience 3 |
| Seed | 42 |
| Hybrid control | `experiments/baseline_reference_2026-07-14/` |

### 5.3 Implementation Philosophy

During Phase 4, the following are **strictly prohibited**:

| Prohibited | Reason |
|------------|--------|
| Peak-Aware Learning | Out of scope for baseline |
| Model-structure optimisation | Would change the locked specification |
| Hyperparameter tuning | Notebook values are locked as-is |
| Additional features | Would change the experimental configuration |
| Preprocessing changes | Shared pipeline must remain identical |
| Duplicating Hybrid modules | Reuse shared utilities instead |

The objective is a **scientifically valid, reproducible baseline** — not maximised accuracy.

### 5.4 Phase 4 Constraints

| Constraint | Rule |
|------------|------|
| Frozen Hybrid | Do **not** modify `utils/hybrid_*.py`, `notebooks/hybrid_model.ipynb`, `models/hybrid_*`, or `baseline_reference/` |
| Frozen preprocessing | Do **not** modify `data/` or `preprocessing.ipynb` |
| Peak-Aware | Do **not** import `utils/peak_*.py` or peak-aware training modules |
| Shared utilities | **Reuse** read-only imports from `hybrid_config`, `hybrid_evaluation`; **extend** `sequence_utils.py` only |
| New modules | **Only** Global GRU–specific logic in `utils/global_*.py` |
| Outputs | Development runs under `experiments/global_gru_baseline_<timestamp>/` |
| Evaluation metrics | Do **not** run Phase 5 multi-container evaluation in this phase |
| Specification | Do **not** deviate from Phase 0.5 locked configuration |

### 5.5 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Extend `utils/sequence_utils.py` with `create_longterm_sequences()` | **Complete** |
| 2 | Implement `utils/global_config.py` (Global-only constants) | **Complete** |
| 3 | Implement `utils/global_training.py` | **Complete** |
| 4 | Implement `utils/global_inference.py` | **Complete** |
| 5 | Implement `utils/global_artifacts.py` | **Complete** |
| 6 | Refactor `notebooks/global_gru_model.ipynb` (two-section workflow) | **Complete** |
| 7 | Fix data leakage (remove `concat(train, val)`) | **Complete** |
| 8 | Train model and save to `models/global_gru.keras` | **Complete** |
| 9 | Lock Phase 4 implementation record and freeze baseline | **Complete** |

---

### Implementation Review — Architecture Correction (2026-07-17)

During Task 3 implementation review, a mismatch was found between the original `notebooks/global_gru_model.ipynb` prototype and the Phase 0.5 / Phase 2 / Phase 3 documentation regarding the Dropout layer.

| Source | Dropout (before correction) |
|--------|----------------------------|
| `notebooks/global_gru_model.ipynb` | `Dropout(0.2)` after first GRU |
| Phase 0.5 / Phase 2 / Phase 3 docs | Incorrectly documented as none |
| `utils/global_training.py` (initial Task 3) | No Dropout (followed incorrect written spec) |

**Resolution (Option A):** The notebook is the architectural source of truth. Phase 2, Phase 3, and Phase 0.5 have been corrected; `global_config.DROPOUT` and `build_global_gru_model()` now include `Dropout(0.2)` after the first GRU layer.

This is a **documentation correction** to restore fidelity to the intended baseline — **not** an experimental modification. Engineering improvements (modularisation, leakage fixes, reproducibility, verification, artifact management) proceed unchanged.

**Corrected architecture (locked):**

```
GRU(128, return_sequences=True) → Dropout(0.2) → GRU(64) → Dense(64, relu) → Dense(96)
```

Phase 4 implementation resumes at Task 4 after this correction.

---

### Task 1 — Extend `utils/sequence_utils.py`

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Add `create_longterm_sequences()` as a **shared** utility parallel to `create_residual_sequences()`, eliminating the inline duplicate in `global_gru_model.ipynb` (notebook update deferred to Task 6).

#### Methodology

1. Read existing `create_residual_sequences()` in `utils/sequence_utils.py` and the inline notebook implementation.
2. Added `create_longterm_sequences()` as the canonical Global GRU entry point.
3. Extracted shared logic into private `_create_sliding_window_sequences()` so both public functions delegate to one implementation — preserves Hybrid backward compatibility without duplicating loop code.
4. Added type hints, module docstring, and function docstrings per `python-coding-standards`.
5. Verified sequence counts from `global_train` only using project venv (`tf_metal_env`).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Private `_create_sliding_window_sequences()` helper | Single source of truth; `create_residual_sequences()` behaviour unchanged |
| `create_longterm_sequences()` as separate public API | Clear Global GRU import path; notebook can drop inline definition in Task 6 |
| Docstring notes `global_train` only | Documents baseline leakage rule at the function boundary |
| No notebook changes in Task 1 | Task scope limited to shared utility; notebook refactor is Task 6 |

#### Code Changes

| File | Change |
|------|--------|
| `utils/sequence_utils.py` | Added module docstring, type hints, `create_longterm_sequences()`, `_create_sliding_window_sequences()`; refactored `create_residual_sequences()` to delegate to helper |

#### Verification (run in `tf_metal_env`)

```bash
tf_metal_env/bin/python -c "
import numpy as np, pandas as pd
from utils.sequence_utils import create_residual_sequences, create_longterm_sequences
train_df = pd.read_parquet('data/train_df.parquet')
selected = np.load('data/selected_containers.npy', allow_pickle=True)
global_train = train_df[train_df['container_id'].isin(selected)]
X_h, y_h, _ = create_residual_sequences(global_train, ['cpu_scaled'], 'cpu_scaled')
X_g, y_g, _ = create_longterm_sequences(global_train, ['cpu_scaled'], 'cpu_scaled')
assert np.array_equal(X_h, X_g)  # backward compatibility
X3, y3, _ = create_longterm_sequences(
    global_train, ['cpu_scaled','cpu_mean','cpu_std'], 'cpu_scaled')
print(X3.shape)  # expect (41650, 96, 3)
"
```

#### Verification Results

| Check | Result |
|-------|--------|
| `create_longterm_sequences` importable | **PASS** |
| Backward compat: identical output to `create_residual_sequences` for same inputs | **PASS** |
| `global_train` sequence count (3 features) | **41,650** |
| `X` shape | **(41650, 96, 3)** |
| `y` shape | **(41650, 96)** |
| Compare to old leaky notebook count (`concat` train+val) | **57,445 → 41,650** (confirms train-only scope) |
| `scripts/verify_hybrid_train_eval_split.py` (full Hybrid regression) | **PASS** — 99 containers, train/load equivalence identical |

#### Findings

- **41,650** sequences from `global_train` with 100 selected containers (99 present in train data) — matches Peak-Aware / Hybrid residual sequence count (`N = 41,650`).
- The old notebook produced **57,445** sequences because it used `concat(global_train, global_val)` — Task 7 will ensure training never uses that path.
- With 3 features, `X` has shape `(41650, 96, 3)`; Hybrid residual path remains `(41650, 96, 1)` when using one feature.

#### Conclusions

Task 1 is complete. `create_longterm_sequences()` is available in `utils/sequence_utils.py` for all subsequent Global GRU modules. Hybrid backward compatibility is preserved via the shared helper.

#### Limitations

- Notebook still contains inline duplicate — removed in Task 6.
- ~~Full Hybrid verification script suite not re-run in this task~~ Hybrid train/load regression check **passed** (2026-07-17).

#### Next Task

**Task 2 — Implement `utils/global_config.py`:** Global-only constants and paths; read-only imports from `hybrid_config` for shared window/horizon.

---

### Task 2 — Implement `utils/global_config.py`

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Create a single configuration module for **Global GRU–only** constants and paths. Reuse shared window/horizon from frozen `hybrid_config`; do not duplicate Hybrid modules or pull heavy evaluation imports into config.

#### Methodology

1. Reviewed locked values in [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md).
2. Followed `utils/peak_config.py` pattern: import geometry from `hybrid_config` only.
3. Defined Global-specific features, training hyperparameters, model form, paths, and verified sequence counts from Task 1.
4. Set `MAPE_EPSILON = 0.01` locally with sync comment (avoids importing `hybrid_evaluation`, which loads inference dependencies).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Import `INPUT_WINDOW` / `FORECAST_HORIZON` from `hybrid_config` | Shared geometry; prevents drift from Hybrid baseline |
| `MAPE_EPSILON` defined locally, not imported | Keeps config module lightweight; value must match frozen `hybrid_evaluation` |
| `GLOBAL_FEATURES` as `tuple[str, ...]` | Immutable; matches Phase 0.5 locked feature list |
| `EXPECTED_N_SEQUENCES = 41_650` | From Task 1 verification on `global_train` |
| `pathlib.Path` for all paths | Matches `peak_config` and project standards |
| No training or inference code in config | Configuration only — logic stays in later tasks |

#### Generated Outputs

| Output | Location |
|--------|----------|
| Global GRU configuration module | `utils/global_config.py` |

#### `global_config.py` Constants Summary

| Category | Key constants |
|----------|---------------|
| Identity | `MODEL_VARIANT="global_gru_v1"` |
| Geometry (from `hybrid_config`) | `INPUT_WINDOW=96`, `FORECAST_HORIZON=96` |
| Features / target | `GLOBAL_FEATURES`, `GLOBAL_TARGET="cpu_scaled"` |
| Training | `OPTIMIZER`, `LOSS`, `EPOCHS_MAX=50`, `BATCH_SIZE=256`, `RANDOM_SEED=42`, early stopping patience `3` |
| Model form | `GRU_UNITS=(128, 64)`, `DENSE_UNITS=64`, `DROPOUT=0.2` |
| Evaluation | `DEMO_CONTAINER_ID="c_11461"`, `MAPE_EPSILON=0.01` |
| Sequence counts | `EXPECTED_N_SEQUENCES=41650`, `N_train=33320`, `N_val=8330` |
| Paths | `DATA_*`, `DEFAULT_MODEL_PATH`, `HYBRID_BASELINE_REFERENCE_DIR` |

#### Verification

```bash
tf_metal_env/bin/python -c "
from utils.global_config import (
    MODEL_VARIANT, INPUT_WINDOW, FORECAST_HORIZON,
    GLOBAL_FEATURES, RANDOM_SEED, EXPECTED_N_SEQUENCES, MAPE_EPSILON,
)
from utils.hybrid_config import DEFAULT_INPUT_WINDOW, DAY1_HORIZON
assert MODEL_VARIANT == 'global_gru_v1'
assert INPUT_WINDOW == DEFAULT_INPUT_WINDOW == 96
assert FORECAST_HORIZON == DAY1_HORIZON == 96
assert GLOBAL_FEATURES == ('cpu_scaled', 'cpu_mean', 'cpu_std')
assert EXPECTED_N_SEQUENCES == 41650
assert MAPE_EPSILON == 0.01
print('PASS')
"
```

#### Conclusions

Task 2 is complete. All Phase 0.5 locked constants and paths are centralized in `utils/global_config.py`. Subsequent tasks (training, inference, artifacts) import from this module.

#### Next Task

**Task 3 — Implement `utils/global_training.py`:** `build_global_gru_model()` and `train_global_gru()` using `create_longterm_sequences()` and `global_config` constants.

---

### Task 3 — Implement `utils/global_training.py`

**Status:** Complete (corrected 2026-07-17)  
**Date:** 2026-07-17

#### Objective

Implement Global GRU model construction and training in a dedicated module, following the locked Phase 0.5 specification and importing all constants from `utils/global_config.py`.

#### Methodology

1. Reviewed `utils/hybrid_training.py` for Keras Sequential / compile / fit patterns.
2. Compared against `notebooks/global_gru_model.ipynb` (legacy inline training).
3. Implemented `set_random_seeds()`, `build_global_gru_model()`, and `train_global_gru()`.
4. Wired sequence generation through `create_longterm_sequences()` with `GLOBAL_FEATURES` / `GLOBAL_TARGET`.
5. Added hard assertions on `EXPECTED_N_SEQUENCES`, `EXPECTED_N_TRAIN`, and `EXPECTED_N_VAL` to catch leakage or wrong input dataframes at training time.
6. Smoke-tested model build and sequence counts (full 50-epoch training deferred to Task 8).
7. **Correction (2026-07-17):** Restored `Dropout(0.2)` after first GRU to match notebook and corrected Phase 0.5 spec (see Implementation Review above).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| `DROPOUT=0.2` per corrected Phase 0.5 — matches notebook | Notebook has `Dropout(0.2)` after first GRU; baseline must preserve original architecture |
| `Dropout(DROPOUT)` always after first GRU | Fixed locked value from `global_config`; matches `global_gru_model.ipynb` cell order |
| `train_global_gru(global_train, ...)` takes pre-filtered train df | Caller filters `train_df` to selected containers; function enforces sequence-count contract |
| Return `(model, training_metadata)` dict | Artifact saving is Task 5; metadata captures hyperparameters and final epoch metrics |
| Lazy TensorFlow imports inside functions | Matches `hybrid_training.py`; avoids import cost when only inspecting module |
| 80/20 index split for early stopping only | Same protocol as notebook; Phase 5 evaluation uses temporal holdout per container |

#### Functions

| Function | Purpose |
|----------|---------|
| `set_random_seeds(seed)` | Python / NumPy / TensorFlow seed 42 |
| `build_global_gru_model(input_window, n_features, forecast_horizon)` | GRU(128)→Dropout(0.2)→GRU(64)→Dense(64,relu)→Dense(96) |
| `train_global_gru(global_train, ...)` | Sequences → split → fit → return model + metadata |

#### Training Pipeline

1. Set random seeds (42)
2. Build sequences from `global_train` via `create_longterm_sequences()`
3. Assert `N = 41_650`; split 80/20 → 33_320 train / 8_330 val sequences
4. Build model via `build_global_gru_model()` — **matches notebook architecture**
5. Compile: Adam, MSE, MAE metric — **no hyperparameter tuning**
6. Fit with `shuffle=False`, EarlyStopping(patience=3, `restore_best_weights=True`)
7. Return trained model + `training_metadata` dict

#### Code Changes

| File | Change |
|------|--------|
| `utils/global_training.py` | **NEW** — `set_random_seeds`, `build_global_gru_model`, `train_global_gru` |
| `utils/global_config.py` | `DROPOUT=0.2` (corrected from erroneous `0.0`) |

#### Must Not

- Concatenate `global_val` into training dataframe
- Use sequence val split for final metrics
- Tune hyperparameters or modify model structure

#### Verification (run in `tf_metal_env`)

```bash
tf_metal_env/bin/python -c "
import numpy as np, pandas as pd
from utils.global_config import (
    INPUT_WINDOW, FORECAST_HORIZON, GLOBAL_FEATURES, GLOBAL_TARGET,
    GRU_UNITS, DENSE_UNITS, DROPOUT, DATA_TRAIN, DATA_SELECTED_CONTAINERS,
    EXPECTED_N_SEQUENCES, EXPECTED_N_TRAIN, EXPECTED_N_VAL,
)
from utils.global_training import build_global_gru_model, set_random_seeds
from utils.sequence_utils import create_longterm_sequences

set_random_seeds(42)
model = build_global_gru_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
assert [l.__class__.__name__ for l in model.layers] == ['GRU', 'Dropout', 'GRU', 'Dense', 'Dense']
assert DROPOUT == 0.2

dummy = np.zeros((2, INPUT_WINDOW, len(GLOBAL_FEATURES)))
out = model.predict(dummy, verbose=0)
assert out.shape == (2, FORECAST_HORIZON)

train_df = pd.read_parquet(DATA_TRAIN)
selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
global_train = train_df[train_df['container_id'].isin(selected)]
X, y, cids = create_longterm_sequences(
    global_train, list(GLOBAL_FEATURES), GLOBAL_TARGET,
    input_window=INPUT_WINDOW, forecast_horizon=FORECAST_HORIZON,
)
assert len(X) == EXPECTED_N_SEQUENCES == 41650
assert int(len(X) * 0.8) == EXPECTED_N_TRAIN == 33320
assert len(X) - int(len(X) * 0.8) == EXPECTED_N_VAL == 8330
print('PASS')
"
```

#### Verification Results

| Check | Result |
|-------|--------|
| Module importable | **PASS** |
| Model layers: GRU → Dropout → GRU → Dense → Dense | **PASS** (after correction) |
| `DROPOUT == 0.2` | **PASS** |
| Predict output shape `(batch, 96)` | **PASS** |
| `global_train` sequence count | **41,650** |
| Train / val sequence split | **33,320 / 8,330** |

#### Findings

- Initial Task 3 followed the written Phase 0.5 spec (`Dropout=None`), which incorrectly omitted the notebook's `Dropout(0.2)`. Corrected before Task 4.
- `train_global_gru()` expects the caller to pass filtered `global_train` (selected containers); passing raw `train_df.parquet` fails the sequence-count assertion by design.

#### Conclusions

Task 3 is complete. Global GRU training logic is centralized in `utils/global_training.py` with architecture matching `global_gru_model.ipynb`. Ready for inference (Task 4), artifact persistence (Task 5), and notebook refactor (Task 6). Full model training and save remain Task 8.

#### Next Task

**Task 4 — Implement `utils/global_inference.py`:** `GlobalInferenceResult` dataclass and `run_global_inference()` for per-container temporal holdout prediction.

---

### Task 4 — Implement `utils/global_inference.py`

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Implement per-container Global GRU Day 1 inference using the locked temporal holdout protocol: last 96 train feature steps → predict first 96 val steps → inverse MinMax to real CPU %.

#### Methodology

1. Reviewed `utils/hybrid_inference.py` for dataclass and inverse-transform patterns.
2. Followed Phase 0.5 / Phase 3 inference sequence (one canonical forecast per container, not sliding windows over val).
3. Implemented `GlobalInferenceResult` dataclass and `run_global_inference()`.
4. Smoke-tested on demo container `c_11461` with untrained model (shape and protocol verification).

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Input = last `input_window` rows of `GLOBAL_FEATURES` | Matches Phase 3 locked inference sequence |
| Target = first `forecast_horizon` val steps of `cpu_scaled` | Temporal holdout Day 1 protocol identical to Hybrid |
| Real CPU via per-container `scalers[container_id]` | Same inverse MinMax protocol as Hybrid baseline |
| Validate train/val length and scaler presence | Fail fast with clear errors before `model.predict()` |
| Dataclass fields per Phase 4 spec only | Phase 5 evaluation imports metrics from `hybrid_evaluation` |

#### Functions

| Function / type | Purpose |
|-----------------|---------|
| `GlobalInferenceResult` | Dataclass holding Day 1 scaled/real predictions and metadata |
| `run_global_inference(container_id, global_train, global_val, model, scalers, ...)` | Single-container temporal holdout inference |

#### Inference Pipeline

1. Extract and sort train/val slices for `container_id`
2. Assert sufficient train (`≥96`) and val (`≥96`) steps
3. Build `X_input` from last 96 train steps of `[cpu_scaled, cpu_mean, cpu_std]`
4. `model.predict()` → `day1_pred_scaled` (length 96)
5. Ground truth: first 96 val steps of `cpu_scaled`
6. Inverse MinMax → `day1_pred_real`, `actual_day1_real`
7. Return `GlobalInferenceResult`

#### Code Changes

| File | Change |
|------|--------|
| `utils/global_inference.py` | **NEW** — `GlobalInferenceResult`, `run_global_inference` |

#### Verification (run in `tf_metal_env`)

```bash
tf_metal_env/bin/python -c "
import pickle, numpy as np, pandas as pd
from utils.global_config import DATA_TRAIN, DATA_VAL, DATA_SCALERS, DEMO_CONTAINER_ID, FORECAST_HORIZON, INPUT_WINDOW
from utils.global_training import build_global_gru_model
from utils.global_inference import run_global_inference

train_df = pd.read_parquet(DATA_TRAIN)
val_df = pd.read_parquet(DATA_VAL)
selected = np.load('data/selected_containers.npy', allow_pickle=True)
global_train = train_df[train_df['container_id'].isin(selected)]
global_val = val_df[val_df['container_id'].isin(selected)]
with open(DATA_SCALERS, 'rb') as f:
    scalers = pickle.load(f)

model = build_global_gru_model()
model.compile(optimizer='adam', loss='mse')
result = run_global_inference(DEMO_CONTAINER_ID, global_train, global_val, model, scalers)
assert result.day1_pred_scaled.shape == (FORECAST_HORIZON,)
assert result.day1_pred_real.shape == (FORECAST_HORIZON,)
assert result.actual_day1_real.shape == (FORECAST_HORIZON,)
assert result.input_window == INPUT_WINDOW == 96
print('PASS')
"
```

#### Verification Results

| Check | Result |
|-------|--------|
| Module importable | **PASS** |
| Demo container `c_11461` inference | **PASS** |
| Output shapes `(96,)` for scaled and real arrays | **PASS** |
| `input_window == 96` | **PASS** |

#### Conclusions

Task 4 is complete. Per-container Global GRU inference is ready for artifact persistence (Task 5) and evaluation orchestration (Phase 5 `global_evaluation.py`).

#### Next Task

**Task 5 — Implement `utils/global_artifacts.py`:** `save_global_artifacts()` and `load_global_artifacts()` for model + JSON metadata persistence.

---

### Task 5 — Implement `utils/global_artifacts.py`

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Persist and reload the trained Global GRU Keras model plus a JSON metadata record for reproducibility and Phase 6 baseline freeze.

#### Methodology

1. Reviewed `utils/hybrid_artifacts.py` for save/load patterns and default path handling.
2. Implemented JSON metadata (Global GRU has no residual-stats pickle like Hybrid).
3. Auto-enriched metadata on save with `saved_at`, `environment`, and `baseline_spec` when absent.
4. Smoke-tested round-trip save/load in a temporary directory.

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| JSON metadata (not pickle) | Human-readable provenance; matches Phase 0.5 artifact design |
| `save_global_artifacts()` enriches metadata | Ensures Phase 0.5 spec snapshot, training timestamp, and environment are always recorded |
| `baseline_spec` built from `global_config` | Single source of truth for locked hyperparameters and architecture |
| Default paths from `global_config` | `models/global_gru.keras`, `models/global_gru_metadata.json` |
| Return enriched metadata from `save_global_artifacts()` | Caller can log exactly what was written |
| `load_global_artifacts()` returns `(model, metadata)` | Matches Phase 5 evaluation notebook pattern |

#### Functions

| Function | Purpose |
|----------|---------|
| `save_global_artifacts(model, metadata, model_path, metadata_path)` | Save Keras model + JSON metadata |
| `load_global_artifacts(model_path, metadata_path)` | Load model + metadata dict |

#### Metadata enrichment (on save)

| Key | Source |
|-----|--------|
| `saved_at` | UTC ISO timestamp (if not provided) |
| `environment` | Python, platform, TensorFlow versions (if not provided) |
| `baseline_spec` | Locked Phase 0.5 fields from `global_config` (if not provided) |
| `model_path`, `metadata_path` | Paths written |

Callers pass `training_metadata` from `train_global_gru()`; save merges training record with provenance fields.

#### Code Changes

| File | Change |
|------|--------|
| `utils/global_artifacts.py` | **NEW** — `save_global_artifacts`, `load_global_artifacts` |

#### Verification (run in `tf_metal_env`)

```bash
tf_metal_env/bin/python -c "
import tempfile
from pathlib import Path
from utils.global_training import build_global_gru_model
from utils.global_artifacts import save_global_artifacts, load_global_artifacts

with tempfile.TemporaryDirectory() as tmp:
    model_path = Path(tmp) / 'global_gru.keras'
    metadata_path = Path(tmp) / 'global_gru_metadata.json'
    model = build_global_gru_model()
    model.compile(optimizer='adam', loss='mse')
    save_global_artifacts(model, {'epochs_run': 0}, model_path, metadata_path)
    loaded_model, loaded_metadata = load_global_artifacts(model_path, metadata_path)
    assert [l.__class__.__name__ for l in loaded_model.layers] == ['GRU', 'Dropout', 'GRU', 'Dense', 'Dense']
    assert 'saved_at' in loaded_metadata
    assert 'environment' in loaded_metadata
    assert loaded_metadata['baseline_spec']['dropout'] == 0.2
    print('PASS')
"
```

#### Verification Results

| Check | Result |
|-------|--------|
| Round-trip save/load | **PASS** |
| Model architecture preserved after reload | **PASS** |
| `saved_at`, `environment`, `baseline_spec` in metadata | **PASS** |
| `baseline_spec.dropout == 0.2` | **PASS** |

#### Conclusions

Task 5 is complete. Global GRU artifacts can be persisted to `models/global_gru.keras` + `models/global_gru_metadata.json` and reloaded for notebook evaluation (Task 6) and full training (Task 8).

#### Next Task

**Task 6 — Refactor `notebooks/global_gru_model.ipynb`:** two-section `RUN_TRAINING` workflow using shared utils modules.

---

### Task 6 — Refactor `notebooks/global_gru_model.ipynb`

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Replace the legacy exploratory notebook with a clean two-section workflow mirroring `hybrid_model.ipynb`, delegating all logic to shared `utils/global_*.py` modules.

#### Methodology

1. Removed legacy inline training, leaky `concat(train, val)` path, and exploratory cells.
2. Rebuilt notebook with shared configuration, data loading, Section 1 (training), and Section 2 (demo evaluation).
3. Wired imports with `importlib.reload()` for iterative development.
4. Verified notebook structure programmatically (no inline sequence builder, no leakage patterns).

#### Section Structure

| Section | Guard | Content |
|---------|-------|---------|
| Shared configuration | Always | Imports, `RUN_TRAINING`, paths, `RANDOM_SEED` |
| Shared data loading | Always | Frozen `data/` artifacts + `selected_containers.npy` |
| Section 1 — Training | `if RUN_TRAINING:` | `train_global_gru()` → `save_global_artifacts()` |
| Section 2 — Evaluation | Always | `load_global_artifacts()` → `run_global_inference()` → metrics → plot |

#### Cleanup completed

| Removed / fixed | Replacement |
|-----------------|-------------|
| Inline `create_longterm_sequences()` | `utils/sequence_utils.py` (via training module) |
| `pd.concat([global_train, global_val])` | Training uses `global_train` only in `train_global_gru()` |
| `global_model` variable name | `global_gru_model` |
| `hybrid_mae` mislabel | `global_mae`, `global_rmse`, `global_mape` |
| `selected_containers` regeneration + `np.save` | Load frozen `data/selected_containers.npy` |
| Demo container `c_11307` | `DEMO_CONTAINER_ID` (`c_11461`) per Phase 0.5 |
| Inline Keras model definition | `build_global_gru_model()` via `train_global_gru()` |

#### Configuration defaults

```python
RUN_TRAINING = False
MODEL_PATH = "../models/global_gru.keras"
METADATA_PATH = "../models/global_gru_metadata.json"
```

#### Verification

```bash
tf_metal_env/bin/python -c "
import json
from pathlib import Path
nb = json.loads(Path('notebooks/global_gru_model.ipynb').read_text())
src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
assert 'def create_longterm_sequences' not in src
assert 'pd.concat' not in src
assert 'global_model' not in src
assert 'hybrid_mae' not in src
assert 'train_global_gru' in src and 'run_global_inference' in src
print('PASS')
"
```

#### Verification Results

| Check | Result |
|-------|--------|
| Two-section `RUN_TRAINING` workflow | **PASS** |
| No inline `create_longterm_sequences()` | **PASS** |
| No `concat(global_train, global_val)` | **PASS** |
| `global_gru_model` naming | **PASS** |
| `global_mae` metrics naming | **PASS** |
| `importlib.reload()` for utils | **PASS** |
| Demo container `c_11461` | **PASS** |

#### Notes

- Section 2 requires saved artifacts at `models/global_gru.keras` (produced in Task 8 when `RUN_TRAINING = True`).
- Full 99-container evaluation remains Phase 5 scope.

#### Conclusions

Task 6 is complete. The notebook is a thin orchestration layer over shared Global GRU modules, aligned with the Hybrid notebook pattern and Phase 0.5 protocol.

#### Next Task

**Task 7 — Fix data leakage verification:** implement `scripts/verify_global_gru_data_split.py` and document leakage gates.

---

### Task 7 — Fix Data Leakage

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Implement a pre-training verification gate ensuring Global GRU sequences are built from `global_train` only, with zero preprocessing validation-period targets in the training set.

#### Methodology

1. Implemented `scripts/verify_global_gru_data_split.py` as the leakage audit gate.
2. Training path already fixed in Tasks 1, 3, and 6 (`train_global_gru(global_train)`; notebook has no `concat`).
3. Script compares correct (`global_train` only) vs legacy leaky (`concat(train, val)`) sequence builders.
4. Audits target timestamps against per-container train-period end times.

#### Verification checks

| Check | Requirement |
|-------|-------------|
| Sequence source | `create_longterm_sequences(global_train)` only for training |
| Sequence count | `41,650` sequences (frozen cohort) |
| Train/val split | `33,320 / 8,330` for early stopping |
| Val-period target leaks (train only) | **0** |
| Val-period target leaks (train 80% split) | **0** |
| Legacy concat contrast | Leak count **> 0** (demonstrates why concat was removed) |
| Notebook static audit | No `pd.concat([global_train, global_val])` |

#### Code changes

| File | Change |
|------|--------|
| `scripts/verify_global_gru_data_split.py` | **NEW** — leakage verification gate |

#### Run verification

```bash
tf_metal_env/bin/python scripts/verify_global_gru_data_split.py
```

#### Verification results (2026-07-17)

| Metric | Result |
|--------|--------|
| Sequences (`global_train` only) | **41,650** |
| Sequences (legacy `concat`) | **56,869** |
| Val-period leaks (train only) | **0** |
| Val-period leaks (legacy concat) | **15,120** |
| Val-period leaks (train 80% split) | **0** |
| Notebook concat pattern | **Absent** |
| Script exit code | **PASS** |

#### Conclusions

Task 7 is complete. The training pipeline is leakage-free by construction; the verification script provides a reproducible pre-training gate before Task 8 model training.

#### Next Task

**Task 8 — Train and save model:** run `RUN_TRAINING = True`, persist `models/global_gru.keras` and metadata.

---

### Task 8 — Train and Save Model

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Train Global GRU v1 on leakage-free `global_train` sequences, persist production artifacts, and record a timestamped experiment directory.

#### Methodology

1. Added `scripts/run_global_gru_baseline_training.py` (verification gate → train → save).
2. Ran pre-training `verify_global_gru_data_split.py` gate.
3. Trained via `train_global_gru(global_train, seed=42, verbose=1)`.
4. Saved production artifacts with `save_global_artifacts()`.
5. Copied model + metadata to `experiments/global_gru_baseline_<timestamp>/`.
6. Verified load + demo inference on `c_11461`.

#### Training run

```bash
tf_metal_env/bin/python scripts/run_global_gru_baseline_training.py
```

#### Outputs

| Artifact | Location |
|----------|----------|
| Trained model | `models/global_gru.keras` |
| Metadata | `models/global_gru_metadata.json` |
| Experiment copy | `experiments/global_gru_baseline_2026-07-17_074743/` |
| Training record | `experiments/global_gru_baseline_2026-07-17_074743/training/training_metadata.json` |

#### Training results (seed 42)

| Metric | Value |
|--------|-------|
| Epochs run (early stopping) | **9** |
| Final train loss | **0.0175** |
| Final val loss | **0.0293** |
| Final train MAE | **0.0912** |
| Final val MAE | **0.1140** |
| Sequences | **41,650** (33,320 train / 8,330 val) |
| Training wall time | **~56 s** (Apple M1 Pro, TF Metal) |

#### Post-save smoke test (demo container `c_11461`)

| Metric | Value (real CPU %) |
|--------|-------------------|
| Day 1 MAE | **2.7000** |
| Day 1 RMSE | **3.4544** |

*Note: These are single-container demo metrics from the saved model smoke test, not Phase 5 cohort evaluation.*

#### Metadata record includes

- Training hyperparameters and sequence counts from `train_global_gru()`
- `epochs_run`, `final_train_loss`, `final_val_loss`, `final_train_mae`, `final_val_mae`
- `saved_at`, `environment`, full `baseline_spec` (Phase 0.5 snapshot)
- `random_seed: 42`, `dropout: 0.2`

#### Code changes

| File | Change |
|------|--------|
| `scripts/run_global_gru_baseline_training.py` | **NEW** — reproducible training + artifact save |
| `models/global_gru.keras` | **NEW** — trained production model |
| `models/global_gru_metadata.json` | **NEW** — enriched metadata record |
| `experiments/global_gru_baseline_2026-07-17_074743/` | **NEW** — timestamped experiment artifacts |

#### Conclusions

Task 8 is complete. Global GRU v1 is trained and persisted. Section 2 of `notebooks/global_gru_model.ipynb` can now run with `RUN_TRAINING = False`. Phase 5 multi-container evaluation is the next phase.

#### Next Task

**Task 9 — Lock Phase 4 implementation record:** finalize exit criteria checklist and phase conclusion.

---

## 5.6 Target Code Layout (After Phase 4)

```
utils/
├── hybrid_*.py              # FROZEN — read-only imports permitted
├── sequence_utils.py        # EXTENDED — create_longterm_sequences (shared)
├── global_config.py         # NEW — Global-only constants and paths
├── global_training.py       # NEW — Global GRU training only
├── global_inference.py      # NEW — Global GRU inference only
├── global_artifacts.py      # NEW — Global GRU save/load only
└── global_evaluation.py     # Phase 5 — thin orchestration (delegates to shared metrics)

notebooks/
├── hybrid_model.ipynb       # FROZEN
└── global_gru_model.ipynb   # REFACTORED

models/
├── hybrid_gru.keras         # FROZEN
├── residual_stats.pkl       # FROZEN
├── global_gru.keras         # NEW
└── global_gru_metadata.json # NEW
```

**Not created:** duplicate metric modules, duplicate config for shared constants, or full mirrors of `hybrid_training.py` / `hybrid_evaluation.py`.

---

## 5.7 Exit Criteria (Before Phase 5)

- [x] Phase 0.5 specification approved
- [x] `create_longterm_sequences()` in `sequence_utils.py`; Hybrid scripts still pass
- [x] Global GRU–specific modules implemented (reuse-first, no unnecessary duplication)
- [x] Notebook refactored to two-section `RUN_TRAINING` workflow
- [x] No `concat(global_train, global_val)` in training path
- [x] Model trained and saved to `models/global_gru.keras`
- [x] Metadata JSON saved with full Phase 0.5 specification
- [x] `verify_global_gru_data_split.py` passes
- [x] Frozen Hybrid files unmodified (`utils/hybrid_*.py`, Hybrid reference, Hybrid models)
- [x] No hyperparameter tuning or model optimisation performed
- [x] Phase 5 multi-container evaluation **not** started
- [x] `docs/global-gru-baseline/phase-04-implementation.md` updated task-by-task
- [x] Frozen reference created at `experiments/global_gru_reference_2026-07-17/`
- [x] `FREEZE_STATEMENT.md` and `config/baseline_metadata.json` recorded

---

## 5.8 Baseline Governance

> The Global GRU v1 baseline is now frozen. Any future modifications — including Peak-Aware Learning, architecture changes, preprocessing changes, feature engineering, or hyperparameter optimisation — must be implemented as separate experiments using this frozen baseline as the control. The baseline itself must remain unchanged to preserve reproducibility and ensure fair experimental comparison.

**Frozen reference:** `experiments/global_gru_reference_2026-07-17/`  
**Production artifacts:** `models/global_gru.keras`, `models/global_gru_metadata.json`  
**Future control policy:** Peak-Aware Global GRU and all subsequent studies compare against this reference — never retrain or overwrite it.

---

## 5.9 Conclusion

Phase 4 is **complete**. All nine implementation tasks have been finished. The implementation conforms to the approved [Phase 0.5 Baseline Specification](phase-03-5-baseline-specification.md): leakage-free training from `global_train` only, notebook-faithful architecture with `Dropout(0.2)`, shared evaluation protocol hooks, and persisted reproducible artifacts.

The Global GRU v1 baseline is **fully reproducible** and **officially frozen** at `experiments/global_gru_reference_2026-07-17/`. Implementation is complete. The project is ready to enter [Phase 5 — Evaluation & Verification](phase-05-evaluation.md), which will populate cohort metrics under the shared 99-container protocol.

**No methodology comparison** between Hybrid Prophet + GRU and Global GRU is made at the end of Phase 4. Single-container demo metrics from notebook smoke tests are not suitable for research conclusions.

**Next phase:** [Phase 5 — Evaluation & Verification](phase-05-evaluation.md)

---

### Task 9 — Freeze Baseline and Lock Phase 4 Record

**Status:** Complete  
**Date:** 2026-07-17

#### Objective

Officially freeze the Global GRU v1 baseline at the same governance standard as the Hybrid control reference, verify all Phase 4 exit criteria, and close the implementation phase.

#### Methodology

1. Verified every Phase 4 exit criterion (see §5.7).
2. Re-ran `verify_global_gru_data_split.py` and `verify_hybrid_train_eval_split.py`.
3. Created immutable reference at `experiments/global_gru_reference_2026-07-17/`.
4. Wrote `FREEZE_STATEMENT.md`, `config/baseline_metadata.json`, and `verification/verification_notes.md`.
5. Copied trained model, metadata, and training record from Task 8 production artifacts.
6. Reserved `evaluation/` and `plots/` for Phase 5 cohort outputs.

#### Frozen reference record

| Item | Value |
|------|-------|
| Baseline name | Global GRU v1 (`global_gru_v1`) |
| Freeze date | **2026-07-17** |
| Experiment identifier | `global_gru_baseline_2026-07-17_074743` |
| Production model | `models/global_gru.keras` |
| Production metadata | `models/global_gru_metadata.json` |
| Reference directory | `experiments/global_gru_reference_2026-07-17/` |
| Phase 0.5 conformance | Architecture, features, training protocol, seed 42, `Dropout(0.2)` |
| Baseline status | **FROZEN** — must not be modified |

#### Phase 4 exit criterion verification

| Criterion | Result |
|-----------|--------|
| Phase 0.5 spec followed | **PASS** |
| Leakage-free training | **PASS** (`verify_global_gru_data_split.py`) |
| Hybrid regression | **PASS** (`verify_hybrid_train_eval_split.py`) |
| Model + metadata saved | **PASS** |
| Notebook refactored | **PASS** |
| No hyperparameter tuning | **PASS** |
| Phase 5 not started | **PASS** |
| Frozen reference created | **PASS** |

#### Code and artifact changes

| File / directory | Change |
|------------------|--------|
| `experiments/global_gru_reference_2026-07-17/` | **NEW** — frozen baseline reference |
| `experiments/global_gru_reference_2026-07-17/FREEZE_STATEMENT.md` | **NEW** |
| `experiments/global_gru_reference_2026-07-17/config/baseline_metadata.json` | **NEW** |
| `experiments/global_gru_reference_2026-07-17/verification/verification_notes.md` | **NEW** |
| `utils/global_config.py` | Added `GLOBAL_GRU_REFERENCE_DIR` |
| `docs/global-gru-baseline/phase-04-implementation.md` | Task 9, exit criteria, governance, conclusion |

#### Conclusions

Task 9 is complete. The Global GRU v1 baseline is officially frozen. Phase 5 may begin using this reference as the control for cohort evaluation and extended verification.

---
