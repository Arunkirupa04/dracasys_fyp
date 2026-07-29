# Stage 1 — Design Lock (Peak-Aware Global GRU)

[← Study Overview](README.md)

**Status:** Complete  
**Locked date:** 2026-07-17  
**Design artifact:** `experiments/peak_aware_global_design_2026-07-17/design_lock.json`  
**Control reference (official):** `experiments/global_gru_baseline_2026-07-17_121748/`  
**Superseded control (historical):** `experiments/global_gru_reference_2026-07-17/` (patience=3)  
**Methodology authority:** `docs/peak-aware-hybrid/` (Phases 1–6 complete)

---

## 1. Inherited Methodology (from Peak-Aware Hybrid)

The following are **fixed by reference** to the completed Peak-Aware Hybrid research. They are **not** re-explored, re-tuned, or re-debated in this study.

| Locked item | Specification | Authority |
|-------------|---------------|-----------|
| Peak definition | P90, train-only, per-container, real CPU (`cpu_real >= P90(container_train)`) | `experiments/peak_exploration_2026-07-14/peak_definition_decision.json` |
| Peak weight λ | 5 | `experiments/peak_aware_design_2026-07-14/phase3_design.json` |
| Non-peak weight | 1 | Same |
| Training loss | Timestep-weighted MSE | Hybrid Phase 3 |
| Weight scope | Forecast-horizon targets only (`y_train`) | Hybrid Phase 3 |
| Early stopping | Unweighted `val_loss`; patience per baseline | Global: **patience 10** (official baseline; superseded reference used patience 3) |
| Evaluation protocol | 99 containers; Day-1 96 steps; MAE/RMSE/MAPE in real CPU % | Hybrid Phase 5 |
| Peak evaluation labels | Same P90 train-fitted thresholds on validation actuals | Hybrid Phase 5 |
| Fairness philosophy | Single controlled variable; parallel modules; frozen control | Hybrid Phase 3 |
| Verification philosophy | Pre-eval fairness gate; post-eval integrity checks | Hybrid Phases 4–5 |

**Explicitly not repeated:**

- P85/P90/P95 peak exploration (Hybrid Phase 2)
- λ selection or mechanism selection (Hybrid Phase 3)
- Peak-definition research debates

Hybrid Phases 1–3 are **cited as authority** — not re-executed.

---

## 2. Study Positioning

This study is the **Global counterpart** of the completed Peak-Aware Hybrid experiment (`experiments/peak_aware_2026-07-14_164518/`). The thesis contains **one Peak-Aware methodology** applied to two architectures:

- **Hybrid Prophet + GRU** — treatment complete and locked
- **Global GRU** — this study

The frozen Global GRU baseline (`global_gru_v1`) is the **control model** for this track. The Hybrid baseline is **not** the control here.

---

## 3. Research Questions

| Tier | Question | Answered in |
|------|----------|-------------|
| **Tier 1** | Under P90 + λ = 5, does peak-aware training change overall Day-1 accuracy vs frozen Global GRU? | Stages 3–4 |
| **Tier 2** | Does it improve accuracy on peak timesteps specifically? | Stages 3–4 |
| **Tier 3** | Does Peak-Aware learning behave differently across Hybrid vs Global architectures? | Stage 4 §4.7 (secondary synthesis only) |

---

## 4. Configuration Module Decision

### Options considered

| Option | Approach |
|--------|----------|
| **A (preferred if safe)** | Extend `utils/global_config.py` with Peak-Aware constants |
| **B (fallback)** | Create `utils/global_peak_aware_config.py` importing read-only from `global_config.py` |

### Decision: **Option B — `utils/global_peak_aware_config.py`**

**Rationale:**

1. **`utils/global_config.py` is frozen.** It is listed in `experiments/global_gru_reference_2026-07-17/config/baseline_metadata.json` under `frozen_code_files`. Modifying it would violate the Global baseline freeze contract established in Phase 0.5 and Phase 6.
2. **Hybrid precedent.** Peak-Aware Hybrid uses a separate `utils/peak_config.py` that imports shared geometry from frozen `utils/hybrid_config.py` without modifying it. Peak-Aware Global follows the same pattern.
3. **Clear naming.** `global_peak_aware_config.py` signals treatment-specific constants and paths, avoiding confusion with the frozen baseline module or a generic duplicate config.

**Implementation rule for Stage 2:**

```python
# global_peak_aware_config.py — read-only imports from frozen global_config
from utils.global_config import (
    INPUT_WINDOW, FORECAST_HORIZON, GLOBAL_FEATURES, GLOBAL_TARGET,
    OPTIMIZER, EPOCHS_MAX, BATCH_SIZE, SHUFFLE, RANDOM_SEED,
    SEQUENCE_VAL_SPLIT, EARLY_STOPPING_*, GRU_UNITS, ...,
    GLOBAL_GRU_REFERENCE_DIR, EXPECTED_N_SEQUENCES, ...
)
# Peak-aware additions only:
PEAK_PERCENTILE = 90
PEAK_WEIGHT = 5.0
MODEL_VARIANT = "global_gru_peak_aware_v1"
# Treatment artifact paths under experiments/peak_aware_global_<timestamp>/
```

Shared peak constants (P90, λ) may also be imported from `utils/peak_config.py` where values are identical, to avoid duplicating locked numbers — but treatment paths and `MODEL_VARIANT` remain in `global_peak_aware_config.py`.

---

## 5. Frozen Baseline Definition

> **Phase B baseline correction (2026-07-17):** The official Global control was promoted to `experiments/global_gru_baseline_2026-07-17_121748/` (`EarlyStopping` patience=10, MAE 1.924). The Stage 1 lock originally referenced `global_gru_reference_2026-07-17/` (patience=3, MAE 2.063); that directory is preserved as a superseded historical artifact.

| Item | Value |
|------|-------|
| **Control identifier** | `global_gru_v1` |
| **Official reference directory** | `experiments/global_gru_baseline_2026-07-17_121748/` |
| **Spec document** | `docs/global-gru-baseline/phase-03-5-baseline-specification.md` |
| **Source model** | `experiments/global_gru_baseline_2026-07-17_121748/models/global_gru.keras` |
| **Control Day-1 MAE (mean)** | **1.924** (99 containers) |
| **Superseded reference (historical)** | `experiments/global_gru_reference_2026-07-17/` — patience 3, MAE 2.063 |
| **Input window** | 96 |
| **Forecast horizon** | 96 (Day 1) |
| **Features** | `cpu_scaled`, `cpu_mean`, `cpu_std` |
| **Target** | `cpu_scaled` (direct CPU forecasting) |
| **Architecture** | GRU(128, return_sequences=True) → Dropout(0.2) → GRU(64) → Dense(64, relu) → Dense(96) |
| **Sequence source** | `global_train` only (train-period data) |
| **Sequence counts** | N = 41,650; N_train = 33,320; N_val = 8,330 |
| **Optimizer / loss** | Adam / MSE (uniform) |
| **Epochs max / batch** | 50 / 256 |
| **Early stopping** | `val_loss`, patience **10**, restore best weights |
| **Shuffle** | False |
| **Random seed** | 42 |
| **Inference** | `utils/global_inference.run_global_inference` |
| **Evaluation** | `utils/global_evaluation` + shared metric helpers |

---

## 6. Global-Specific Adaptation

### 6.1 Single experimental variable

| Component | Rule |
|-----------|------|
| Everything in frozen `global_gru_v1` specification | **Identical** |
| **Single approved change** | GRU training loss: uniform MSE → timestep-weighted MSE (λ = 5 on peak timesteps) |
| Control reference | `experiments/global_gru_baseline_2026-07-17_121748/` |
| Implementation | Parallel modules only |
| Outputs | `experiments/peak_aware_global_<timestamp>/` only |

### 6.2 Architecture-specific differences from Hybrid Peak-Aware

| Aspect | Hybrid Peak-Aware | Peak-Aware Global |
|--------|-------------------|-------------------|
| Target | `residual_scaled` | `cpu_scaled` |
| Sequence builder | `create_residual_sequences()` | `create_longterm_sequences()` |
| Input features | `(N, 96, 1)` residual | `(N, 96, 3)` cpu + stats |
| Prophet | Required (frozen) | None |
| Residual stats | `residual_stats.pkl` | None |
| Model builder | `build_hybrid_gru_model()` | `build_global_gru_model()` (parallel copy in peak-aware module) |
| Inference | `run_hybrid_inference` | `run_global_inference` (unchanged) |
| Prediction column | `day1_final_real` | `day1_pred_real` |
| ES patience | 10 (Hybrid baseline) | **10** (official Global baseline; archived primary run used 3) |
| Batch size | 64 (Hybrid baseline) | **256** (Global baseline) |
| Epochs max | 100 (Hybrid baseline) | **50** (Global baseline) |

Hyperparameters follow the **Global** frozen baseline, not the Hybrid baseline.

### 6.3 Shared peak utilities (reuse, do not duplicate)

Reuse `utils/peak_detection.py` for architecture-independent logic:

- `compute_peak_thresholds()`
- `build_sequence_weight_matrix()`
- `label_peak_timesteps()`
- `save_peak_thresholds()` / `load_peak_thresholds()`

**Extension (Stage 2, if needed):** add `align_with_longterm_sequences()` — Global analogue of `align_with_residual_sequences()`. Do **not** create a separate Global peak-detection module unless a genuine constraint cannot be handled by extending the shared module.

---

## 7. Fairness Contract

### 7.1 Identical components (control vs treatment)

| Component | Must match frozen Global baseline |
|-----------|----------------------------------|
| Preprocessing | `data/train_df.parquet`, `data/val_df.parquet`, `data/scalers.pkl`, `data/selected_containers.npy` |
| Train/val temporal split | Per-container 80/20 chronological (preprocessing) |
| Container cohort | 100 selected → 99 evaluable (`c_14674` skipped) |
| Sequence construction | `create_longterm_sequences()` on `global_train` |
| Input window / forecast horizon | 96 / 96 |
| Features and target | `(cpu_scaled, cpu_mean, cpu_std)` → `cpu_scaled` |
| GRU architecture | GRU(128→64) + Dense(64) + Dense(96) |
| Optimizer, epochs, batch, shuffle, seed | Adam, 50, 256, False, 42 |
| Sequence-level 80/20 split index | `split_idx = int(len(X_all) * 0.8)` |
| Early stopping criterion | Unweighted `val_loss`, patience **10** (official control) |
| Inference pipeline | `run_global_inference` (unchanged) |
| Primary evaluation protocol | 99-container Day-1 cohort |

### 7.2 Intentional difference (single variable)

| Component | Control | Treatment |
|-----------|---------|-----------|
| Training loss on `y_train` | Uniform MSE | Timestep-weighted MSE (λ = 5 on peak horizon steps) |

### 7.3 Mandatory sequence identity assertion (Stage 2 gate)

**Before any evaluation**, fairness verification **must** assert:

| Tensor / property | Requirement |
|-------------------|-------------|
| `X_all` | **Byte-identical** to baseline `create_longterm_sequences()` on same `global_train` |
| `y_all` | **Byte-identical** to baseline |
| `sample_cids` / sample ordering | **Identical** row order |
| Sequence count | **41,650** (`EXPECTED_N_SEQUENCES`) |
| `W_all` | **Only allowed difference** — shape `(41650, 96)`; values in `{1.0, 5.0}` |

Implementation reference (Hybrid precedent): `scripts/verify_peak_aware_fairness.py` check #3 (`identical_sequences`). Global fairness script must implement the equivalent using `create_longterm_sequences()`.

**Gate rule:** Evaluation **must not** start until this check and all other fairness checks **PASS**.

---

## 8. Component Parity Table

| Component | Frozen Global Baseline | Peak-Aware Global Treatment |
|-----------|------------------------|----------------------------|
| Config module | `utils/global_config.py` | `utils/global_peak_aware_config.py` (new) |
| Training module | `utils/global_training.py` | `utils/global_training_peak_aware.py` (new; **reuses** `build_global_gru_model()` read-only) |
| Peak detection | N/A | `utils/peak_detection.py` (shared; extend if needed) |
| Peak evaluation | N/A | Reuse `utils/peak_evaluation.py` or thin Global wrapper |
| Inference | `utils/global_inference.py` | **Same module** (unchanged) |
| Evaluation orchestration | `utils/global_evaluation.py` | **Same module** (unchanged) |
| Model artifact | `global_gru.keras` | `global_gru_peak_aware.keras` |
| Metadata artifact | `global_gru_metadata.json` | `global_gru_peak_aware_metadata.json` |
| Peak thresholds | N/A | `peak/peak_thresholds.pkl` |
| Peak config | N/A | `peak/peak_config.json` |
| Residual stats | N/A | **None** |
| Prophet | N/A | **None** |

---

## 9. Experiment Layout

```
experiments/peak_aware_global_<timestamp>/
├── design/
│   └── design_lock.json              # optional copy of Stage 1 lock
├── models/
│   ├── global_gru_peak_aware.keras
│   └── global_gru_peak_aware_metadata.json
├── peak/
│   ├── peak_thresholds.pkl
│   └── peak_config.json
├── reference/
│   └── reference_metadata.json       # baseline + design lock traceability (Stage 2)
├── training/
│   └── training_metadata.json
├── verification/
│   └── fairness_check.json           # Stage 2 gate (must PASS — 10 checks)
├── evaluation/                       # Stage 3
│   ├── evaluation_df.csv
│   ├── evaluation_summary.csv
│   ├── baseline_evaluation_df.csv
│   ├── peak_subset_*.csv
│   ├── comparison_table.csv
│   ├── per_container_delta.csv
│   ├── evaluation_summary.json
│   ├── plots/
│   └── verification_check.json
├── discussion/                       # Stage 4
│   ├── results_synthesis.json
│   └── stratified_summary.csv
└── phase_evaluation.json             # final study lock
```

---

## 10. Implementation Scope (Stage 2)

### 10.1 New modules and scripts

| File | Purpose |
|------|---------|
| `utils/global_peak_aware_config.py` | Treatment constants and paths (Option B) |
| `utils/global_training_peak_aware.py` | Weighted training on `create_longterm_sequences` |
| `utils/peak_detection.py` | Extend with `align_with_longterm_sequences()` if needed |
| `scripts/run_peak_aware_global_experiment.py` | Train + save artifacts |
| `scripts/verify_peak_aware_global_fairness.py` | Pre-eval fairness gate |

### 10.2 Stage 3 modules (planned, not Stage 1)

| File | Purpose |
|------|---------|
| `scripts/run_peak_aware_global_evaluation.py` | Treatment primary eval |
| `scripts/run_peak_aware_global_peak_subset_evaluation.py` | Control re-run + peak metrics |
| `scripts/build_peak_aware_global_comparison.py` | Tables + plots |
| `scripts/verify_peak_aware_global_evaluation.py` | Post-eval integrity |
| `notebooks/peak_aware_global_container_comparison.ipynb` | Manual inspection |

### 10.3 Frozen read-only imports (Stage 2)

- `utils.global_training.build_global_gru_model` — **mandatory reuse**; do not duplicate architecture definitions
- `utils.sequence_utils.create_longterm_sequences`
- `utils.peak_detection.*` (shared peak logic)

### 10.4 Implementation order (Stage 2)

1. `utils/global_peak_aware_config.py`
2. Extend `utils/peak_detection.py` with `align_with_longterm_sequences()` (if needed)
3. `utils/global_training_peak_aware.py`
4. `scripts/run_peak_aware_global_experiment.py` (smoke train)
5. `scripts/verify_peak_aware_global_fairness.py` — **must PASS**
6. Full training run → save artifacts

---

## 11. Out of Scope

| Item | Reason |
|------|--------|
| Peak percentile exploration (P85/P90/P95) | Locked by Hybrid Phase 2 |
| λ selection or sweep | Locked at λ = 5 by Hybrid Phase 3 |
| Mechanism selection (focal loss, oversampling, etc.) | Rejected in Hybrid Phase 3 |
| Modifying frozen Global or Hybrid baseline code/artifacts | Fair comparison requirement |
| Hyperparameter tuning | Confounds single-variable design |
| Architecture changes | Separate experiment |
| Preprocessing changes | Shared pipeline |
| Changes to evaluation protocol | Fair comparison requirement |
| Prophet or residual pipeline | Not applicable to Global |
| Hybrid vs Global methodology comparison | Already complete (`hybrid_vs_global_2026-07-17_095147`) |
| Cross-architecture synthesis as primary evaluation | Secondary only — Stage 4 §4.7 |

---

## 12. Fairness Verification Checklist (Stage 2)

| # | Check | Mandatory |
|---|-------|-----------|
| 1 | Frozen file integrity (no writes to control reference or frozen `utils/global_*.py`) | Yes |
| 2 | **Identical `X_all`, `y_all`, sample ordering, N = 41,650** | **Yes — hard gate** |
| 3 | `W_all` shape `(41650, 96)`; only values `{1.0, 5.0}` | Yes |
| 4 | Identical Global GRU architecture vs reference (`build_global_gru_model` reused) | Yes |
| 5 | Global hyperparams match `baseline_metadata.json` | Yes |
| 6 | Unweighted validation early stopping | Yes |
| 7 | Shared Global inference/evaluation path unchanged | Yes |
| 8 | Valid P90 peak threshold artifact (train-fitted; no Hybrid threshold comparison) | Yes |
| 9 | Weight matrix row order aligns with `create_longterm_sequences` | Yes |
| 10 | **Single behavioural difference** — only weighted `train_step` differs; optimizer, callbacks, batch, epochs, shuffle, split, architecture, data, sequences identical | **Yes — hard gate** |

**Task plan:** [stage-02-implementation-plan.md](stage-02-implementation-plan.md)

---

## 13. Evaluation Preview (Stage 3)

Mirrors Hybrid Phase 5 protocol with Global runners:

| Tier | Metrics |
|------|---------|
| Tier 1 (overall) | `day1_mae`, `day1_rmse`, `day1_mape` |
| Tier 2 (peak subset) | `peak_mae`, `peak_rmse`, `non_peak_mae`, `non_peak_rmse` |
| Per-container | ΔMAE, improved/worsened counts |
| Optional stratification | `pattern_type`, peak quartiles |

Delta convention: **treatment − control** (Peak-Aware Global minus frozen Global).

---

## 14. Stage 4 Preview — Architecture Sensitivity (Secondary)

After both Peak-Aware studies are complete, Stage 4 §4.7 will include a cross-architecture synthesis table:

| Architecture | Baseline Overall MAE | Peak-Aware Overall MAE | Δ Overall | Baseline Peak MAE | Peak-Aware Peak MAE | Δ Peak | … |
|--------------|---------------------|------------------------|-----------|-------------------|---------------------|--------|---|

Hybrid row populated from locked `peak_aware_2026-07-14_164518` results. Global row populated from this study. **Not part of Stage 3 primary evaluation.**

---

## 15. Implementation Checklist (Stage 1 exit)

- [x] Inherited methodology documented (§1)
- [x] Global control reference identified (§5)
- [x] Configuration module decision recorded — Option B (§4)
- [x] Fairness contract written including mandatory sequence identity (§7)
- [x] Component parity table complete (§8)
- [x] Experiment layout defined (§9)
- [x] Implementation scope and order defined (§10)
- [x] Out-of-scope items listed (§11)
- [x] `design_lock.json` written
- [x] README created with thesis methodology tree

---

## 16. Exit Criteria for Stage 1

Stage 1 is **complete** when:

1. This document and `design_lock.json` are written and internally consistent.
2. Methodology inheritance is explicit — no repeated peak/λ exploration.
3. Frozen Global baseline is the designated control.
4. Configuration decision (Option B) is justified.
5. Mandatory `X_all` / `y_all` / ordering identity rule is documented as a Stage 2 gate.
6. Implementation scope is bounded; Stage 2–4 deliverables are identified but **not implemented**.

**Stage 2 must not begin until this design lock is reviewed and approved.**

---

## 17. Next Stage

**Stage 2 — Implementation + Fairness Verification**

Deliverables: parallel training modules, experiment runner, fairness script (with mandatory sequence identity check), trained model artifacts, `fairness_check.json` with all checks PASS.

**Do not start Stage 2 until explicit approval.**
