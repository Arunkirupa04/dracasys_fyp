# Phase 5 — Evaluation

[← Overview](README.md) · [Phase 4 Implementation](phase-04-implementation.md) · [Design lock JSON](../experiments/peak_aware_design_2026-07-14/phase3_design.json) · [Phase 4 lock JSON](../experiments/peak_aware_2026-07-14_164518/phase4_implementation.json)

---

## 6. Phase 5 — Evaluation

### 6.1 Objective

Execute a **fair paired comparison** between the frozen baseline Hybrid Prophet + GRU and the Peak-Aware treatment model. Report overall Day 1 metrics plus peak-subset metrics using the locked P90 definition — without modifying frozen baseline code, retraining models, or tuning hyperparameters.

**Control:** `experiments/baseline_reference_2026-07-14/`  
**Treatment:** `experiments/peak_aware_2026-07-14_164518/`  
**Design authority:** `experiments/peak_aware_design_2026-07-14/phase3_design.json`

### 6.2 Locked Evaluation Inputs

| Item | Value |
|------|-------|
| Evaluation period | Temporal validation split — `data/val_df.parquet` |
| Train context | `data/train_df.parquet` (per-container Prophet + GRU input) |
| Cohort | 100 selected containers → **99 evaluable** (`c_14674` skipped) |
| Primary horizon | Day 1 — **96 steps** |
| Overall metrics | MAE, RMSE, MAPE (per container → aggregate mean/std) |
| Peak definition | P90 per-container, train-fitted thresholds — `peak/peak_thresholds.pkl` |
| Peak-subset metrics | Peak-only MAE, RMSE; non-peak MAE, RMSE |
| Inference / eval modules | **Frozen** `utils/hybrid_inference.py`, `utils/hybrid_evaluation.py` |

### 6.3 Baseline Reference (Control Metrics)

Frozen Day 1 aggregate metrics from Phase 1 (`baseline_metadata.json`):

| Metric | Mean |
|--------|------|
| MAE | **1.746** |
| RMSE | **2.388** |
| MAPE | **111.94** |

Per-container record: `experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv`

### 6.4 Phase 5 Constraints

| Constraint | Rule |
|------------|------|
| Frozen baseline code | Do **not** modify `utils/hybrid_*.py`, `sequence_utils.py`, notebooks, or `models/` |
| No retraining | Load pre-trained weights only |
| No λ / P90 tuning | Use locked Phase 3 / Phase 4 artifacts as-is |
| Shared protocol | Both models evaluated through identical `evaluate_selected_containers()` |
| Peak labels | Reporting only — use saved `peak_thresholds.pkl`; do not change model |
| Outputs | Save under `experiments/peak_aware_2026-07-14_164518/evaluation/` (+ optional timestamped comparison dir) |
| Phase 6 scope | Global GRU comparison and λ sensitivity deferred |

### 6.5 Metric Tiers (Locked from Phase 3)

| Tier | Metrics | Unit | Label rule |
|------|---------|------|------------|
| **Tier 1 — Overall** | MAE, RMSE, MAPE | Per container → cohort mean/std | All 96 Day-1 validation timesteps |
| **Tier 2 — Peak subset** | MAE, RMSE | Pooled across peak timesteps | `actual_cpu_real >= P90(container_train)` |
| **Tier 3 — Non-peak subset** | MAE, RMSE | Pooled across non-peak timesteps | Complement of Tier 2 |
| **Tier 4 — Optional** | MAE, RMSE by workload type | `stable` / `medium` / `spiky` | Same P90 rule; interpret `spiky` cautiously (Phase 2) |

**Primary research question (Tier 1):** Does peak-aware training change overall Day 1 accuracy vs baseline?

**Contribution question (Tier 2):** Does peak-aware training improve accuracy specifically on peak timesteps?

### 6.6 Task Plan

| Task | Focus | Status |
|------|-------|--------|
| 1 | Prepare evaluation workspace + `peak_evaluation.py` | **Complete** |
| 2 | Run treatment primary evaluation (peak-aware) | **Complete** |
| 3 | Compute peak-subset metrics (both models) | **Complete** |
| 4 | Build comparison tables + plots | **Complete** |
| 5 | Verification + sanity checks | **Complete** |
| 6 | Lock Phase 5 evaluation record | **Complete** |

---

### Task 1 — Prepare Evaluation Workspace + `peak_evaluation.py`

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Add peak-subset metric utilities as a parallel module without modifying frozen `hybrid_evaluation.py`.

#### Methodology

1. Created `utils/peak_evaluation.py` with Day-1 peak labelling and subset metric functions.
2. Reused `label_peak_timesteps()` from `peak_detection.py` for consistent P90 rule.
3. Initialized evaluation workspace under `experiments/peak_aware_2026-07-14_164518/evaluation/`.
4. Ran single-container sanity check on `c_11461` against manual MAE/RMSE calculation.

#### Functions Implemented

| Function | Purpose |
|----------|---------|
| `label_day1_peak_timesteps()` | P90 peak mask on Day-1 `actual_day1_real` |
| `compute_peak_subset_metrics()` | Per-container peak / non-peak / overall MAE, RMSE |
| `compute_peak_subset_metrics_batch()` | Batch wrapper over inference results dict |
| `summarize_peak_subset_metrics()` | Cohort-level **pooled** peak / non-peak MAE, RMSE |
| `compare_overall_metrics()` | Treatment vs control overall summary helper |
| `compare_peak_subset_summaries()` | Treatment vs control pooled subset helper |
| `build_overall_evaluation_summary()` | Thin wrapper over frozen `summarize_evaluation_metrics` |

#### Sanity Check Results (`c_11461`)

| Check | Result |
|-------|--------|
| Peak MAE vs manual | **Match** |
| Peak RMSE vs manual | **Match** |
| Non-peak MAE vs manual | **Match** |
| Non-peak RMSE vs manual | **Match** |
| Overall MAE / RMSE vs manual | **Match** |
| Peak step count | **14 / 96** (14.6% peak fraction) |
| Pooled summary (single container) | **Match** |
| Pipeline status | **PASS** |

**Records:**
- `experiments/peak_aware_2026-07-14_164518/evaluation/peak_evaluation_sanity.json`
- `experiments/peak_aware_2026-07-14_164518/evaluation/evaluation_workspace.json`

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Separate `peak_evaluation.py` module | Keeps frozen `hybrid_evaluation.py` unchanged |
| Peak labels on **actual** CPU % | Matches locked Phase 2/3 evaluation reporting rule |
| Pooled cohort metrics concatenate timesteps | Phase 3 protocol: subset MAE/RMSE pooled across timesteps |
| Reuse frozen `compute_day1_metrics` for overall | Guarantees per-container overall metrics match baseline eval |

#### Conclusions

Task 1 exit criterion met: peak-subset functions verified on demo container.

#### Next Task

**Task 2 — Run treatment primary evaluation:** `scripts/run_peak_aware_evaluation.py`; evaluate 99 containers via frozen `evaluate_selected_containers()`.

---

### Task 1 — Prepare Evaluation Workspace + `peak_evaluation.py` *(plan reference)*

**Objective:** Add peak-subset metric utilities without modifying frozen evaluation modules.

**Deliverables:**
- `utils/peak_evaluation.py`:
  - `label_day1_peak_timesteps()` — apply P90 thresholds to Day-1 `actual_day1_real`
  - `compute_peak_subset_metrics()` — pooled peak / non-peak MAE, RMSE from inference results
  - `summarize_peak_subset_metrics()` — cohort-level aggregation
  - `compare_overall_metrics()` — treatment vs control summary helper
- Initialize evaluation metadata under treatment experiment dir

**Design notes:**
- Input: `inference_results` dict from frozen `evaluate_selected_containers()` (contains `actual_day1_real`, `day1_final_real` per container)
- Thresholds: `load_peak_thresholds()` from treatment `peak/peak_thresholds.pkl`
- Do **not** add peak logic to `utils/hybrid_evaluation.py`

**Exit criterion:** Peak-subset functions verified on one container (`c_11461`) against manual calculation.

---

### Task 2 — Run Treatment Primary Evaluation

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Evaluate the peak-aware treatment model on the frozen 99-container validation cohort using unchanged `evaluate_selected_containers()`.

#### Methodology

1. Created `scripts/run_peak_aware_evaluation.py`.
2. Loaded treatment artifacts from `experiments/peak_aware_2026-07-14_164518/models/`.
3. Called frozen `evaluate_selected_containers()` with shared data and scalers.
4. Saved per-container metrics, aggregate summary, metadata, and inference cache for Task 3.

#### Treatment Evaluation Results

| Item | Value |
|------|-------|
| Evaluated containers | **99** |
| Skipped | **1** (`c_14674` — same as baseline) |
| Failures | **0** |
| Runtime | **~25 s** |
| Mean Day-1 MAE | **1.741** |
| Mean Day-1 RMSE | **2.384** |
| Mean Day-1 MAPE | **110.17** |

**Baseline reference (control):** MAE **1.746**, RMSE **2.388**, MAPE **111.94**

Early observation: treatment overall metrics are marginally lower than baseline on all three primary metrics. Formal comparison and peak-subset analysis are Tasks 3–4.

#### Saved Artifacts

| Output | Location |
|--------|----------|
| Per-container metrics | `evaluation/evaluation_df.csv` |
| Aggregate summary | `evaluation/evaluation_summary.csv` |
| Evaluation metadata | `evaluation/evaluation_metadata.json` |
| Inference cache (Task 3) | `evaluation/treatment_inference_cache.pkl` |

#### Implementation Decisions

| Decision | Rationale |
|----------|-----------|
| Frozen `evaluate_selected_containers` only | Fairness contract — identical protocol to baseline |
| Save lightweight inference cache | Avoids re-running 99 Prophet fits in Task 3 peak-subset pass |
| No baseline re-run in Task 2 | Control overall metrics already frozen in `baseline_reference` |

#### Conclusions

Task 2 exit criterion met: 99 containers evaluated, 0 failures, output format matches baseline reference.

#### Next Task

**Task 3 — Compute peak-subset metrics (both models):** re-run control inference; apply `peak_evaluation.py` to both.

---

### Task 2 — Run Treatment Primary Evaluation *(plan reference)*

**Objective:** Evaluate peak-aware model through the frozen pipeline.

**Deliverables:**
- `scripts/run_peak_aware_evaluation.py` (or split runner + comparison script)
- Load from `experiments/peak_aware_2026-07-14_164518/models/`:
  - `hybrid_gru_peak_aware.keras`
  - `residual_stats.pkl`
- Call frozen `evaluate_selected_containers()` with unchanged arguments
- Save:
  - `evaluation/evaluation_df.csv`
  - `evaluation/evaluation_summary.csv`
  - `evaluation/evaluation_metadata.json` (runtime, container counts, artifact paths)

**Exit criterion:** 99 containers evaluated (0 failures); shapes match baseline reference format.

---

### Task 3 — Compute Peak-Subset Metrics (Both Models)

**Status:** Complete  
**Date:** 2026-07-14

#### Objective

Compute per-container and pooled peak-subset Day-1 metrics for both treatment and control using the same P90 thresholds.

#### Methodology

1. Created `scripts/run_peak_subset_evaluation.py`.
2. **Treatment:** Reused `treatment_inference_cache.pkl` from Task 2.
3. **Control:** Re-ran `evaluate_selected_containers()` with `baseline_reference/models/hybrid_gru.keras`.
4. Applied `peak_evaluation.py` to both models with `peak/peak_thresholds.pkl`.
5. Saved per-container, long-format, comparison, and pooled summaries.

#### Results

| Item | Value |
|------|-------|
| Containers (each model) | **99** |
| Pooled peak timesteps | **2,438 / 9,504** (**25.65%**) |
| Phase 2 Day-1 feasibility range | **17–31%** — **within range** |

**Pooled peak-subset (Tier 2):**

| Model | Peak MAE | Peak RMSE | Non-peak MAE |
|-------|----------|-----------|--------------|
| Baseline | **2.501** | **5.186** | **1.485** |
| Peak-aware | **2.526** | **5.204** | **1.470** |

**Baseline re-run overall (sanity):** MAE **1.746**, RMSE **2.388** — matches frozen `baseline_reference` exactly.

#### Saved Artifacts

| Output | Location |
|--------|----------|
| Control overall eval | `evaluation/baseline_evaluation_df.csv` |
| Long-format peak metrics | `evaluation/peak_subset_metrics.csv` |
| Treatment per-container | `evaluation/treatment_peak_subset_per_container.csv` |
| Baseline per-container | `evaluation/baseline_peak_subset_per_container.csv` |
| Side-by-side comparison | `evaluation/peak_subset_per_container_comparison.csv` |
| Pooled summary | `evaluation/peak_subset_summary.csv` |
| Metadata | `evaluation/peak_subset_metadata.json` |

#### Conclusions

Task 3 exit criterion met: 99/99 containers; pooled peak fraction feasible; per-container peak metrics saved for both models.

#### Next Task

**Task 5 — Verification + sanity checks:** `scripts/verify_peak_aware_evaluation.py`; save `evaluation/verification_check.json`.

---

### Task 4 — Build Comparison Tables + Plots

**Status:** Complete  
**Script:** `scripts/build_peak_aware_comparison.py`  
**Completed:** 2026-07-14

#### Methodology

1. Read existing Task 2/3 artifacts only — no re-inference or retraining.
2. Built `comparison_table.csv` from frozen `baseline_reference` overall metrics + Task 3 pooled peak-subset summary.
3. Built `per_container_delta.csv` by pairing treatment and baseline re-run `evaluation_df.csv` files.
4. Saved lock-ready `evaluation_summary.json`.
5. Generated thesis plots under `evaluation/plots/` (PNG + PDF).

#### Results — Overall (Tier 1, frozen baseline vs peak-aware)

| Metric | Baseline | Peak-aware | Δ (treatment − baseline) |
|--------|----------|------------|--------------------------|
| MAE | **1.746** | **1.741** | **−0.005** |
| RMSE | **2.388** | **2.384** | **−0.003** |
| MAPE | **111.94** | **110.17** | **−1.77** |

#### Results — Peak-subset (Tier 2, pooled)

| Metric | Baseline | Peak-aware | Δ |
|--------|----------|------------|---|
| Peak MAE | **2.501** | **2.526** | **+0.024** |
| Peak RMSE | **5.186** | **5.204** | **+0.018** |
| Non-peak MAE | **1.485** | **1.470** | **−0.015** |
| Non-peak RMSE | **3.491** | **3.451** | **−0.040** |

**Per-container ΔMAE:** mean **−0.005**; 45 containers improved, 54 worsened.

#### Saved Artifacts

| Output | Location |
|--------|----------|
| Comparison table | `evaluation/comparison_table.csv` |
| Per-container deltas | `evaluation/per_container_delta.csv` |
| Machine-readable summary | `evaluation/evaluation_summary.json` |
| Actual vs predicted (3 samples) | `evaluation/plots/actual_vs_predicted_samples.png` |
| Error distribution | `evaluation/plots/error_distribution.png` |
| Peak-subset bar chart | `evaluation/plots/peak_subset_comparison.png` |

#### Conclusions

Task 4 exit criterion met: comparison table complete; plots saved as PNG under `evaluation/plots/`.

#### Next Task

**Task 6 — Lock Phase 5 evaluation record.**

---

### Task 5 — Verification + Sanity Checks

**Status:** Complete  
**Script:** `scripts/verify_peak_aware_evaluation.py`  
**Completed:** 2026-07-14

#### Methodology

1. Audited frozen `hybrid_evaluation.py` / `hybrid_inference.py` (existence, import, no uncommitted git changes).
2. Verified treatment artifacts (`hybrid_gru_peak_aware.keras`, treatment `residual_stats.pkl`).
3. Confirmed control baseline re-run matches frozen `baseline_reference` per-container MAE.
4. Validated shared 99-container cohort (`c_14674` skipped).
5. Recomputed overall metrics from inference caches via `compute_day1_metrics()`.
6. Compared saved `peak_thresholds.pkl` against train-fitted recompute (no drift).
7. Confirmed pooled peak fraction **25.65%** within Phase 2 range (**17–31%**).
8. Confirmed no git changes under `baseline_reference/`; treatment model not in production `models/`.

#### Results

| Check | Status |
|-------|--------|
| 1 — Frozen eval modules | **PASS** |
| 2 — Treatment artifacts | **PASS** |
| 3 — Control baseline model | **PASS** |
| 4 — Shared cohort | **PASS** |
| 5 — Metrics reproducible | **PASS** |
| 6 — Peak thresholds stable | **PASS** |
| 7 — Peak-subset feasibility | **PASS** |
| 8 — Protected paths unmodified | **PASS** |

**Overall: 8/8 PASS** → `evaluation/verification_check.json`

#### Conclusions

Task 5 exit criterion met: all evaluation integrity checks passed; results ready for Phase 5 lock (Task 6).

#### Next Task

**Task 6 — Lock Phase 5 evaluation record** (`phase5_evaluation.json`, README update, handoff note).

---

### Task 3 — Compute Peak-Subset Metrics (Both Models) *(plan reference)*

**Objective:** Fair peak-subset comparison requires timestep-level arrays for **both** control and treatment.

**Methodology:**
1. **Treatment:** Reuse inference results from Task 2.
2. **Control:** Run `evaluate_selected_containers()` loading baseline model from `experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras` and matching `residual_stats.pkl` — read-only, no baseline code changes.
3. Apply `compute_peak_subset_metrics()` to both inference result sets with the **same** `peak_thresholds.pkl`.
4. Build per-container and pooled peak / non-peak summaries.

**Deliverables:**
- `evaluation/peak_subset_metrics.csv` (per-container peak/non-peak MAE, RMSE)
- `evaluation/peak_subset_summary.csv` (pooled cohort stats)
- `evaluation/baseline_evaluation_df.csv` (control re-run for paired comparison)
- `evaluation/comparison_table.csv` (overall + peak-subset side-by-side)

**Exit criterion:** Peak-subset metrics computed for ≥99% of evaluable containers; pooled peak timestep count consistent with Phase 2 validation feasibility (~17–31% Day-1 peak rate).

---

### Task 4 — Build Comparison Tables + Plots

**Status:** Complete — see execution record above.  
**Objective:** Produce thesis-ready comparison artifacts.

**Deliverables:**

| Output | Content |
|--------|---------|
| `comparison_table.csv` | Baseline vs peak-aware: overall MAE/RMSE/MAPE + peak/non-peak MAE/RMSE |
| `per_container_delta.csv` | Per-container ΔMAE, ΔRMSE (treatment − control) |
| `evaluation_summary.json` | Machine-readable comparison record |
| Plots (optional but recommended) | Actual vs Predicted (sample containers), error distribution, peak vs non-peak bar chart |

**Comparison rules:**
- Overall Tier 1: compare treatment `evaluation_summary.csv` vs frozen `baseline_reference` aggregates
- Peak Tier 2/3: compare pooled peak-subset summaries from Task 3
- Report **both** absolute values and deltas (treatment − control)

**Exit criterion:** Comparison table complete; plots saved as PNG under `evaluation/plots/`.

---

### Task 5 — Verification + Sanity Checks

**Status:** Complete — see execution record above.  
**Objective:** Audit evaluation integrity before locking results.

**Deliverables:**
- `scripts/verify_peak_aware_evaluation.py` (or extend verification section in runner)
- Save `evaluation/verification_check.json`

**Checks:**

| # | Assertion |
|---|-----------|
| 1 | Frozen `hybrid_evaluation.py` / `hybrid_inference.py` unmodified |
| 2 | Treatment uses correct artifacts (`hybrid_gru_peak_aware.keras`, treatment `residual_stats.pkl`) |
| 3 | Control uses baseline reference model (not peak-aware weights) |
| 4 | Same 99-container cohort as `baseline_reference` |
| 5 | Overall metrics reproducible via `compute_day1_metrics()` on inference outputs |
| 6 | Peak thresholds identical to `peak_thresholds.pkl` (no recompute drift) |
| 7 | Peak-subset pooled count > 0 and within Phase 2 feasibility range |
| 8 | No writes to `models/` or `baseline_reference/` |

**Exit criterion:** All checks PASS.

---

### Task 6 — Lock Phase 5 Evaluation Record

**Status:** Complete  
**Lock record:** `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`  
**Completed:** 2026-07-14

**Objective:** Consolidate results and mark Phase 5 complete.

**Deliverables:**
- `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`
- Final summary section in this document
- Update `docs/peak-aware-hybrid/README.md` — Phase 5 complete
- Handoff note for Phase 6 (discussion, limitations, thesis narrative)

**Exit criterion:** Phase 5 marked complete; results ready for Phase 6 interpretation.

---

### 6.7 Planned Artifact Layout

```
experiments/peak_aware_2026-07-14_164518/evaluation/
├── evaluation_df.csv                  # treatment overall (Task 2)
├── evaluation_summary.csv
├── baseline_evaluation_df.csv         # control re-run (Task 3)
├── peak_subset_metrics.csv            # per-container peak/non-peak (Task 3)
├── peak_subset_summary.csv            # pooled peak/non-peak (Task 3)
├── comparison_table.csv               # baseline vs peak-aware (Task 4)
├── per_container_delta.csv            # paired deltas (Task 4)
├── evaluation_metadata.json
├── evaluation_summary.json            # lock-ready comparison (Task 4)
├── verification_check.json            # Task 5
└── plots/
    ├── actual_vs_predicted_*.png
    ├── error_distribution.png
    └── peak_subset_comparison.png
```

### 6.8 New Files (Planned)

| File | Role |
|------|------|
| `utils/peak_evaluation.py` | Peak-subset metric computation (parallel to frozen eval) |
| `scripts/run_peak_aware_evaluation.py` | Orchestrate Tasks 2–4 |
| `scripts/verify_peak_aware_evaluation.py` | Task 5 verification gate |

**Frozen (read-only):** `utils/hybrid_evaluation.py`, `utils/hybrid_inference.py`, `utils/peak_detection.py` (load thresholds only)

### 6.9 Exit Criteria (Before Phase 6)

- [x] Treatment primary evaluation complete (99 containers)
- [x] Control evaluation re-run for paired peak-subset comparison
- [x] Overall comparison vs `baseline_reference` documented
- [x] Peak-subset and non-peak-subset metrics reported
- [x] Comparison table and metadata saved
- [x] Verification checks PASS
- [x] Frozen baseline code unmodified
- [x] Phase 6 discussion **not** started in this phase

### 6.10 Known Limitations (Carry Forward)

From Phase 2–3 — interpret results with these in mind:

1. **Near-idle zero-threshold artefact** — ~29% of train peak labels from 5 near-idle containers at P90
2. **Train/val peak rate divergence** — validation peaks ~28% vs train ~17% under fixed thresholds
3. **Spiky workload mislabelling** — high-CV near-idle containers inflate spiky-group stats
4. **MAPE instability** — near-zero CPU values; epsilon-floor MAPE used (same as baseline)
5. **Paired peak-subset requires inference re-run for control** — frozen `evaluation_df.csv` lacks timestep arrays

### 6.11 Estimated Effort

| Task | Effort |
|------|--------|
| Task 1 — `peak_evaluation.py` | ~1–2 hours |
| Task 2 — Treatment eval | ~30–60 min (Prophet per container) |
| Task 3 — Peak-subset (both models) | ~1–2 hours |
| Task 4 — Tables + plots | ~1 hour |
| Task 5 — Verification | ~30 min |
| Task 6 — Lock record | ~30 min |

**Total:** ~1 day including Prophet inference runtime (~2× 99 containers).

### 6.12 What Phase 5 Does NOT Include

- Retraining or fine-tuning either model
- λ sensitivity analysis (Phase 6)
- Global GRU comparison (Phase 6)
- Modifying frozen baseline code or `models/`
- Changing peak definition or thresholds using validation data
- Architecture or inference pipeline changes

---

## 6.13 Phase 5 — Final Summary (Locked)

**Status:** **Complete**  
**Lock record:** `experiments/peak_aware_2026-07-14_164518/phase5_evaluation.json`  
**Locked:** 2026-07-14

### Evaluation Outcome

Fair paired comparison on 99 containers (Day-1, 96 steps, P90 peak definition):

| Tier | Question | Result |
|------|----------|--------|
| **Tier 1 — Overall** | Does peak-aware training improve aggregate Day-1 metrics? | **Marginal yes** — MAE −0.005, RMSE −0.003 (not practically significant) |
| **Tier 2 — Peak subset** | Does it improve accuracy on peak timesteps? | **No** — peak MAE +0.024, peak RMSE +0.018 |
| **Tier 3 — Contribution** | Does peak-aware GRU loss help peak forecasting? | **Not demonstrated** under locked λ=5, P90 protocol |

**Per-container:** mean ΔMAE −0.005; **45 improved / 54 worsened** (high variance).

### Deliverables Completed

| Category | Artifacts |
|----------|-----------|
| Treatment eval | `evaluation/evaluation_df.csv`, `evaluation_summary.csv` |
| Control + peak subset | `baseline_evaluation_df.csv`, `peak_subset_*.csv` |
| Comparison | `comparison_table.csv`, `per_container_delta.csv`, `evaluation_summary.json` |
| Plots | `evaluation/plots/` |
| Verification | `verification_check.json` (8/8 PASS) |
| Manual inspection | `notebooks/peak_aware_container_comparison.ipynb` |
| Lock record | `phase5_evaluation.json` |

### Handoff to Phase 6

Phase 5 is **measurement only**. Interpretation belongs in [Phase 6 — Discussion](phase-06-discussion.md):

1. Why overall metrics improved slightly while peak metrics did not
2. Train/eval objective mismatch (weighted train loss vs unweighted Day-1 eval)
3. Prophet vs GRU error attribution at peak timesteps
4. Limitations from §6.10 and design lock (λ=5 not empirically tuned)
5. Thesis narrative: valid negative/refinement result for peak-aware residual training

**Do not** retrain, retune λ, or modify frozen baseline when entering Phase 6 unless explicitly scoped as a new experiment.

---

*Phase 5 locked: 2026-07-14 — all tasks complete; ready for Phase 6 interpretation*
