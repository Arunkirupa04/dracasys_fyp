# Module 1 — Prioritized Action Plan

**Date:** July 2026  
**Status:** Planned — not yet implemented  
**Purpose:** Ordered roadmap for fixing the current implementation, completing Module 1 research, and planning future work

---

## Overview

This plan is ordered so you fix **methodological correctness first**, then **complete defensible research results**, then **harden the codebase**, and only then pursue **future extensions**.

```
PHASE A  Fix current implementation (critical)     ← do first
PHASE B  Complete Module 1 research experiments
PHASE C  Engineering & reproducibility
PHASE D  Future research extensions                ← only after A–C
```

**Rules:**

- Do **not** write thesis comparison results until **Phase A** is complete.
- Do **not** claim generalization until **Phase B.1** is complete.
- Do **not** start **Phase D** until at least **A + B2** are done.

**Related documents:**

- [Methodology Gap Analysis](methodology-gap-analysis.md) — what is wrong today
- [Recommended Evaluation Protocol](recommended-evaluation-protocol.md) — target evaluation methodology
- [Project Audit](project-audit.md) — current repository status

---

## PHASE A — Fix Current Implementation (Critical)

**Goal:** Make the existing notebooks methodologically correct and internally consistent.

### A1. Fix Global GRU Data Leakage — Highest Priority

**Problem:** `global_gru_model.ipynb` concatenates train + val, builds sequences, then splits 80/20 by index. Approximately 26.6% of sequences target the validation period; ~21% can end up in training.

**Changes:**

1. Build sequences from `global_train` only.
2. Use an 80/20 split on train sequences for early stopping only.
3. Evaluate final forecasts on `global_val` (preprocessing validation period).
4. For each container: take the last 96 train timesteps as input, predict the first 96 val timesteps (same logic Hybrid uses for Day 1).

**Done when:** Global GRU never trains on validation-period targets, and evaluation uses the temporal holdout.

**Files:** `notebooks/global_gru_model.ipynb`

---

### A2. Decide and Lock the Hybrid Input Window — High Priority

**Problem:** Research spec says 96-step input; Hybrid uses 288.

**Choose one (document in thesis + `docs/`):**

| Option | Action |
|--------|--------|
| **A — Align with spec** | Change Hybrid to 96-step input |
| **B — Research extension** | Keep 288; document as extended residual context |

**Recommendation:** Run a small ablation (96 vs 288) once A3 is in place, then lock the choice. Do not compare architectures until this is fixed.

**Files:** `notebooks/hybrid_model.ipynb`, `notebooks/hhybrid_pred.ipynb`

---

### A3. Unify Evaluation Protocol Across Both Models — High Priority

**Problem:** Hybrid evaluated on temporal holdout (`c_11461`); Global GRU on sequence split + different container (`c_11307`). Metrics are not comparable.

**Changes:**

1. Evaluate **both** models on the **same 100 containers**.
2. Use the **same validation period** (preprocessing val split).
3. Primary horizon: **96 steps (Day 1 only)**.
4. Report **MAE, RMSE, MAPE** in **real CPU %** (after inverse MinMax).
5. Aggregate over all 100 containers (mean ± std), not one demo container.

**Done when:** One results table compares both models fairly.

**Files:** `notebooks/global_gru_model.ipynb`, `notebooks/hybrid_model.ipynb`

---

### A4. Fix Preprocessing Artifact Gaps — Medium Priority

**Problem:** `scalers.pkl` is required but never saved by preprocessing. Stale `.npy` files do not match current Hybrid config.

**Changes:**

1. Add `pickle.dump(scalers, "../data/scalers.pkl")` to `preprocessing.ipynb`.
2. Delete or archive stale `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy` (or regenerate with current config).
3. Remove unused TensorFlow imports from preprocessing.

**Done when:** Full pipeline runs from raw CSV without manual/orphan artifacts.

**Files:** `notebooks/preprocessing.ipynb`

---

### A5. Clean Up Code Duplication and Naming — Medium Priority

**Problem:** Duplicated logic, confusing names, split inference notebook.

**Changes:**

1. Use `utils/sequence_utils.py` everywhere; remove inline duplicates in notebooks.
2. Extend `sequence_utils.py` with `create_longterm_sequences()` (Global GRU variant).
3. Rename Hybrid's `global_model` → `hybrid_gru_model`.
4. Fix mislabeled `hybrid_mae` in Global GRU notebook.
5. Decide fate of `hhybrid_pred.ipynb`: merge into hybrid notebook as an inference-only section, or keep as a thin wrapper that only loads artifacts.

**Done when:** One shared sequence utility; no shadowed imports.

**Files:** `utils/sequence_utils.py`, all model notebooks

---

### A6. Document Prophet Seasonality Decision — Medium Priority

**Problem:** Spec requires weekly seasonality; implementation disables it.

**Action:**

1. Try `weekly_seasonality=True` on a few containers.
2. If unstable (likely with ~6.4 days of train data), keep it off and **document why** in thesis and [methodology-gap-analysis.md](methodology-gap-analysis.md).
3. Optionally add `yearly_seasonality=False` explicitly for clarity.

**Done when:** Decision is recorded with evidence, not left implicit.

---

### A7. Add Random Seeds — Medium Priority

**Changes:** Set seeds for Python, NumPy, and TensorFlow at the top of every model notebook.

**Done when:** Re-running a notebook produces identical weights (given same hardware).

**Files:** All model notebooks

---

### Phase A Checklist

```
[ ] A1  Global GRU: train on train period only; eval on val period
[ ] A2  Lock Hybrid input window (96 or 288) with documented rationale
[ ] A3  Unified eval: same containers, period, metrics, real CPU %
[ ] A4  Preprocessing saves scalers.pkl; stale .npy removed
[ ] A5  Deduplicate sequence utils; fix naming
[ ] A6  Prophet seasonality decision documented
[ ] A7  Random seeds in all notebooks
```

**Estimated effort:** 1–2 weeks focused work

---

## PHASE B — Complete Module 1 Research (Before Thesis Results)

**Goal:** Produce defensible experiments that satisfy research objectives.

### B1. Unseen Container Evaluation — High Priority

**Research objective:** *Evaluate forecasting performance on unseen containers*

**Changes:**

1. Replace `unique()[:100]` with a **stratified split** using stable/medium/spiky labels from preprocessing.
2. Suggested split from 486 containers:

   | Set | Containers | Purpose |
   |-----|------------|---------|
   | Train | 80 | Model training |
   | Validation | 10 | Early stopping / tuning |
   | Test | 10 | Never seen during training (generalization) |

3. Save split to `data/container_splits.json` or `.npy`.
4. Report metrics separately for seen-val and unseen-test containers.

**Done when:** You can report generalization on 10 unseen containers.

---

### B2. Run Primary Comparison Experiment — High Priority

After Phase A is complete:

1. Train Hybrid and Global GRU on the same 80 train containers.
2. Evaluate on the same 10 val containers (temporal holdout).
3. Evaluate on the same 10 test containers (unseen).
4. Save everything to a timestamped folder:

```
experiments/2026-XX-XX_primary-comparison/
├── config.yaml
├── metrics.json
├── predictions/
├── models/
└── plots/
```

**Done when:** One experiment folder contains reproducible comparison results.

---

### B3. Persist Global GRU Model — Medium Priority

**Changes:**

1. Add `global_model.save("../models/global_gru.keras")` after training.
2. Save hyperparameters alongside the model.

**Done when:** Both architectures have saved artifacts.

---

### B4. Expand Evaluation Outputs — Medium Priority

**Add:**

1. **MAPE** (handle near-zero CPU values carefully).
2. Actual vs Predicted plots with train/val boundary and validation region shading.
3. Residual plots and error distribution histograms.
4. Export all plots as PNG + PDF (not notebook-only).
5. Per-container and aggregate metrics tables.

**Done when:** Results are thesis-ready without re-running notebooks manually.

---

### B5. Hybrid Day 2 as Secondary Experiment — Lower Priority

Day 2 recursive forecast (192 steps) exceeds the 96-step spec.

**Action:** Report Day 1 (96 steps) as **primary**. Report Day 2 separately as an extension showing error compounding.

---

### Phase B Checklist

```
[ ] B1  Stratified train/val/test container split
[ ] B2  Primary comparison experiment (timestamped, saved)
[ ] B3  Global GRU model persisted
[ ] B4  MAPE + publication plots exported
[ ] B5  Day 2 reported as secondary metric
```

**Estimated effort:** 2–3 weeks (includes experiment runs and analysis)

---

## PHASE C — Engineering & Reproducibility

**Goal:** Move from notebook prototype to maintainable research codebase.

Do this **in parallel with or after Phase B**, depending on FYP timeline.

### C1. Create `requirements.txt`

Pin versions from `tf_metal_env`. Regenerate `scalers.pkl` with the pinned sklearn version.

### C2. Extract Notebooks into Python Modules

Suggested layout:

```
src/
├── config/
│   └── forecast_config.py      # 96/96 window, 15min freq, paths
├── data/
│   └── loader.py
├── preprocessing/
│   └── pipeline.py
├── features/
│   └── sequences.py            # move sequence_utils here
├── models/
│   ├── hybrid.py               # Prophet + GRU
│   └── global_gru.py
├── evaluation/
│   ├── metrics.py              # MAE, RMSE, MAPE
│   └── plots.py
└── experiments/
    └── run_experiment.py       # CLI entry point
```

Keep notebooks for exploration; move final logic to modules per `python-coding-standards` skill.

### C3. Add Root `README.md`

Link to `docs/`, describe setup, data placement, and how to run experiments.

### C4. Standardize Experiment Logging

Every run logs: seeds, hyperparameters, git commit hash, data version, metrics, artifact paths.

### C5. Version-Control Policy

Decide what to commit:

| Type | Policy |
|------|--------|
| Code, docs, configs | Commit |
| Raw CSV, large `.npy`, venv | Gitignore |
| Trained models | Commit if small enough, or document download instructions / DVC |

---

### Phase C Checklist

```
[ ] C1  requirements.txt
[ ] C2  Modular src/ package
[ ] C3  Root README.md
[ ] C4  Experiment runner with logging
[ ] C5  Clear data/model versioning policy
```

**Estimated effort:** 2–4 weeks (can overlap with thesis writing)

---

## PHASE D — Future Research Extensions

**Only after Phases A–B (minimum) are complete.** These are explicitly out of current scope per the research specification.

| Extension | Purpose | When |
|-----------|---------|------|
| **Peak-Aware Learning** | Better handling of bursty workloads | After baseline comparison is solid |
| **Multi-resource forecasting** | mem, disk, network metrics | Module 1 extension or Module 2 overlap |
| **Adaptive retraining** | Retrain when drift detected | After offline pipeline is stable |
| **Online learning** | Update model incrementally | Research stretch goal |
| **Concept drift adaptation** | Detect and respond to drift | Pairs with anomaly modules (3/4) |
| **Additional temporal features** | Lag features, rolling stats, holidays | Incremental accuracy improvement |
| **Module 2: Short-term prediction** | Multi-resource short horizon | Separate module after Module 1 complete |
| **Module 3: Performance anomaly detection** | SLA/performance alerts | Parallel track if timeline allows |
| **Module 4: Security anomaly detection** | Security events | Parallel track if timeline allows |

**Do not start Phase D until:**

- Phase A checklist is complete
- At least one fair Hybrid vs Global GRU comparison exists (Phase B2)
- Unseen container results exist (Phase B1)

---

## Recommended Timeline

| Week | Focus |
|------|-------|
| **1** | A1 (Global GRU leakage) + A4 (scalers) + A7 (seeds) |
| **2** | A2 (input window decision) + A3 (unified eval) + A5 (dedup utils) |
| **3** | B1 (container splits) + B2 (primary experiment) |
| **4** | B3 + B4 (persist models, plots, MAPE) + start thesis results section |
| **5+** | C1–C3 (requirements, README, partial modularization) |
| **Later** | Phase D extensions as time permits |

---

## What to Do First (This Week)

If you can only do three things, do these in order:

1. **Fix Global GRU leakage (A1)** — current Global GRU results are not defensible without this.
2. **Unify evaluation (A3)** — same containers, same val period, real CPU %, MAE/RMSE/MAPE.
3. **Save scalers in preprocessing (A4)** — makes the pipeline reproducible end-to-end.

After that, lock the Hybrid input window (A2) and run the first fair comparison experiment (B2).

---

## Success Criteria

You are ready for thesis-quality Module 1 results when:

- [ ] No validation-target leakage in either architecture
- [ ] Hybrid vs Global GRU compared on identical protocol
- [ ] Metrics reported on ≥100 containers (not one demo)
- [ ] Unseen container test set evaluated separately
- [ ] Both models saved with logged hyperparameters
- [ ] Plots exported and reproducible from saved artifacts
- [ ] All methodological decisions documented (288 vs 96, weekly seasonality)

---

## Task Dependency Graph

```
A4 (scalers) ──► A1 (Global GRU fix) ──► A3 (unified eval) ──► B2 (primary experiment)
                      │                        │
A2 (input window) ────┘                        ├──► B1 (unseen containers)
A5 (dedup utils) ──────────────────────────────┘
A6 (Prophet docs) ──► B2
A7 (seeds) ──► B2

B2 ──► B3 (save Global GRU) ──► B4 (plots/MAPE)
B2 ──► C2 (modularize) ──► C4 (experiment runner)

Phase A + B complete ──► Phase D (future extensions)
```

---

## Master Progress Tracker

Copy this to track overall progress:

```
PHASE A — Fix current implementation
[ ] A1  Global GRU data leakage
[ ] A2  Hybrid input window decision
[ ] A3  Unified evaluation protocol
[ ] A4  Preprocessing artifacts
[ ] A5  Code deduplication
[ ] A6  Prophet seasonality documented
[ ] A7  Random seeds

PHASE B — Complete Module 1 research
[ ] B1  Unseen container evaluation
[ ] B2  Primary comparison experiment
[ ] B3  Persist Global GRU
[ ] B4  MAPE + exported plots
[ ] B5  Day 2 as secondary metric

PHASE C — Engineering & reproducibility
[ ] C1  requirements.txt
[ ] C2  Modular src/ package
[ ] C3  Root README.md
[ ] C4  Experiment logging
[ ] C5  Version-control policy

PHASE D — Future extensions (after A + B)
[ ] Peak-Aware Learning
[ ] Multi-resource forecasting
[ ] Adaptive / online learning
[ ] Module 2 / 3 / 4
```
