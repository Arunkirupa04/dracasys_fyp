# Global GRU Baseline — Research Documentation

**Project:** DracaSys Module 1 — Long-Term CPU Utilization Forecasting  
**Institution:** University of Moratuwa — Bachelor of Information Technology (Final Year Research Project)  
**Research contribution:** Establishing a rigorous, reproducible **Global GRU baseline** for fair comparison against the frozen Hybrid Prophet + GRU baseline and future Peak-Aware extensions.

**Primary research documentation:** This folder is the continuous research log for the Global GRU baseline study, organized by phase. Methodology, outputs, findings, and decisions are recorded progressively and are intended for reuse in the thesis Methodology and Results chapters.

**Hybrid control specification:** `docs/hybrid-prophet-gru.md`  
**Frozen Hybrid control reference:** `experiments/baseline_reference_2026-07-14/`  
**Baseline specification (research contract):** [Phase 0.5 — Global GRU Baseline Specification](phase-03-5-baseline-specification.md)  
**Methodology gap register:** `docs/methodology-gap-analysis.md`  
**Target evaluation protocol:** `docs/recommended-evaluation-protocol.md`

---

## Documentation Index

| Phase | Document | Status |
|-------|----------|--------|
| 1 | [Research Planning & Scope](phase-01-research-planning.md) | **Complete** (8/8 tasks) |
| 2 | [Methodology Audit & Fair Comparison](phase-02-methodology-audit.md) | **Complete** |
| 3 | [Pipeline & Architecture Design](phase-03-pipeline-design.md) | **Complete** |
| **0.5** | [**Global GRU Baseline Specification**](phase-03-5-baseline-specification.md) | **Complete** |
| 4 | [Implementation](phase-04-implementation.md) | **Complete** (9/9 tasks) |
| 5 | [Evaluation & Verification](phase-05-evaluation.md) | **Complete** (7/7 tasks) |
| 6 | [Baseline Freeze & Methodology Comparison](phase-06-baseline-freeze-and-comparison.md) | **Complete** (7/7 tasks) |

---

## 1. Research Overview

### 1.1 Problem

Module 1 investigates two complementary **forecasting methodologies**. The **Hybrid Prophet + GRU** baseline is complete, verified, and frozen (`experiments/baseline_reference_2026-07-14/`). The **Global GRU** methodology — a single shared GRU trained across many containers for direct CPU forecasting — has been implemented, verified, evaluated, and compared against Hybrid under the shared 99-container Day-1 protocol.

**Official Global GRU baseline (2026-07-17):** `experiments/global_gru_baseline_2026-07-17_121748/` — EarlyStopping patience=10 (corrected from an implementation oversight in the superseded reference).  
**Superseded historical reference:** `experiments/global_gru_reference_2026-07-17/` (patience=3 — preserved, not deleted).  
**Methodology comparison:** `experiments/hybrid_vs_global_2026-07-17_095147/` (regenerated against corrected baseline).

This study track is **closed**. Future Peak-Aware Global GRU work proceeds as separate experiments against the official Global GRU control.

### 1.2 Proposed Baseline

Implement **Global GRU v1** as a modular, reproducible pipeline that:

- Reuses frozen preprocessing artifacts without modification
- Trains on train-period sequences only (no validation-target leakage)
- Evaluates on the preprocessing validation period (temporal holdout, Day 1 = 96 steps)
- Reports MAE, RMSE, MAPE in real CPU % on the same 99-container cohort as Hybrid
- Saves timestamped experiment artifacts and a permanent frozen reference

**No Peak-Aware Learning** is introduced in this study track.

### 1.3 Research Workflow

| Phase | Name | Status |
|-------|------|--------|
| 1 | Research Planning & Scope | **Complete** (8 tasks — see [Phase 1 log](phase-01-research-planning.md)) |
| 2 | Methodology Audit & Fair Comparison | **Complete** |
| 3 | Pipeline & Architecture Design | **Complete** |
| **0.5** | **Global GRU Baseline Specification** | **Complete** |
| 4 | Implementation | **Complete** (9/9 tasks) |
| 5 | Evaluation & Verification | **Complete** (7/7 tasks) |
| 6 | Baseline Freeze & Methodology Comparison | **Complete** (7/7 tasks) |

### 1.3.1 Phase 1 Task Summary (Complete)

| Task | Focus | Status |
|------|-------|--------|
| 1 | Review documentation, skills, and Hybrid control | **Complete** |
| 2 | Audit Global GRU notebook prototype | **Complete** |
| 3 | Document comparison philosophy | **Complete** |
| 4 | Define objectives, questions, success criteria | **Complete** |
| 5 | Define non-goals and implementation philosophy | **Complete** |
| 6 | Design phased research workflow | **Complete** |
| 7 | Create research documentation folder | **Complete** |
| 8 | Lock Phase 1 research planning record | **Complete** |

Full task methodology, findings, and outputs: [Phase 1 — Research Planning & Scope](phase-01-research-planning.md).

### 1.4 Comparison Philosophy

This study compares **forecasting methodologies** under an identical experimental protocol — not neural network architectures in isolation.

The following remain **fixed** across both methodologies:

- Preprocessing pipeline and frozen `data/` artifacts
- Temporal train/validation split
- Evaluation procedure (Day 1 temporal holdout, 99-container cohort)
- Metrics (MAE, RMSE, MAPE in real CPU %) and aggregation
- Reporting format

The **forecasting methodology** differs by design:

| Methodology | Hybrid Prophet + GRU (control) | Global GRU (baseline) |
|-------------|-------------------------------|----------------------|
| Approach | Prophet decomposition + GRU residual correction | Single shared GRU, direct CPU forecast |
| Inference | Per-container Prophet + shared GRU | Shared GRU only |

Internal model differences (layer sizes, features, training hyperparameters) are consequences of each methodology, not independent experimental variables.

### 1.5 Implementation Philosophy

During the Global GRU baseline experiment:

- **Peak-Aware Learning** is completely out of scope.
- **No model-structure optimisation** should be performed.
- **No hyperparameter tuning** should be performed.
- **No additional features** should be introduced.
- **No preprocessing changes** should be made.

The objective is a scientifically valid, reproducible baseline — not maximised accuracy. Any future enhancement (Peak-Aware Learning, architecture changes, feature engineering, hyperparameter optimisation) is a **separate experiment** after the baseline is frozen.

See [Phase 0.5 — Baseline Specification](phase-03-5-baseline-specification.md) for the complete locked configuration.

### 1.6 Code Reuse Philosophy

Implementation follows a **reuse-first** approach:

- **Reuse** shared utilities wherever logic is common (sequence builders, metrics, container pre-filtering, forecast constants).
- **Do not duplicate** Hybrid modules or copy their structure unnecessarily.
- **Implement new modules only** for Global GRU–specific behaviour (direct CPU training, direct CPU inference, Global GRU artifact paths).

The goal is a maintainable research codebase, not a mirror of the Hybrid implementation.

### 1.7 Explicit Non-Goals (This Study Track)

| Item | Deferred to |
|------|-------------|
| Peak-Aware Global GRU | After Global GRU baseline freeze |
| Preprocessing changes | Not permitted during baseline |
| Hyperparameter tuning / model optimisation | Separate experiments only |
| Unseen container holdout | Post-freeze follow-on experiment |
| Feature or layer ablations | Separate experiments only |
| Day 2 recursive forecast (Global GRU) | Supplementary, not primary |

---

## 2. Hybrid Control Reference (Read-Only)

| Metric | Mean | Std | Cohort |
|--------|------|-----|--------|
| MAE | 1.745882 | 2.486804 | 99 containers |
| RMSE | 2.387813 | 3.219141 | 99 containers |
| MAPE | 111.940318 | 615.299100 | 99 containers |

- **Skipped:** `c_14674` (missing from frozen train/val data)
- **Source:** `experiments/baseline_reference_2026-07-14/config/baseline_metadata.json`

---

## 3. Forecasting Methodologies — At a Glance

| Dimension | Hybrid Prophet + GRU (frozen) | Global GRU v1 (frozen) |
|-----------|-------------------------------|------------------------|
| Forecasting methodology | Prophet trend/seasonality + GRU residuals | Direct `cpu_scaled` forecasting |
| Per-container component | Prophet refit at inference | None — shared model only |
| Model input (GRU) | `(96, 1)` normalized residuals | `(96, 3)` CPU + train stats |
| Model output (GRU) | 96 residual steps | 96 CPU steps (scaled) |
| Final prediction | `prophet + gru_residual` → inverse MinMax | `gru_prediction` → inverse MinMax |
| Code — shared | `sequence_utils.py`, metric functions (read-only) | Same shared utilities |
| Code — methodology-specific | `utils/hybrid_*.py` (frozen) | `utils/global_*.py` (frozen) |
| Frozen reference | `experiments/baseline_reference_2026-07-14/` | `experiments/global_gru_baseline_2026-07-17_121748/` (supersedes `global_gru_reference_2026-07-17/`) |
| Methodology comparison | — | `experiments/hybrid_vs_global_2026-07-17_095147/` |

---

## 4. Global GRU Baseline Reference (Official)

| Metric | Mean | Std | Cohort |
|--------|------|-----|--------|
| MAE | 1.924241 | 2.315546 | 99 containers |
| RMSE | 2.610536 | 2.965272 | 99 containers |
| MAPE | 116.502574 | 642.923537 | 99 containers |

- **Skipped:** `c_14674` (missing from frozen train/val data)
- **Official reference:** `experiments/global_gru_baseline_2026-07-17_121748/`
- **Superseded reference (historical, patience=3):** `experiments/global_gru_reference_2026-07-17/`
- **Source:** `experiments/global_gru_baseline_2026-07-17_121748/config/baseline_metadata.json`

---

## 5. Methodology Comparison (Phase 6 — Complete, regenerated 2026-07-17)

**Experiment:** `experiments/hybrid_vs_global_2026-07-17_095147/`  
**Executive summary:** `FINAL_COMPARISON_SUMMARY.md` (in comparison experiment directory)

| Methodology | MAE (mean) | RMSE (mean) | MAPE (mean) |
|-------------|------------|-------------|-------------|
| Hybrid Prophet + GRU | 1.7459 | 2.3878 | 111.9403 |
| Global GRU v1 | 1.9242 | 2.6105 | 116.5026 |
| **Δ (Global − Hybrid)** | **+0.1784** | **+0.2227** | **+4.5623** |

**Per-container MAE:** Global better on 32 containers; Hybrid better on 67.

**Verification:** `scripts/verify_hybrid_vs_global_comparison.py` — **PASS** (regenerated vs corrected Global baseline)

**Scope:** Configuration-specific conclusions only. See [Phase 6 documentation](phase-06-baseline-freeze-and-comparison.md).

**Study track status:** **Closed** — Global GRU baseline research complete. Peak-Aware Global GRU complete (Stages 1–4).

---

## Appendix A — Frozen Files (Do Not Modify)

**Hybrid baseline (permanent control):**

- `utils/hybrid_config.py`
- `utils/hybrid_training.py`
- `utils/hybrid_inference.py`
- `utils/hybrid_evaluation.py`
- `utils/hybrid_artifacts.py`
- `notebooks/hybrid_model.ipynb`
- `notebooks/preprocessing.ipynb`
- `models/hybrid_gru.keras`
- `models/residual_stats.pkl`
- `experiments/baseline_reference_2026-07-14/`

**Global GRU baseline modules (frozen):**

- `utils/global_config.py`
- `utils/global_training.py`
- `utils/global_inference.py`
- `utils/global_evaluation.py`
- `utils/global_artifacts.py`
- `utils/global_plotting.py`
- `utils/methodology_comparison.py`
- `utils/global_comparison_plotting.py`
- `notebooks/global_gru_model.ipynb`
- `models/global_gru.keras`
- `models/global_gru_metadata.json`
- `experiments/global_gru_baseline_2026-07-17_121748/` (official Global GRU baseline)
- `experiments/global_gru_reference_2026-07-17/` (superseded historical reference)
- `experiments/hybrid_vs_global_2026-07-17_095147/` (methodology comparison — regenerated with corrected baseline)

**Shared preprocessing (frozen for all methodologies):**

- `data/train_df.parquet`
- `data/val_df.parquet`
- `data/scalers.pkl`
- `data/selected_containers.npy`
- `notebooks/preprocessing.ipynb`

**Peak-Aware modules (out of scope for Global GRU baseline):**

- `utils/peak_*.py`
- `utils/hybrid_training_peak_aware.py`

---

## Appendix C — Baseline Correction (Phase B, 2026-07-17)

The original Global GRU reference (`experiments/global_gru_reference_2026-07-17/`) used `EarlyStopping(patience=3)`, an implementation oversight. The intended configuration is `patience=10`.

| Item | Path / value |
|------|----------------|
| **Official baseline** | `experiments/global_gru_baseline_2026-07-17_121748/` |
| **Config constant** | `utils.global_config.PRIMARY_GLOBAL_BASELINE_DIR` |
| **Promotion record** | `official_baseline_promotion.json` |
| **Superseded reference** | `experiments/global_gru_reference_2026-07-17/` (preserved) |

Downstream comparisons (Hybrid vs Global, Peak-Aware Global vs Global) were regenerated against the corrected baseline without retraining non-Global models.

---

## Appendix B — Related Documentation

| Document | Purpose |
|----------|---------|
| [Hybrid Prophet + GRU](../hybrid-prophet-gru.md) | Hybrid methodology specification |
| [Peak-Aware Hybrid](../peak-aware-hybrid/README.md) | Separate Hybrid enhancement study |
| [Data Pipeline Reference](../data-pipeline-reference.md) | Stage-by-stage preprocessing flow |
| [Implementation Roadmap](../implementation-roadmap.md) | Project-wide prioritized action plan |
| [Methodology Gap Analysis](../methodology-gap-analysis.md) | Known Global GRU defects |

---

*Last updated: 2026-07-17 — Phases 1–6 complete (7/7 tasks in Phase 6); Global GRU baseline study track closed*
