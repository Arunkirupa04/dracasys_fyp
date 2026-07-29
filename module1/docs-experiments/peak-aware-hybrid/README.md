# Peak-Aware Hybrid Prophet + GRU — Research Documentation

**Project:** DracaSys Module 1 — Long-Term CPU Utilization Forecasting  
**Institution:** University of Moratuwa — Bachelor of Information Technology (Final Year Research Project)  
**Research contribution:** Enhancing a Hybrid Prophet + GRU model with a Peak-Aware learning mechanism for long-term cloud resource forecasting.

**Primary research documentation:** This folder is the continuous research log for the Peak-Aware Hybrid study, organized by phase. Methodology, outputs, findings, and decisions are recorded progressively and are intended for reuse in the thesis Methodology and Results chapters.

**Baseline specification:** `docs/hybrid-prophet-gru.md`  
**Frozen baseline reference:** `experiments/baseline_reference_2026-07-14/`

---

## Documentation Index

| Phase | Document | Status |
|-------|----------|--------|
| 1 | [Freeze the Baseline](phase-01-baseline-freeze.md) | **Complete** |
| 2 | [Peak Exploration](phase-02-peak-exploration.md) | **Complete** |
| 3 | [Peak-Aware Learning Design](phase-03-learning-design.md) | **Complete** |
| 4 | [Implementation](phase-04-implementation.md) | **Complete** |
| 5 | [Evaluation](phase-05-evaluation.md) | **Complete** |
| 6 | [Discussion and Comparison](phase-06-discussion.md) | **Complete** |

---

## 1. Research Overview

### 1.1 Problem

Long-term CPU utilization forecasting in containerized cloud environments must remain accurate during utilization spikes. The finalized baseline Hybrid Prophet + GRU model treats all timesteps equally during GRU residual training (uniform MSE). This may under-emphasize rare but operationally important peak periods.

### 1.2 Proposed Enhancement

Introduce **one controlled experimental variable**: a Peak-Aware learning mechanism applied during GRU residual training. All other components remain identical to the frozen baseline.

### 1.3 Research Workflow

| Phase | Name | Status |
|-------|------|--------|
| 1 | Freeze the Baseline | **Complete** |
| 2 | Peak Exploration | **Complete** |
| 3 | Peak-Aware Learning Design | **Complete** |
| 4 | Implementation | **Complete** |
| 5 | Evaluation | **Complete** |
| 6 | Discussion and Comparison | **Complete** |

### 1.4 Fair Comparison Principle

The baseline Hybrid Prophet + GRU implementation is permanent and must not be modified. Peak-Aware work uses parallel modules and separate experiment outputs only. The only intended difference between baseline and Peak-Aware models is the training objective (peak-aware weighting).

---

## Appendix A — Frozen Baseline Files (Do Not Modify)

- `utils/hybrid_config.py`
- `utils/hybrid_training.py`
- `utils/hybrid_inference.py`
- `utils/hybrid_evaluation.py`
- `utils/hybrid_artifacts.py`
- `utils/sequence_utils.py`
- `notebooks/hybrid_model.ipynb`
- `notebooks/preprocessing.ipynb`
- `models/hybrid_gru.keras`
- `models/residual_stats.pkl`
- `experiments/baseline_reference_2026-07-14/`

---

*Last updated: 2026-07-17 — Phase 6 complete; Peak-Aware Hybrid research track (Phases 1–6) locked*
