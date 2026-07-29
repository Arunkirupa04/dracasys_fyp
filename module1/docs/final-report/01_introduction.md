# Chapter 1 — Introduction

**Module:** Module 1 — Long-Term Resource Forecasting (Hybrid Prophet + GRU)  
**Scope:** Final-report documentation for Module 1 only  
**Evidence basis:** Production service under `module1/app/` and `module1/artifacts/hybrid_v1/`; research metrics under `module1/experiments/`

---

## 1.1 Background and Motivation

Container orchestration platforms such as Docker and Kubernetes host dense, multi-tenant workloads whose resource demand varies with traffic, batch jobs, and application behaviour [1], [2]. Short-horizon reactive scaling (for example, act when CPU exceeds a fixed threshold) can already be too late for capacity planning: new nodes, reservation changes, and budget decisions need **hours-ahead** visibility into utilisation [3], [4].

**Long-term resource forecasting** aims to predict utilisation patterns that include gradual growth, daily seasonality, and recurring workload cycles [5], [6]. Classical statistical tools such as Prophet decompose time series into trend and seasonal components and are strong when structure is approximately additive [7]. Deep sequence models (LSTM/GRU families) can capture nonlinear residuals that purely seasonal models miss [8], [9].

In practice, container CPU traces are noisy, often near-idle for long stretches, and may contain short gaps after resampling. A practical long-term forecaster for containers therefore needs:

1. A stable **seasonal baseline** for capacity planning.
2. A **nonlinear residual corrector** for structure Prophet cannot explain.
3. A clear production interface so other system modules can request forecasts without embedding training code.

Module 1 addresses these needs with a **hybrid Prophet + GRU** model that forecasts **24 hours** of container **CPU utilisation** at **15-minute** resolution, exposed as a frozen REST service (`hybrid_v1`).

**Scope notes (proposal versus delivery).**

| Proposal wording | Delivered implementation |
|------------------|--------------------------|
| Prophet + **LSTM** | Prophet + **GRU** (no LSTM in the Module 1 codebase) |
| Peak-aware learning as a core novelty | Evaluated in research; **not** promoted into production `hybrid_v1` (peak MAE did not improve) |
| Memory / multi-metric forecasting | **CPU only** |
| Online drift / retrain in the API | **Not implemented** (stateless frozen artifacts) |

---

## 1.2 Aim and Objectives

### Aim

To design, implement, and evaluate a hybrid long-term CPU forecasting system for containerized environments that combines Prophet’s trend/seasonality modelling with a GRU residual learner, and to expose a production-ready inference API for 24-hour capacity-oriented forecasts.

### Objectives

1. Build a leakage-aware preprocessing pipeline from Alibaba cluster container CPU traces to 15-minute series suitable for Prophet and residual learning.
2. Implement an additive hybrid forecaster: Prophet baseline + GRU residual prediction over a 96-step (24 h) horizon.
3. Compare the hybrid model against a Global GRU baseline (and related diagnostics) using MAE, RMSE, and MAPE on a multi-container Day-1 evaluation cohort.
4. Investigate peak-aware training as a research variant and report whether peak-period error improves.
5. Provide missing-data handling in the research/production data pipeline via regular resampling and linear interpolation.
6. Package a frozen `hybrid_v1` artifact bundle and FastAPI service for integration with the wider DracaSys system.
7. Document honestly which proposed elements (including LSTM wording and peak-aware production claims) match the as-built system.

---

## 1.3 Proposed Solution Overview

### Users

| User group | Role |
|------------|------|
| Researchers / examiners | Reproduce experiments under `module1/experiments/` and inspect evaluation CSVs |
| Integrators | Call `POST /forecast` on the Module 1 FastAPI sidecar |
| Downstream planners (intended) | Use 96-step CPU forecasts for capacity / proactive scaling logic (outside Module 1) |

Module 1 produces **continuous CPU % forecasts**, not anomaly alerts.

### Inputs

| Input | Description |
|-------|-------------|
| Target series | Container CPU utilisation (%) |
| Sampling | Fixed **15-minute** intervals |
| API history | Minimum **200** steps (~50 h); recommended **672** (~7 days) |
| Research tensors | 96-step residual window → 96-step residual forecast |

### Outputs

| Output | Description |
|--------|-------------|
| `predicted_cpu_percent` | Hybrid forecast, 96 steps (24 h) |
| `prophet_component_percent` | Prophet contribution |
| `gru_residual_component_percent` | GRU residual contribution |

### Process (high level)

```text
Alibaba container_usage.csv
  → filter containers → 15-min resample → linear interpolate
  → per-container MinMax scale → chronological train/val split
  → fit Prophet (daily seasonality) → residual = cpu_scaled − prophet
  → train GRU on residual windows (96 → 96)
  → at inference: refit Prophet on request history + frozen GRU
  → final = inverse_scale(prophet_future + residual_forecast)
```

### Technology

| Layer | Technology |
|-------|------------|
| Seasonal baseline | Facebook Prophet (`daily_seasonality=True`, weekly off) |
| Residual learner | TensorFlow/Keras **GRU** stack |
| API | FastAPI (port 8000) |
| Artifacts | `model.keras`, `scalers.pkl`, `residual_stats.pkl`, config JSON |

### Main features

- Additive hybrid Prophet + GRU residual forecasting
- 24-hour (96-step) CPU forecast at 15-minute resolution
- Per-request Prophet fit; frozen GRU weights
- Frozen per-container MinMax scalers (435 known IDs)
- Research evaluation of peak-aware and residual-learning variants
- Pipeline resampling + linear interpolation for irregular/missing ticks

### System requirements

| Item | Requirement (as evidenced) |
|------|----------------------------|
| Runtime | Python + TensorFlow + Prophet; CPU inference supported |
| Latency | Typically **5–30+ s** per request (Prophet refit dominates) |
| Artifacts | Read-only `artifacts/hybrid_v1/` bundle |
| Data | Alibaba-style CPU % series at 15-minute spacing |

### Proposed versus implemented (summary)

| Element | Status |
|---------|--------|
| Hybrid statistical + deep forecasting | ✅ Delivered (Prophet + **GRU**) |
| LSTM residual network | ❌ **Not used** — GRU replaces proposal LSTM |
| Peak-aware production model | ⚠️ Researched; not in `hybrid_v1` |
| Missing-data handling in pipeline | ✅ Resample + linear interpolate |
| Online retrain / drift loop | ❌ Not in API |

---

## 1.4 Dissertation Structure

This Module 1 final-report set comprises five companion Markdown files:

| File | Title | Role |
|------|-------|------|
| `01_introduction.md` | Introduction | Background, aim, solution overview |
| `02_literature_review.md` | Literature Review | Related work and research gap |
| `03_technology_adapted.md` | Technologies and Techniques Adopted | Techniques mapped to Module 1 |
| `04_approach_analysis_and_design.md` | Approach, Analysis and Design | Method and architecture |
| `05_implementation_and_discussion.md` | Implementation, Discussion and Evaluation | Code-level detail, experiments, results |

Quantitative claims cite frozen evaluation under `module1/experiments/` (especially `baseline_reference_2026-07-14/`). The production model identity is **`hybrid_v1` / `hybrid_prophet_gru`**.

---

## References

[1] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, vol. 1, no. 3, 2014.  
[2] B. Burns et al., “Borg, Omega, and Kubernetes,” *Communications of the ACM*, vol. 59, no. 5, 2016.  
[3] C. Delimitrou and C. Kozyrakis, “Quasar: Resource-efficient and QoS-aware cluster management,” *ASPLOS*, 2014.  
[4] J. Duggan et al., “Predicting the performance of virtual machine migration,” *IEEE MASCOTS*, 2015.  
[5] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, OTexts.  
[6] G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*.  
[7] S. J. Taylor and B. Letham, “Forecasting at scale,” *The American Statistician*, vol. 72, no. 1, 2018.  
[8] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, 1997.  
[9] K. Cho et al., “Learning phrase representations using RNN encoder–decoder…,” *EMNLP*, 2014.
