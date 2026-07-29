# Chapter 1 — Introduction

**Module:** Module 2 — Drift-Aware Short-Term Resource Prediction in Containerized Environments  
**Scope:** Final-report documentation for Module 2 only  
**Evidence basis:** Canonical freeze under `module2/final_notebook/`

---

## 1.1 Background and Motivation

Container orchestration platforms pack many services onto shared hosts. Resource demand—especially CPU and memory—varies with traffic, batch jobs, and application behaviour [1], [2]. Reactive autoscaling that waits until utilisation crosses a threshold can already be too late: queues grow, latency rises, and neighbouring containers suffer noisy-neighbour effects [3].

**Short-term resource forecasting** aims to predict near-future demand so that scaling or placement decisions can be made before saturation [4], [5]. Classical statistical smoothers (e.g. exponential smoothing) are strong on stable series but struggle when the data-generating process shifts. Deep sequence models such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks can capture nonlinear temporal structure in multivariate telemetry [6], [7].

Two further challenges matter in production-like settings:

1. **Concept drift.** Workload regimes change; a model trained offline may see rising forecast error unless monitoring and adaptation are present [8], [9].
2. **Workload bursts.** Sudden spikes inflate short-horizon error and can invalidate assumptions tuned on calm periods [10].

Module 2 addresses short-horizon forecasting of container CPU and memory metrics using a residual GRU, combined with variability-aware window lengths, synthetic burst evaluation, and a streaming drift-aware adaptation loop.

---

## 1.2 Aim and Objectives

### Aim

To design, implement, and evaluate a GRU-based short-term forecaster for container resource usage that supports drift-aware monitoring and burst-stress evaluation, enabling proactive insight into near-future CPU and memory demand.

### Objectives

1. Build a leakage-aware preprocessing pipeline from raw container KPI CSVs to fixed-feature, variable-length windows.
2. Train a residual/persistence-anchored GRU to forecast four targets at horizons of 15 s, 30 s, and 45 s.
3. Compare the GRU against Persistence and Simple Exponential Smoothing (SES) baselines under clean and burst-injected test conditions.
4. Implement error monitoring (EWMA + z-score) and error-triggered incremental fine-tuning in a simulated streaming evaluation.
5. Evaluate spike versus normal regimes to quantify burst-related forecasting difficulty.
6. Document honestly which proposed novelty components are fully realised, partial, or not implemented.

---

## 1.3 Proposed Solution Overview

### Users

| User group | Role |
|------------|------|
| Researchers / examiners | Reproduce Phase 1–4 notebooks and inspect metrics |
| Integrators | Load `production_model.pt` / checkpoints for demo forecasting |
| Downstream consumers (intended) | Use numeric forecasts for scaling logic (outside Module 2) |

Module 2 produces **continuous forecasts**, not anomaly alerts or scaling actions.

### Inputs

| Input | Description |
|-------|-------------|
| Raw telemetry | Per-container, 15 s interval KPIs (Prometheus/cAdvisor-style) |
| Required raw metrics (7) | CPU usage/system/user totals; memory usage, working set, RSS, cache |
| Model tensor | Shape `(B, L, 27)`, `L ∈ [500, 1000]`, normalized + engineered features |

### Outputs

| Output | Description |
|--------|-------------|
| Forecast vector | 4 values: `cpu_usage`, `mem_usage`, `mem_working_set`, `mem_rss` |
| Horizon | One checkpoint per horizon (H1=15 s, H2=30 s, H3=45 s) |
| Units | After de-normalization: CPU-seconds (cumulative), memory in bytes |
| Stream artefacts | Per-chunk errors, drift triggers, adaptive vs static MAPE |

### Process (high level)

```text
Raw AIOpsArena CSVs (complex_case1)
  → pivot to wide metrics
  → chronological 70/15/15 split + embargo
  → optional synthetic burst injection
  → train-only z-score + lag/rolling features (27 cols)
  → adaptive-length windows [500, 1000]
  → train residual AdaptiveGRUModel per horizon
  → static eval (clean / injected / spike / normal)
  → streaming DriftMonitor → OnlineAdapter loop
```

### Technology

| Layer | Technology |
|-------|------------|
| Framework | PyTorch (`AdaptiveGRUModel`) |
| Data | pandas, NumPy (memmap `.npy`) |
| Platform | Kaggle notebooks (GPU for Phase 3) |
| Metrics | MAE, RMSE, MAPE (de-normalized) |
| Shared code | `model_defs.py` (Phase 2 export) |

### Main features

- Residual GRU anchored at last observation (starts at persistence)
- Variable-length packed sequences (500–1000)
- Multivariate 4-target forecasting (CPU + memory)
- Synthetic burst injection and spike/normal evaluation
- EWMA/z-score drift monitoring with incremental fine-tuning
- Persistence and SES baselines

### System requirements

| Item | Requirement (as evidenced) |
|------|----------------------------|
| Training | GPU recommended (Phase 3 ~40–70 min for 3 horizons) |
| Inference | Single forward pass on `(1, L, 27)`; CPU feasible |
| Data | AIOpsArena-style `complex_case1` container KPIs |
| Dependencies | No `requirements.txt` in `module2/`; PyTorch version recorded in notebook output (e.g. 2.10.0+cu128) |

### Proposed versus implemented (scope note)

| Proposal element | Final implementation |
|------------------|----------------------|
| Forecast targets | **CPU + memory** (4 targets) |
| Adaptive sliding window | Variable lookback in [500, 1000] from train variability (Phase 1); used by all later phases |
| Online learning + error monitoring | **Fully implemented** in Phase 4; stream validation shows MAPE improvement vs static |
| Drift-aware forecasting pipeline | Adaptive windows + EWMA/z drift monitor + error-triggered fine-tune (+ logged confidence band) |
| Burst-aware forecasting | **Implemented** via injection + regime eval (+ modest adaptive gains) |

---

## 1.4 Dissertation Structure

This Module 2 final-report set comprises five companion Markdown files:

| File | Title | Role |
|------|-------|------|
| `01_introduction.md` | Introduction | Background, aim, solution overview |
| `02_literature_review.md` | Literature Review | Related work and research gap |
| `03_technology_adapted.md` | Technologies and Techniques Adopted | Techniques mapped to Module 2 |
| `04_approach_analysis_and_design.md` | Approach, Analysis and Design | Method and architecture |
| `05_implementation_and_discussion.md` | Implementation, Discussion and Evaluation | Code-level detail, experiments, results |

Quantitative claims cite frozen artefacts under `module2/final_notebook/output-metrics/`. Novelty claims are classified against repository evidence rather than proposal text alone.

---

## References

[1] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, vol. 1, no. 3, pp. 81–84, 2014.  
[2] B. Burns, B. Grant, D. Oppenheimer, E. Brewer, and J. Wilkes, “Borg, Omega, and Kubernetes,” *Communications of the ACM*, vol. 59, no. 5, pp. 50–57, 2016.  
[3] C. Delimitrou and C. Kozyrakis, “Quasar: Resource-efficient and QoS-aware cluster management,” *ASPLOS*, 2014.  
[4] J. Duggan et al., “Predicting the performance of virtual machine migration,” *IEEE MASCOTS*, 2015.  
[5] Y. Xie et al., “Real-time prediction of Docker container resource usage based on time series,” applied cloud/container forecasting literature.  
[6] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.  
[7] K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” *EMNLP*, 2014.  
[8] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, vol. 46, no. 4, 2014.  
[9] J. Lu et al., “Learning under concept drift: A review,” *IEEE TKDE*, vol. 31, no. 12, 2019.  
[10] A. Ali-Eldin et al., “How will your workload look like in 6 years? Analyzing Wikimedia’s workload,” *ICPE*, 2014.
