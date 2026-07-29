# Chapter 1 — Introduction

**Module:** Module 3 — System Anomaly Detector (Drift-Aware Approach)  
**Scope:** Final-report documentation for Module 3 only  
**Evidence basis:** Canonical pipeline under `module3/module3_pipeline/`; frozen metrics under `module3/models/`; integration handoff under `module3/final-model/`

---

## 1.1 Background and Motivation

Container orchestration platforms such as Docker and Kubernetes have become the dominant substrate for cloud-native microservices [1], [2]. They enable dense packing of services onto shared hosts, rapid redeployment, and fine-grained horizontal scaling. At the same time, they introduce continuous variability: traffic spikes, rolling updates, noisy-neighbour contention, and injected or emergent faults can all change how “normal” CPU and memory behaviour looks from one hour to the next [3], [4].

**Performance anomalies**—CPU saturation, memory leaks or jumps, and pod failures—degrade latency and availability if they are detected only after user-visible failure [5]. Traditional monitoring relies on static thresholds (for example, alert when CPU exceeds 80%). Such rules are brittle under concept drift: a threshold tuned for one workload regime becomes either too noisy or too silent when the deployment’s baseline shifts [6], [7].

Unsupervised deep anomaly detectors, particularly autoencoders and variational autoencoders (VAEs), learn a model of normal behaviour from benign telemetry and flag large reconstruction errors as anomalies [8], [9]. In container settings this is attractive because labelled faults are scarce at runtime and because many performance faults leave a multivariate signature across CPU and memory gauges [10].

Three practical requirements motivate Module 3:

1. **Real-time, lightweight scoring** on short windows of container KPIs without a heavy ensemble.
2. **Multi-metric correlation** so jointly abnormal CPU–memory patterns are not missed by single-metric rules.
3. **Drift awareness**—adaptive decision boundaries and optional incremental updates—so the detector remains usable when the notion of normal changes across deployments or over time [6], [11].

Module 3 addresses these needs with a PCA-whitened VAE trained only on normal windows from the AIOpsArena benchmark, reconstruction MSE as the anomaly score, a blended adaptive threshold, and KS-test-triggered fine-tuning evaluated under simulated stream replay.

**Scope note (proposal versus delivery).** The project proposal also named *network bottlenecks* as a target anomaly class. The implemented feature set uses **CPU and memory container metrics only**. Network-layer faults (`delay`, `loss` in AIOpsArena ground truth) are relabelled to normal in preprocessing and are **not delivered** as a detection capability. This limitation is intentional and documented throughout this report set.

---

## 1.2 Aim and Objectives

### Aim

To design, implement, and evaluate a lightweight, drift-aware system anomaly detector for containerized microservices that identifies performance-related faults from multivariate CPU and memory telemetry using a VAE trained exclusively on normal behaviour, reconstruction error as the primary anomaly signal, and adaptive mechanisms that avoid full retraining.

### Objectives

1. Construct a leakage-aware preprocessing pipeline from AIOpsArena container KPIs to fixed-length windows with PCA whitening.
2. Train a Variational Autoencoder exclusively on normal (`complex_case1` train) windows and score anomalies via reconstruction MSE.
3. Implement and evaluate blended adaptive thresholding based on recent and anchor reconstruction-error statistics.
4. Implement error-distribution monitoring (Kolmogorov–Smirnov test) and incremental fine-tuning triggered by detected drift.
5. Compare the VAE against classical and deep baselines (Isolation Forest, deterministic autoencoder) under in-distribution and drift evaluation sets.
6. Document honestly which proposed novelty components are fully realised, partially realised, or out of scope.

---

## 1.3 Proposed Solution Overview

### Users

| User group | Role |
|------------|------|
| Researchers / examiners | Reproduce `module3_pipeline/` notebooks and inspect `models/*.pkl` metrics |
| Integrators | Load `final-model/vae_cc1_final_model.pt` via `vae_alone_loader.py` for demo inference |
| Operators (intended) | Consume `is_anomaly` / reconstruction MSE for alerting (outside Module 3 package scope) |

Module 3 produces **anomaly scores and binary decisions**, not resource forecasts or scaling actions.

### Inputs

| Input | Description |
|-------|-------------|
| Raw telemetry | Per-container AIOpsArena KPIs at **15 s** intervals |
| Features (7) | 3 CPU rates + 4 memory gauges (exact column order fixed) |
| Model window | Shape `(30, 7)` → flatten → PCA → `(26,)` whitened vector |

### Outputs

| Output | Description |
|--------|-------------|
| `reconstruction_mse` | Continuous anomaly score (higher = more abnormal) |
| `is_anomaly` | Binary decision against a threshold |
| Research-only extras | Blended adaptive threshold trace; KS-triggered fine-tune events |

### Process (high level)

```text
Raw AIOpsArena CSVs (complex_case1 + holdout cases)
  → merge, label, deduplicate, gap-mark
  → delay/loss → normal (scope cut)
  → CC1 time-split 70/10/20; RobustScaler on cc1_train
  → sliding windows (W=30, stride=1), gap-aware
  → PCA (99% variance, whiten) → 26 dims
  → train VAE on cc1_train only
  → score with reconstruction MSE (deterministic μ)
  → static threshold (val_p99) and/or blended adaptive threshold
  → periodic KS-test vs frozen train MSE → optional fine-tune
```

### Technology

| Layer | Technology |
|-------|------------|
| Deep learning | PyTorch VAE (`26→64→32→z(32)→…→26`) |
| Classical ML | scikit-learn `PCA`, `RobustScaler`, `IsolationForest` |
| Drift / stats | SciPy `ks_2samp`; per-container error buffers |
| Persistence | `torch.save` / `joblib` / `pickle` under `models/` |
| Execution | Jupyter notebooks in `module3_pipeline/` (CPU-feasible) |

### Main features

- Benign-only VAE with reconstruction-MSE scoring
- PCA whitening for multi-metric (CPU + memory) correlation
- Sliding window of 30 timesteps (7.5 minutes)
- Blended adaptive threshold (shrinkage toward per-container train anchors)
- KS-test drift monitoring every 5,000 windows with incremental fine-tuning
- Leak-free threshold calibration on anomaly-free `cc1_val`

### System requirements

| Item | Requirement (as evidenced) |
|------|----------------------------|
| Training / eval | CPU sufficient (`torch.cuda.is_available()` False in recorded runs); ~10.8k parameters |
| Inference | Sub-millisecond per window on CPU (~0.45 ms single-window in IL notebook) |
| Data | AIOpsArena Complex Case-1 for train/val/test; Complex Case-2 for drift |
| Kernel note | Development used Python 3.9 + PyTorch 2.8.0+cpu Jupyter kernel |

### Proposed versus implemented (scope note)

| Proposal element | Final implementation |
|------------------|----------------------|
| VAE + PCA + reconstruction error | **Implemented** in `module3_pipeline/` |
| Sliding window 30–60 | Canonical **30**; 45/50/60 explored in side experiments, not the deployed default |
| Adaptive threshold (mean + α×std) | **Implemented** as blended shrinkage form with calibrated **k = 6.2** |
| Online incremental learning (“SGD”) | **Implemented** as small-LR **Adam** fine-tune (5 epochs, lr=1e-4) on KS trigger |
| Network bottleneck detection | **Not delivered** (CPU/memory features only) |
| Integration package | **VAE-alone** static threshold; adaptive/IL remain research pipeline |

---

## 1.4 Dissertation Structure

This Module 3 final-report set comprises five companion Markdown files:

| File | Title | Role |
|------|-------|------|
| `01_introduction.md` | Introduction | Background, aim, solution overview |
| `02_literature_review.md` | Literature Review | Related work and research gap |
| `03_technology_adapted.md` | Technologies and Techniques Adopted | Techniques mapped to Module 3 |
| `04_approach_analysis_and_design.md` | Approach, Analysis and Design | Method and architecture |
| `05_implementation_and_discussion.md` | Implementation, Discussion and Evaluation | Code-level detail, experiments, results |

Quantitative claims cite frozen artefacts under `module3/models/` produced by `module3_pipeline/`. Novelty claims are classified against repository evidence rather than proposal text alone.

---

## References

[1] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, vol. 1, no. 3, pp. 81–84, 2014.  
[2] B. Burns, B. Grant, D. Oppenheimer, E. Brewer, and J. Wilkes, “Borg, Omega, and Kubernetes,” *Communications of the ACM*, vol. 59, no. 5, pp. 50–57, 2016.  
[3] C. Delimitrou and C. Kozyrakis, “Quasar: Resource-efficient and QoS-aware cluster management,” *ASPLOS*, 2014.  
[4] AIOpsArena benchmark documentation (`module3/docs/AIOpsArena.pdf`).  
[5] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” *ACM Computing Surveys*, vol. 41, no. 3, 2009.  
[6] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, vol. 46, no. 4, 2014.  
[7] J. Lu et al., “Learning under concept drift: A review,” *IEEE TKDE*, vol. 31, no. 12, 2019.  
[8] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” *MLSDA*, 2014.  
[9] J. An and S. Cho, “Variational autoencoder based anomaly detection using reconstruction probability,” SNU Tech. Report, 2015.  
[10] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” *ICLR*, 2014.  
[11] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” *SDM*, 2007.
