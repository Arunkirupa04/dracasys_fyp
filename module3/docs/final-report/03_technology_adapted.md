# Chapter 3 — Technologies and Techniques Adopted

**Module:** Module 3 — System Anomaly Detector (Drift-Aware Approach)  
**Purpose:** Explain each adopted technique in terms of the Module 3 problem, and map it to the actual implementation under `module3/module3_pipeline/`.

---

## 3.1 Overview

Module 3 combines representation learning, reconstruction-error scoring, and drift-aware monitoring. Technologies are selected because the problem is (i) label-scarce at training time, (ii) multivariate across CPU and memory, and (iii) subject to distribution shift between `complex_case1` and later deployments such as `complex_case2`.

Canonical implementation homes:

| Technique | Primary source |
|-----------|----------------|
| Clean / split / scale | `module3_pipeline/clean_and_split.ipynb` |
| Windowing + PCA | `module3_pipeline/windowing_pca.ipynb` |
| VAE train | `module3_pipeline/train_vae.ipynb` |
| Static eval | `module3_pipeline/vae_eval.ipynb` |
| Blended adaptive threshold | `module3_pipeline/adaptive_threshold_blended.ipynb` |
| KS + fine-tune | `module3_pipeline/incremental_learning.ipynb` |
| Baselines / ablation | `baseline_comparison.ipynb`, `final_comparison.ipynb` |
| Integration scorer | `final-model/vae_alone_loader.py` |

---

## 3.2 Variational Autoencoder (VAE)

### What it is

A VAE is a generative latent-variable model that encodes \(x\) into parameters of an approximate posterior \(q(z\mid x)=(\mu,\sigma^2)\) and decodes samples (or \(\mu\)) to a reconstruction \(\hat{x}\) [1]. Training maximises a variational lower bound combining reconstruction fidelity and a KL regulariser toward a prior (typically \(\mathcal{N}(0,I)\)).

### How it works for anomaly detection

Trained only on normal windows, the decoder approximates the normal manifold. Anomalous windows reconstruct poorly, yielding elevated MSE [2], [3]. At inference, Module 3 uses the **deterministic encoder mean** \(\mu(x)\) (no stochastic sampling) so scores are repeatable.

### Why it is suitable for Module 3

Performance faults are heterogeneous and sparse. A benign-only VAE does not require fault labels at train time and produces a continuous severity score suitable for thresholding under imbalance.

### How it is used in the implementation

Architecture (PCA input dim 26):

```text
Linear(26→64)→ReLU→Linear(64→32)→ReLU
  → fc_μ / fc_logvar (32→32; logvar clamped [-10,10])
  → decode: Linear(32→32)→ReLU→Linear(32→64)→ReLU→Linear(64→26)
```

| Hyperparameter | Value |
|----------------|------:|
| `latent_dim` | 32 |
| `beta_max` (KL weight) | 0.01 |
| KL warmup | 10 epochs linear |
| Optimizer (train) | Adam, lr \(10^{-3}\) |
| Batch size | 512 |
| Max epochs / patience | 300 / 20 |
| Parameters | 10,778 |

---

## 3.3 Principal Component Analysis (PCA)

### What it is

PCA finds orthogonal directions of maximal variance and can retain a cumulative-variance fraction of components [4]. Whitening rescales retained components to unit variance.

### How it works

Each window is flattened \((30\times7)\rightarrow 210\). PCA fit on `cc1_train` with `n_components=0.99`, `whiten=True` yields **26** dimensions.

### Why it is suitable for Module 3

CPU and memory KPIs are correlated. PCA provides a compact joint representation. Whitening prevents MSE from ignoring low-variance CPU axes dominated by high-variance memory components—verified design rationale in `windowing_pca.ipynb` (99% preferred over 95% for this reason).

### How it is used in the implementation

Order is fixed: **window → flatten → PCA → VAE**. Scaler and PCA objects are bundled into the integration checkpoint for VAE-alone serving.

---

## 3.4 Reconstruction Error as Anomaly Signal

### What it is

The discrepancy between input and reconstruction, aggregated to a scalar score [3], [5].

### How it works

```text
score = mean( (decode(μ(x)) − x)² )   over the 26 PCA dimensions
```

### Why it is suitable

It is unsupervised at inference, continuous, and comparable across windows once preprocessing is frozen.

### How it is used

Primary ranking metric in evaluation is **PR-AUC** (class imbalance: ~0.58% anomalies on `cc1_test` windows). Binary decisions compare score to a threshold (`val_p99` for VAE-alone; blended adaptive threshold in research stream).

---

## 3.5 Sliding-Window Processing

### What it is

A fixed-length contiguous segment of recent timesteps used as one model input [6].

### How it works

| Parameter | Canonical value |
|-----------|-----------------|
| `WINDOW_SIZE` | **30** |
| Wall-clock span | \(30\times15\,\mathrm{s}=7.5\,\mathrm{min}\) |
| `STRIDE` | 1 |
| Gap handling | Windows spanning `is_gap` are skipped |
| Label rule | Window anomalous if any timestep is anomalous |

### Why it is suitable

Captures short temporal context for leaks and saturation while keeping memory bounded (one window per container at inference).

### How it is used

Implemented in `windowing_pca.ipynb`. Proposal range 30–60: canonical delivery is **30**. Side experiments at 45/50/60 exist under `experiments/` and `module3_pipeline_v2/` but are not the final default.

---

## 3.6 Adaptive Thresholding (Blended Shrinkage)

### What it is

A decision boundary that updates from recent score statistics rather than remaining frozen [7].

### How it works (implemented formula)

```text
w = n_local / (n_local + PRIOR_STRENGTH)
blended_mean = w·local_mean + (1−w)·anchor_mean
blended_std  = w·local_std  + (1−w)·anchor_std
threshold    = blended_mean + k · blended_std
```

| Parameter | Value |
|-----------|------:|
| `BUFFER_SIZE` | 500 |
| `PRIOR_STRENGTH` | 500 |
| `k` (`k_adaptive`) | **6.2** (calibrated ≈1% FPR on `cc1_val`) |
| Anchor | Per-container train MSE mean/std (std capped) |

When \(n_{\mathrm{local}}=0\), the rule reduces to the train anchor (static-like behaviour).

### Why it is suitable

Drift changes the normal score distribution; a fixed `val_p99` from CC1 becomes miscalibrated on CC2. Shrinkage avoids unstable thresholds when the local buffer is small.

### How it is used

`adaptive_threshold_blended.ipynb`. Relative scores \((\mathrm{mse}-\mathrm{blended\_mean})/\mathrm{blended\_std}\) also improve PR-AUC under adaptation. A naive recent-only mean+3·std experiment (`experiments/adaptive_threshold.ipynb`) is historical and superseded.

**Proposal wording note:** “Threshold = Mean + α×Std” is realised, but with **blended** mean/std and calibrated \(k\), not a pure recent-window mean+α·std alone.

---

## 3.7 Error Distribution Monitoring (KS-Test)

### What it is

The two-sample Kolmogorov–Smirnov test compares empirical CDFs of two samples [8].

### How it works

Every `REFIT_INTERVAL=5000` windows, compare a buffer of recent likely-normal MSEs to a frozen sample of `cc1_train` MSEs from the original model. Drift is declared when \(p < \alpha\) with \(\alpha=0.001\).

### Why it is suitable

It detects distributional shift in the anomaly score space without requiring fault labels at runtime.

### How it is used

Offline characterisation in `vae_eval.ipynb` (CC2 vs train: KS statistic ≈ **0.410**, \(p\approx 0\)). Online trigger in `incremental_learning.ipynb` via `scipy.stats.ks_2samp`.

---

## 3.8 Online / Incremental Learning

### What it is

Small weight updates on recent data instead of full offline retrain [9].

### How it works

| Parameter | Value |
|-----------|------:|
| Trigger | KS \(p < 0.001\) at refit checkpoints |
| Buffer | Up to **2000** likely-normal PCA windows |
| Epochs | **5** |
| Learning rate | **\(10^{-4}\)** (10× smaller than train) |
| Optimiser | **`torch.optim.Adam`** |

### Why it is suitable

Keeps the 10.8k-parameter model adaptable with low compute. Avoids heavy ensembles.

### How it is used

`incremental_learning.ipynb` `run_stream()`. Control vs treatment on `drift_cc2` shows F1 **0.347 → 0.390**.

**Proposal wording note:** comments say “SGD”; code uses **Adam**. Documentation treats this as incremental fine-tuning with a small learning rate, not as a mismatch that voids the mechanism.

---

## 3.9 Container System Performance Metrics

### What they are

cAdvisor/Prometheus-style container counters and gauges [10].

### How they are used

| # | Feature | Role |
|---|---------|------|
| 1–3 | CPU usage/system/user **rates** | Saturation / burn |
| 4–7 | Memory usage, WSS, RSS, cache | Leaks / pressure |

Rates are derived from cumulative CPU counters before windowing. Istio/network KPIs and logs/traces in AIOpsArena are **not used**.

### Why this set

Matches detectable ground-truth classes `{cpu, memory, pod-failure}` after delay/loss rescoping.

---

## 3.10 Classical Baselines (for evaluation)

| Baseline | Role in Module 3 |
|----------|------------------|
| Deterministic AE | Same PCA inputs; non-variational recon baseline |
| Isolation Forest | Classical unsupervised baseline on window features |

Trained/evaluated in `baseline_comparison.ipynb` / rolled up in `final_comparison.ipynb`.

---

## 3.11 Technology-to-Pipeline Mapping

| Pipeline stage | Technologies |
|----------------|--------------|
| Preprocess | Dedup, gap flags, RobustScaler, delay/loss→normal |
| Representation | Sliding window, flatten, PCA+whiten |
| Model | VAE + recon MSE |
| Decide (static) | `val_p99` from anomaly-free val |
| Decide (adaptive) | Blended mean + \(k\)·std |
| Adapt | KS monitor + Adam fine-tune |

---

## References

[1] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” *ICLR*, 2014.  
[2] J. An and S. Cho, “Variational autoencoder based anomaly detection using reconstruction probability,” SNU Tech. Report, 2015.  
[3] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders…,” *MLSDA*, 2014.  
[4] I. T. Jolliffe, *Principal Component Analysis*, Springer, 2002.  
[5] V. Chandola et al., “Anomaly detection: A survey,” *ACM Computing Surveys*, 2009.  
[6] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” *SDM*, 2007.  
[7] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, 2014.  
[8] F. J. Massey Jr., “The Kolmogorov-Smirnov test for goodness of fit,” *JASA*, 1951.  
[9] J. Lu et al., “Learning under concept drift: A review,” *IEEE TKDE*, 2019.  
[10] Prometheus / cAdvisor container metric conventions; AIOpsArena dataset schema.
