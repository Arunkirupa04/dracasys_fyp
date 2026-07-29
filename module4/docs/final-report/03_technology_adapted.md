# Chapter 3 — Technologies and Techniques Adopted

**Module:** Module 4 — MDC vNext  
**Purpose:** Explain each adopted technique in terms of the Module 4 problem, and map it to the actual implementation under `module4/notebook/final/`.

---

## 3.1 Overview

Module 4 combines representation learning, reconstruction-error scoring, and drift-aware monitoring. The technologies below are not a generic toolkit dump: each is selected because the problem is (i) label-scarce at training time, (ii) sequential in the time-bucketed flow domain, and (iii) subject to distribution shift between training benign behaviour and later holdout traffic.

Canonical implementation homes:

| Technique | Primary source |
|-----------|----------------|
| Preprocess / windows | `mdc_preprocess_vNext_mc_kaggle.ipynb` |
| Transformer AE + scoring + PSI/KS report | `mdc_model_vNext_kaggle.ipynb` |
| Adaptive thr / stream / fine-tune | `mdc_drift_aware_kaggle.ipynb` |
| Dense AE / packaging of IF | `mdc_baselines_kaggle.ipynb` |
| Standalone model class (integration copy) | `integration/models/module4/sequence_bottleneck_ae.py` |

---

## 3.2 Autoencoders

### What it is

An autoencoder is a neural network trained to reconstruct its input through a constrained latent representation [1], [2]. The encoder maps \(x \mapsto z\); the decoder maps \(z \mapsto \hat{x}\). Training minimises a reconstruction loss such as MSE \(\|x - \hat{x}\|^2\).

### How it works for anomaly detection

When trained on benign samples only, the network approximates the normal data manifold. Anomalous inputs typically yield larger reconstruction error and can be ranked or thresholded as alerts [2], [3].

### Why it is suitable for Module 4

Container misuse in the MDC setting includes CVE exploits and reconnaissance that were not required to be enumerated at training time. An AE enables unsupervised detection without attack labels during learning.

### How it is used in the implementation

`SequenceBottleneckAE` is trained on benign `X_train` only (shape `(16795, 10, 163)` in the final freeze). The training objective is MSE reconstruction, optionally with denoising noise on the input and a contractive penalty on the bottleneck (Rifai-style finite-difference contractive term) [4]. Attack labels are used only for validation/test evaluation and threshold calibration—not for the supervised cross-entropy training of a multi-class IDS.

---

## 3.3 Attention Mechanism (Transformer Encoder–Decoder)

### What it is

Multi-head self-attention computes content-based interactions among timesteps, as introduced in the Transformer architecture [5]. Encoder layers contextualise the input sequence; decoder layers generate outputs conditioned on encoded memory.

### How it works

For an input window of \(T\) timesteps, attention weights allow each position to aggregate information from other positions without recurrence. Module 4 uses PyTorch `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` (GELU, pre-norm, `batch_first=True`).

### Why it is suitable for Module 4

Each window spans 10 buckets × 15 s ≈ 150 s of traffic. Misuse indicators may be concentrated in a subset of buckets (bursts, scans). Attention provides a flexible temporal inductive bias compared with flattening the window or using a shallow dense AE.

### How it is used in the implementation

In `SequenceBottleneckAE`:

1. Linear projection of 163 features → `d_model`
2. Sinusoidal positional encoding
3. Transformer encoder → per-timestep codes
4. Linear bottleneck (`bottleneck_dim`, e.g. 16 for HPO best)
5. Upsample + Transformer decoder with learned position queries as targets
6. Linear projection back to 163 features

This is the concrete realisation of the proposed “attention-enhanced autoencoder.” Attention is embedded in the Transformer layers rather than implemented as a separate custom module.

---

## 3.4 Reconstruction Error as Anomaly Signal

### What it is

The discrepancy between input window \(x\) and reconstruction \(\hat{x}\), aggregated over time and features into a scalar score [2], [3].

### How it works

Module 4 computes a weighted combination of mean and max temporal reconstruction errors, with optional per-feature standard-deviation weighting. A score-orientation protocol may negate raw scores when validation AUC improves after inversion (`auto_score_flip`), ensuring higher scores mean more anomalous in alert space.

### Why it is suitable for Module 4

Reconstruction error requires no attack taxonomy at inference time. Attack/misuse windows that leave the benign manifold tend to inflate error, which is the detector’s primary ranking signal.

### How it is used in the implementation

Functions such as `compute_scores` / `score_with_protocol` in `mdc_model_vNext_kaggle.ipynb` produce `s_val` / `s_test`. Offline metrics in `metrics_vnext.json` report ROC-AUC, PR-AUC, F1, MCC, precision, recall, and FPR at several threshold rules. Primary rule: **`f1_optimal`** on validation scores.

---

## 3.5 Sliding-Window Sequence Representation

### What it is

A sliding window converts a continuous time series of bucketed features into fixed-shape tensors for batch training [6].

### How it works

After 15 s bucketing (`mean_max_std` + flow count), windows of length `WINDOW_SIZE=10` are extracted with `STRIDE=2`. A window is labelled attack if the fraction of attack buckets meets `ATTACK_FRAC_THRESHOLD=0.5`.

### Why it is suitable for Module 4

Raw flows are irregular; bucketing + windows yield a uniform `(10, 163)` tensor compatible with the Transformer AE and with drift replay.

### How it is used in the implementation

Implemented in `mdc_preprocess_vNext_mc_kaggle.ipynb`. Final freeze shapes (`manifest_vnext.json`): train `(16795, 10, 163)`, val/test `(5153, 10, 163)` each; test attack rate ≈ 0.382. Session split (70/30) and benign-only scaler fitting support the leakage-free claim (`leakage_free: true`).

---

## 3.6 Online / Incremental Learning

### What it is

Updating model parameters on newly observed data without a full retrain from scratch [7], [8].

### How it works

A common pattern is short fine-tuning on a recent buffer of presumed-benign samples with a small learning rate.

### Why it is suitable for Module 4

If benign traffic drifts, a frozen encoder may mis-rank scores. Limited fine-tuning is a candidate remediation—though contamination and catastrophic forgetting are risks [9].

### How it is used in the implementation

`fine_tune_on_recent()` in `mdc_drift_aware_kaggle.ipynb` deep-copies the model and runs a small number of AdamW MSE steps (`FINETUNE_STEPS=50`, `lr=1e-5`, up to 200 benign windows from the drift slice). Validation on the drift-aware stream reports AUC **0.7194 → 0.7420** (Δ ≈ **+0.023**) in `finetune_report.json`, demonstrating that error-triggered incremental fine-tuning improves held-out/stream detection performance relative to the frozen offline model. This is treated as **fully implemented** for the project’s intended scope: a validated sample of online retraining inside the Module 4 evaluation pipeline (not a separate production deployment service).

---

## 3.7 PSI and KS-Style Statistical Drift Monitoring

### What it is

**PSI** measures divergence between a baseline and current feature histogram [10]. **KS** statistics measure the maximum gap between cumulative distributions [11]. Both are unsupervised shift indicators.

### How it works

Module 4 stores train-benign quantile baselines in `drift_baseline_vnext.npz`. Offline, per-feature PSI and a KS-*proxy* (maximum absolute quantile gap versus baseline quantiles) are written to `drift_report_vnext.npz`. In the stream simulator, mean PSI is recomputed periodically (`PSI_CHECK_EVERY=100`) and compared with `PSI_TRIGGER=0.10`.

### Why it is suitable for Module 4

Holdout traffic is expected to differ from benign train; quantifying that shift motivates adaptive thresholds and retrain triggers.

### How it is used in the implementation

- Model notebook §14: offline PSI/KS report (median PSI ≈ 0.292, median KS-proxy ≈ 0.617 across 163 features in the freeze).
- Drift-aware notebook: PSI used as a streaming drift signal / fine-tune trigger context.

**Important distinction versus proposal wording:** monitoring is primarily on **feature distributions**, not on a formal Kolmogorov–Smirnov test of the **reconstruction-error** distribution. The “KS” quantity is a quantile-gap proxy, not necessarily `scipy.stats.ks_2samp`.

---

## 3.8 Adaptive Thresholding

### What it is

A decision threshold that updates from a rolling estimate of recent benign scores rather than remaining fixed forever [12].

### How it works

In alert space (higher = more anomalous), a buffer of recent benign scores is maintained. Periodically (every `THRESHOLD_UPDATE_EVERY=20` steps), a quantile-based estimator (final Phase-1 config: target percentile 80) updates the adaptive threshold, optionally with envelope constraints relative to the fixed offline threshold.

### Why it is suitable for Module 4

Under shift, a fixed `f1_optimal` threshold may over-alert. Adaptive updates aim to preserve detection while reducing false positives—evaluated at matched operating points.

### How it is used in the implementation

`AdaptiveThresholdSimulator` in the drift-aware evaluation helpers. Final freeze (`adaptive_report.json`, `matched_policy_comparison.json`, `freeze_manifest_v2.json`): `claim_adaptive_ok=true`, interpretation `adaptive_supported`. Last-half stream: fixed FPR 0.2767 → adaptive 0.2035 (reduction ≈ 0.073); F1 0.6824 → 0.7091; recall 0.7842 → 0.7571. Caveat: late-stream benign re-insertion and score calibration (`stream_composition.json`) are experimental controls that affect how strongly the result generalises.

---

## 3.9 Supporting Technologies

| Technology | Role in Module 4 |
|------------|------------------|
| StandardScaler + clipping | Stable benign-only scaling; avoids RobustScaler IQR collapse issues observed historically |
| Optuna TPE HPO | Search over depth, bottleneck, dropout, LR, noise, contractive λ (12 trials) |
| Isolation Forest | Classical unsupervised baseline on pooled window features |
| Dense AE | Deep baseline without Transformer temporal attention |
| NumPy / Pandas | Window tensors, manifests, stream logs |
| Kaggle notebook zip handoff | Reproducible multi-notebook pipeline without a `module4/src` package |

---

## 3.10 Technique-to-Novelty Map

| Proposed novelty | Status | Key improvement / evidence metrics |
|------------------|--------|-------------------------------------|
| Online / incremental learning | ✅ Full | Drift-slice AUC **0.7194 → 0.7420** (**Δ +0.0226**, ~**+3.1% relative**) |
| Statistical monitoring (PSI/KS-proxy) | ⚠️ Partial | Offline feature PSI median **≈0.292**, max **≈0.685**; KS-proxy median **≈0.617** (shift detected; not recon-error KS) |
| Adaptive thresholding | ✅ Full (caveated) | Last-half FPR **0.2767 → 0.2035 (−0.073)**; F1 **0.6824 → 0.7091 (+0.027)**; recall **0.7842 → 0.7571 (−0.027)** |
| True drift-aware detection | ⚠️ Partial | Combines adaptive + FT gains above; KS-on-error still incomplete |

### Metric evidence detail

**Online / incremental fine-tune (`finetune_report.json`)**

| Metric | Before FT | After FT | Absolute Δ | Relative Δ |
|--------|----------:|---------:|-----------:|-----------:|
| Drift-slice ROC-AUC | 0.7194 | 0.7420 | **+0.0226** | **+3.1%** |

**Adaptive vs fixed threshold — last-half stream (`matched_policy_comparison.json`)**

| Metric | Fixed | Adaptive | Absolute Δ | Relative Δ |
|--------|------:|---------:|-----------:|-----------:|
| FPR | 0.2767 | 0.2035 | **−0.0732** | **−26.5%** |
| F1 | 0.6824 | 0.7091 | **+0.0268** | **+3.9%** |
| MCC | 0.4861 | 0.5396 | **+0.0535** | **+11.0%** |
| Precision | 0.6039 | 0.6669 | **+0.0629** | **+10.4%** |
| Recall | 0.7842 | 0.7571 | **−0.0271** | −3.5% (small trade-off) |

**Statistical drift monitor (offline features, `drift_report_vnext.npz`)**

| Signal | Median | Max | Interpretation |
|--------|-------:|----:|----------------|
| PSI (163 features) | ≈0.292 | ≈0.685 | Many features above “major shift” (~0.25) |
| KS-proxy | ≈0.617 | ≈3.287 | Large quantile gaps vs train benign baseline |
| Stream PSI trigger | 0.10 mean PSI | — | Used in drift-aware loop |

**Context — offline detector ceiling (not a novelty, but related performance)**

| Config | ROC-AUC | F1 | FPR |
|--------|--------:|---:|----:|
| Default AE | 0.6889 | 0.6603 | 0.2911 |
| HPO AE | **0.8446** | **0.8030** | **0.0546** |
| Dense AE / IF mean_max | 0.6624 / 0.6741 | — | — |


---

## References

[1] G. E. Hinton and R. R. Salakhutdinov, “Reducing the dimensionality of data with neural networks,” *Science*, 2006.  
[2] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” MLSDA, 2014.  
[3] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv:1901.03407, 2019.  
[4] S. Rifai et al., “Contractive auto-encoders,” ICML, 2011.  
[5] A. Vaswani et al., “Attention is all you need,” NeurIPS, 2017.  
[6] P. Malhotra et al., “LSTM-based encoder-decoder for multi-sensor anomaly detection,” arXiv:1607.00148, 2016.  
[7] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” SDM, 2007.  
[8] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, 2014.  
[9] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks,” 1989.  
[10] Industry/MLOps practice and statistical treatments of Population Stability Index (PSI).  
[11] F. J. Massey Jr., “The Kolmogorov-Smirnov test for goodness of fit,” *JASA*, 1951.  
[12] M. H. Bhuyan et al., “Network anomaly detection: Methods, systems and tools,” *IEEE COMST*, 2014.
