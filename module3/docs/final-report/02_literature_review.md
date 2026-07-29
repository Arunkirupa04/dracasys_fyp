# Chapter 2 — Literature Review: System Performance Anomaly Detection, VAEs, and Drift-Aware Monitoring

**Module:** Module 3 — System Anomaly Detector (Drift-Aware Approach)  
**Purpose:** Position Module 3 against existing work and identify the research gap addressed by the as-built system.

---

## 2.1 System Performance Anomaly Detection

Anomaly detection seeks patterns that deviate from expected behaviour [1]. In computing systems, performance anomalies include resource saturation, leaks, thrashing, and service-level degradations that may not match a known fault signature [2], [3]. Classical approaches combine expert thresholds, statistical control charts, and rule engines. These methods are interpretable but struggle when baselines shift and when anomalies are multivariate [4].

Module 3 focuses on **container-level performance faults** visible in CPU and memory KPIs—CPU saturation, memory pressure/leaks, and pod failures—rather than network intrusion detection.

---

## 2.2 Machine-Learning-Based Anomaly Detection

Supervised classifiers require labelled faults and often fail to generalise to unseen fault types [5]. Semi-supervised and unsupervised methods train primarily on normal data and score novelty [1], [6]. Popular classical baselines include Isolation Forest [7], One-Class SVM [8], and PCA residual methods [9]. Deep learning surveys document strong results for autoencoder-family detectors on high-dimensional telemetry [10], [11].

Module 3 adopts an unsupervised reconstruction paradigm and compares against Isolation Forest and a deterministic autoencoder on the same windows.

---

## 2.3 Autoencoders and Reconstruction-Error Scoring

An autoencoder (AE) compresses input \(x\) to a latent code and reconstructs \(\hat{x}\). When trained on normal data, anomalous inputs tend to yield larger \(\|x-\hat{x}\|\) [12], [13]. Temporal variants (LSTM-AE) capture sequential structure [14]. Scoring may use mean squared error, maximum pooled residuals, or probabilistic reconstructions.

Module 3 uses **mean reconstruction MSE** over whitened PCA features, computed from the VAE encoder mean (deterministic at inference). Side experiments with max-pool scoring degraded PR-AUC and were rejected.

---

## 2.4 Variational Autoencoders for Anomaly Detection

A Variational Autoencoder (VAE) learns an approximate posterior \(q(z\mid x)\) and regularises the latent space with a KL term toward a prior [15]. An and Cho popularised VAE-based anomaly scoring via reconstruction probability [16]. In practice, many deployments use reconstruction MSE with a carefully tuned KL weight to avoid posterior collapse [17].

Module 3’s VAE uses a small \(\beta_{\max}=0.01\) with linear KL warmup after an explicit beta search that found larger \(\beta\) values collapsed the latent posterior. Latent dimension 32 was selected by validation loss ablation.

---

## 2.5 PCA for Dimensionality Reduction and Multi-Metric Correlation

Principal Component Analysis (PCA) projects correlated features onto orthogonal components ordered by variance [18]. In anomaly detection, PCA residuals or whitened components support multi-sensor correlation analysis [9]. Whitening equalises component scales so MSE is not dominated by high-variance axes.

Module 3 flattens each \((30,7)\) window to 210 dimensions, fits PCA retaining **99% variance with whitening** on train-only windows, and feeds **26** components to the VAE. The 99% choice was motivated by retaining small-variance CPU-carrying components that a 95% cutoff risked discarding.

---

## 2.6 Concept Drift Detection

Concept drift denotes changes in the data-generating distribution over time [19], [20]. Detectors include error-rate monitors, windowed statistics, ADWIN, and distributional tests such as the Kolmogorov–Smirnov (KS) test [21], [22]. In anomaly detection, drift may appear as a shift in the score distribution of presumed-normal samples, not only as a change in raw features.

Module 3 monitors reconstruction-error distributions with a **two-sample KS test** against a frozen train-error reference every 5,000 windows during stream replay.

---

## 2.7 Online and Incremental Learning

Online and incremental learners update models as new samples arrive, avoiding full offline retrain [19], [23]. Risks include catastrophic forgetting and contamination when “recent normal” buffers contain silent faults [24]. Lightweight fine-tuning with reduced learning rates is a common compromise for neural detectors.

Module 3 implements incremental fine-tuning of the VAE on a self-selected normal buffer when the KS test rejects the null. The proposal text refers to “SGD”; the implemented optimiser is **Adam** with lr \(10^{-4}\) for five epochs—functionally a small incremental update, not a full retrain.

---

## 2.8 Adaptive Thresholding

Fixed thresholds calibrated once on a validation set degrade under drift [4], [20]. Adaptive schemes recompute decision boundaries from recent score statistics (mean + \(k\)·std, percentiles, or Bayesian shrinkage toward a prior) [25]. Shrinkage / empirical-Bayes blends stabilise estimates when local buffers are small [26].

Module 3’s final adaptive mechanism uses a **blended mean + \(k\)·blended std** with per-container train anchors and prior strength 500. A naive recent-only mean+3·std experiment historically harmed F1 and was superseded.

---

## 2.9 Real-Time Detection in Container Environments

Container monitoring stacks (cAdvisor, Prometheus) expose fine-grained gauges and counters [2], [27]. Real-time detectors must keep latency low and memory bounded. Sliding windows over recent timesteps are a standard pattern for temporal context without storing full histories [28].

Module 3’s canonical window is **30 steps × 15 s = 7.5 minutes**, stride 1, with gap-aware construction. The VAE has 10,778 parameters and sub-millisecond CPU inference per window in recorded measurements.

---

## 2.10 Benchmarks and Fault-Injection Datasets

Public microservice fault-injection corpora (including AIOpsArena-style Sock-Shop / Chaos Mesh setups) provide labelled performance faults for evaluation [29]. Care is required when network faults are present but the feature set excludes network KPIs.

Module 3 trains and tests on AIOpsArena **complex_case1**, reports drift metrics on **complex_case2**, and explicitly excludes network delay/loss from the detectable label space.

---

## 2.11 Comparative Summary

| Approach | Typical input | Drift handling | Strength | Typical limitation |
|----------|---------------|----------------|----------|--------------------|
| Static thresholds | Single KPI | None | Simple | Brittle under regime change |
| Isolation Forest | Feature vector | None (retrain) | Strong classical baseline | Weak sequential structure |
| Deterministic AE | Window / vector | Rare | Learns nonlinear normal manifold | Fixed threshold ages poorly |
| VAE (offline) | Window / vector | None | Probabilistic latent + recon error | Drift degrades operating point |
| Ensemble / heavy AD | Multi-model | Sometimes | High accuracy potential | Costly for edge/container agents |
| **Module 3 (as built)** | PCA-whitened CPU/mem windows | Blended adaptive thr + KS fine-tune | Lightweight; leak-free eval | Network faults out of scope |

---

## 2.12 Research Gap and Module 3 Positioning

From the survey, three gaps are relevant:

1. **Container performance AD** often stops at offline AE accuracy without leak-free threshold discipline under severe class imbalance.
2. **Drift-aware decisioning** (adaptive thresholds + distributional monitoring + incremental update) is frequently proposed but incompletely evaluated as an ablation stack.
3. **Honest scope accounting** is rare: multi-metric claims sometimes include network/syscall modalities that are absent from the implementation.

Module 3 addresses gap (1) with a PCA+VAE pipeline on AIOpsArena CPU/memory windows, PR-AUC-primary evaluation, and validation-only threshold calibration. It addresses gap (2) with blended adaptive thresholding and KS-triggered fine-tuning under simulated chronological replay, reported as ablations against VAE-alone. Gap (3) is addressed in this documentation by stating that **network bottleneck detection was proposed but not delivered**.

**What Module 3 does not claim to close:** multi-modal fusion with network/syscall telemetry; production continuous online serving of the full drift loop (integration ships VAE-alone); or a literal SGD optimiser as named in the proposal text.

---

## References

[1] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” *ACM Computing Surveys*, vol. 41, no. 3, 2009.  
[2] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, 2014.  
[3] B. Burns et al., “Borg, Omega, and Kubernetes,” *CACM*, 2016.  
[4] D. M. Hawkins, *Identification of Outliers*. Chapman and Hall, 1980.  
[5] R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network intrusion detection,” *IEEE S&P*, 2010.  
[6] M. Ahmed, A. N. Mahmood, and J. Hu, “A survey of network anomaly detection techniques,” *JNCA*, 2016.  
[7] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” *ICDM*, 2008.  
[8] B. Schölkopf et al., “Support vector method for novelty detection,” *NeurIPS*, 2000.  
[9] M.-L. Shyu et al., “A novel anomaly detection scheme based on principal component classifier,” *ICDM Workshop*, 2003.  
[10] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv:1901.03407, 2019.  
[11] G. Pang et al., “Deep learning for anomaly detection: A review,” *ACM Computing Surveys*, 2021.  
[12] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” *MLSDA*, 2014.  
[13] P. Malhotra et al., “LSTM-based encoder-decoder for multi-sensor anomaly detection,” arXiv:1607.00148, 2016.  
[14] B. Lindemann et al., “A survey on anomaly detection for technical systems using LSTM networks,” *Computers in Industry*, 2021.  
[15] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” *ICLR*, 2014.  
[16] J. An and S. Cho, “Variational autoencoder based anomaly detection using reconstruction probability,” SNU Tech. Report, 2015.  
[17] I. Higgins et al., “β-VAE: Learning basic visual concepts with a constrained variational framework,” *ICLR*, 2017.  
[18] I. T. Jolliffe, *Principal Component Analysis*, 2nd ed. Springer, 2002.  
[19] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, 2014.  
[20] J. Lu et al., “Learning under concept drift: A review,” *IEEE TKDE*, 2019.  
[21] F. J. Massey Jr., “The Kolmogorov-Smirnov test for goodness of fit,” *JASA*, 1951.  
[22] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” *SDM*, 2007.  
[23] G. I. Webb et al., “Characterizing concept drift,” *DMKD*, 2016.  
[24] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks,” *Psychology of Learning and Motivation*, 1989.  
[25] L. Ruff et al., “A unifying review of deep and shallow anomaly detection,” *Proc. IEEE*, 2021.  
[26] B. Efron, *Large-Scale Inference*. Cambridge University Press, 2010.  
[27] Prometheus / cAdvisor documentation (CNCF monitoring practice).  
[28] V. Chandola et al., survey sections on temporal anomaly detection [1].  
[29] AIOpsArena benchmark (`module3/docs/AIOpsArena.pdf`).
