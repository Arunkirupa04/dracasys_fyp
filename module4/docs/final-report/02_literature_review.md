# Chapter 2 — Literature Review: Anomaly Detection, Autoencoders, and Drift-Aware Monitoring for Container Security

**Module:** Module 4 — MDC vNext  
**Purpose:** Position Module 4 against existing work and identify the research gap addressed by the implementation.

---

## 2.1 Background of the Problem Domain

Container platforms package applications with their dependencies and share the host kernel, which improves density but weakens isolation relative to full virtualisation [1], [2]. Documented threats include vulnerable application services, privilege escalation, container escape, and lateral movement through the cluster network fabric [2], [3]. Telemetry available for detection ranges from host metrics (CPU, memory) to network flows, system calls, and filesystem events. Network-flow features—packet and byte counts, durations, flag patterns, and related CICFlowMeter-style aggregates—are attractive because they are comparatively portable across orchestrators and do not require invasive in-container agents [4].

The Misuse Detection in Containers (MDC) dataset of Sever and Doğan [5] provides labelled container network telemetry spanning benign behaviour and multiple attack scenarios (CVE-driven exploits, Node-RED misuse, scanning). Module 4 uses this dataset as its empirical substrate. Attack labels are used for evaluation; training remains benign-only.

---

## 2.2 Existing Anomaly Detection Approaches

Chandola et al. [6] classify anomaly detection into statistical, proximity-based, clustering, classification, and information-theoretic families. In cybersecurity, network anomaly detection has historically used statistical thresholds, clustering, and one-class classifiers [7], [8].

**Classical unsupervised methods.** Isolation Forest (IF) isolates anomalies via random partitioning and is computationally efficient [9]. One-Class SVM (OCSVM) learns a boundary around normal data in a kernel space [10]. Principal Component Analysis (PCA) and related subspace methods flag large reconstruction residuals in a linear latent space [11]. These methods are strong baselines but often struggle with high-dimensional sequential structure unless features are carefully engineered or flattened.

**Deep anomaly detection.** Deep learning approaches model complex manifolds of normal behaviour [12], [13]. Autoencoders, variational autoencoders (VAEs), generative adversarial networks (GANs), and recurrent sequence models have all been applied to intrusion and system-log anomaly detection [12], [14]. Strengths include representation power; limitations include sensitivity to preprocessing, threshold choice, training instability, and difficulty interpreting alerts.

**Supervised and hybrid NIDS.** Supervised deep classifiers on labelled IDS corpora (e.g., CIC-IDS variants) can report high accuracy [8], but they inherit closed-world assumptions: unseen attack classes and concept drift degrade performance unless continual labelling is available.

---

## 2.3 Autoencoder-Based Anomaly Detection

Sakurada and Yairi [15] demonstrated nonlinear AE dimensionality reduction for anomaly detection. Subsequent work established reconstruction error—typically mean squared error (MSE) between input and output—as a practical anomaly score when models are trained on normal data only [12], [16]. Contractive autoencoders penalise sensitivity of latent codes to input perturbations, encouraging locally stable representations of the normal manifold [17]. Denoising autoencoders train the model to reconstruct clean inputs from corrupted copies, further regularising the learned manifold [18].

For intrusion detection, AE-based systems have been applied to flow and packet features with competitive results against classical baselines [12], [19]. Common limitations include: (i) score orientation problems when attack samples are accidentally reconstructible; (ii) brittle fixed thresholds; and (iii) degradation under distribution shift after deployment.

---

## 2.4 Attention Mechanisms and Sequence Models

Recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks have been widely used for sequential anomaly detection [14], [20]. Transformers replace recurrence with multi-head self-attention, enabling parallel training and long-range dependency modelling [21]. Attention-enhanced autoencoders and Transformer autoencoders have been explored for time-series and multivariate anomaly detection [22], [23].

**Relevance to Module 4.** Container flow telemetry is naturally sequential when aggregated into time buckets. Module 4’s `SequenceBottleneckAE` uses Transformer encoder and decoder layers with a low-dimensional bottleneck, so attention is the mechanism by which temporal structure is encoded—matching the proposed “attention-enhanced autoencoder” framing without requiring a separate custom attention module.

---

## 2.5 Reconstruction-Error-Based Detection

Reconstruction-error detectors assume that normal points lie on a learned manifold while anomalies do not [6], [12]. Operating points are typically set from validation score distributions (percentiles of benign scores, Youden’s *J*, or F1-optimal thresholds when labels are available for calibration only) [24]. Important methodological pitfalls include:

- **Score inversion:** if attacks reconstruct better than benign samples, raw MSE is anti-correlated with “attackness” and must be flipped or redesigned.
- **Threshold leakage:** fitting thresholds on test scores invalidates claimed generalisation.
- **Imbalance:** ROC-AUC can look optimistic when precision–recall behaviour is poor [25].

Module 4 addresses orientation with an automatic score-flip protocol on validation AUC and fits primary thresholds on validation scores only.

---

## 2.6 Online and Incremental Learning

Batch models become stale when the data-generating process changes. Incremental and online learning update model parameters as new samples arrive [26], [27]. In anomaly detection, common strategies include: (i) continual fine-tuning on recent presumed-normal samples; (ii) replay buffers; and (iii) retraining triggers fired by drift detectors. Risks include **catastrophic forgetting**—overwriting previously useful decision structure—and **contamination**, where attacks enter the “benign” update buffer [28].

Module 4 implements incremental fine-tuning (`fine_tune_on_recent`) inside the drift-aware evaluation stream and validates that the update improves detection AUC on the held-out/stream test path (see Chapter 5).

---

## 2.7 Concept Drift Detection and Statistical Monitoring

Concept drift denotes a change in the joint distribution of features and labels over time [29], [30]. Drift detectors may monitor error rates (when labels arrive) or unsupervised distributional divergence (when labels do not).

**Population Stability Index (PSI)** is widely used in credit scoring and MLOps to quantify feature distribution shift against a baseline; values above ~0.25 are often treated as major shift [31]. **Kolmogorov–Smirnov (KS)** tests compare empirical cumulative distributions [32]. Related distances (KL, Wasserstein, MMD) are also used [30].

Many production pipelines monitor *feature* drift rather than *score* drift; score monitoring is more directly tied to alert rates. Module 4’s offline drift report computes PSI and a KS-style quantile gap per feature versus a benign-train baseline; the streaming layer primarily uses mean PSI as a trigger.

---

## 2.8 Adaptive Thresholding

Fixed thresholds assume stationarity of the score distribution. Adaptive thresholds recalibrate the decision boundary from a rolling window of recent scores, often restricted to presumed-benign samples [33], [34]. Typical estimators include rolling mean ± *k* standard deviations, quantiles, or exponentially weighted statistics. Evaluation must compare policies at **matched false-positive rates**; otherwise an “FPR reduction” may simply reflect a stricter threshold that also destroys recall.

Module 4’s adaptive layer uses a benign score buffer with periodic quantile-based updates and matched-policy comparison gates—directly reflecting this methodological requirement.

---

## 2.9 Comparative Summary of Related Approaches

| Approach | Training labels | Sequential? | Drift handling | Typical strength | Typical limitation |
|----------|-----------------|-------------|----------------|------------------|--------------------|
| Isolation Forest [9] | Benign / unsupervised | Usually flattened | Rarely native | Fast baseline | Weak on structured sequences unless pooled carefully |
| OCSVM [10] | Benign | Rarely | Rarely | Solid classical boundary | Scalability; kernel choice |
| Dense / shallow AE [15], [16] | Benign | Optional | Rarely | Simple recon-error detector | Limited temporal modelling |
| LSTM-AE [14], [20] | Benign | Yes | Rarely | Temporal modelling | Slower train; vanishing long-range context |
| Transformer / Attn-AE [21]–[23] | Benign | Yes | Rarely | Long-range attention | Data/compute hungry; threshold brittle |
| Supervised DNN-NIDS [8] | Attack + benign | Optional | Needs retrain | High in-distribution accuracy | Poor unknown-class generalisation |
| Drift monitors (PSI/KS) [31], [32] | N/A (monitor) | N/A | Yes | Operational shift alarms | Not a detector by itself |
| **Module 4 (MDC vNext)** | Benign-only AE | Yes (T=10) | Simulated adaptive + PSI + FT | Integrated AE + drift layer on MDC | Single dataset; stream not live; some novelties partial |

---

## 2.10 Research Gap and Module 4 Positioning

From the survey, gaps recur:

1. Container-specific sequential unsupervised detection with leakage-aware windowing on a public container misuse corpus remains less mature than generic IDS benchmarks.
2. Drift-aware decisioning—not only offline AE accuracy—is often omitted or evaluated without integrity gates (score orientation, matched FPR).
3. Honest novelty accounting is rare: proposal claims of online learning and KS-on-error are frequently stronger than the experimental protocol.

Module 4 addresses gap (1) with a Transformer bottleneck AE on MDC flow windows and leakage-safe preprocessing. It addresses gap (2) with a simulated streaming layer, adaptive thresholding, PSI monitoring, and validated incremental fine-tuning under integrity checks. Gap (3) is addressed in this documentation by classifying each proposed novelty against repository evidence rather than equating proposal text with implementation.

**What Module 4 does not yet close:** multi-dataset validation; live temporal deployment; classical KS tests on reconstruction-error distributions.

---

## References

[1] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, vol. 1, no. 3, pp. 81–84, 2014.  
[2] M. Souppaya, J. Morello, and K. Scarfone, “Application container security guide,” NIST SP 800-190, 2017.  
[3] A. Martin et al., “Docker ecosystem – vulnerability analysis,” *Computer Communications*, vol. 122, pp. 30–43, 2018.  
[4] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward generating a new intrusion detection dataset and intrusion traffic characterization,” in *Proc. ICISSP*, 2018.  
[5] Y. Sever and A. Doğan, Misuse Detection in Containers (MDC) dataset, ITU Journal / Kaggle, 2023.  
[6] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” *ACM Computing Surveys*, vol. 41, no. 3, pp. 1–58, 2009.  
[7] M. Ahmed, A. N. Mahmood, and J. Hu, “A survey of network anomaly detection techniques,” *JNCA*, vol. 60, pp. 19–31, 2016.  
[8] A. L. Buczak and E. Guven, “A survey of data mining and machine learning methods for cyber security intrusion detection,” *IEEE COMST*, vol. 18, no. 2, pp. 1153–1176, 2016.  
[9] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation Forest,” in *Proc. ICDM*, 2008.  
[10] B. Schölkopf et al., “Support vector method for novelty detection,” in *Proc. NeurIPS*, 2000.  
[11] M.-L. Shyu et al., “A novel anomaly detection scheme based on principal component classifier,” in *Proc. ICDM Workshop*, 2003.  
[12] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv:1901.03407, 2019.  
[13] G. Pang, C. Shen, L. Cao, and A. V. D. Hengel, “Deep learning for anomaly detection: A review,” *ACM Computing Surveys*, vol. 54, no. 2, pp. 1–38, 2021.  
[14] P. Malhotra et al., “LSTM-based encoder-decoder for multi-sensor anomaly detection,” arXiv:1607.00148, 2016.  
[15] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” in *Proc. MLSDA*, 2014.  
[16] J. An and S. Cho, “Variational autoencoder based anomaly detection using reconstruction probability,” SNU Data Mining Center Tech. Report, 2015.  
[17] S. Rifai et al., “Contractive auto-encoders: Explicit invariance during feature extraction,” in *Proc. ICML*, 2011.  
[18] P. Vincent et al., “Extracting and composing robust features with denoising autoencoders,” in *Proc. ICML*, 2008.  
[19] A. Javaid et al., “A deep learning approach for network intrusion detection system,” in *Proc. EAI*, 2016.  
[20] B. Lindemann et al., “A survey on anomaly detection for technical systems using LSTM networks,” *Computers in Industry*, vol. 131, 2021.  
[21] A. Vaswani et al., “Attention is all you need,” in *Proc. NeurIPS*, 2017.  
[22] J. Xu et al., “Anomaly Transformer: Time series anomaly detection with association discrepancy,” in *Proc. ICLR*, 2022.  
[23] S. Tuli, G. Casale, and N. R. Jennings, “TranAD: Deep transformer networks for anomaly detection in multivariate time series data,” *VLDB*, 2022.  
[24] C. X. Ling, J. Huang, and H. Zhang, “AUC: A statistically consistent and more discriminating measure than accuracy,” in *Proc. IJCAI*, 2003.  
[25] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, 2015.  
[26] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” in *Proc. SDM*, 2007.  
[27] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, vol. 46, no. 4, 2014.  
[28] M. McCloskey and N. J. Cohen, “Catastrophic interference in connectionist networks,” *Psychology of Learning and Motivation*, vol. 24, 1989.  
[29] G. I. Webb et al., “Characterizing concept drift,” *DMKD*, vol. 30, pp. 964–994, 2016.  
[30] J. Lu et al., “Learning under concept drift: A review,” *IEEE TKDE*, vol. 31, no. 12, pp. 2346–2363, 2019.  
[31] B. Yurdakul, “Statistical properties of population stability index,” Ph.D. dissertation / applied statistics literature on PSI, Western Michigan Univ., 2018 (and industry MLOps practice).  
[32] F. J. Massey Jr., “The Kolmogorov-Smirnov test for goodness of fit,” *JASA*, vol. 46, no. 253, pp. 68–78, 1951.  
[33] D. M. Hawkins, *Identification of Outliers*. Chapman and Hall, 1980.  
[34] M. H. Bhuyan, D. K. Bhattacharyya, and J. K. Kalita, “Network anomaly detection: Methods, systems and tools,” *IEEE COMST*, vol. 16, no. 1, pp. 303–336, 2014.  
[35] R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network intrusion detection,” in *Proc. IEEE S&P*, 2010.
