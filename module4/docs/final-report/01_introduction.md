# Chapter 1 — Introduction

**Module:** Module 4 — Misuse Detection in Containers (MDC vNext)  
**Scope:** Final-report documentation for Module 4 only  
**Evidence basis:** Canonical freeze under `module4/notebook/final/`

---

## 1.1 Background and Motivation

Containerised microservice deployments have become a dominant execution model in cloud and edge environments. Isolation between containers is weaker than isolation between virtual machines: shared kernels, exposed orchestration APIs, and densely packed network services expand the attack surface [1], [2]. When a container is compromised—through a web application CVE, remote code execution, container escape, or reconnaissance activity—the resulting misuse often appears first as a behavioural deviation in telemetry rather than as a signature that a static rule set already knows.

Traditional network intrusion detection systems (NIDS) and supervised classifiers can achieve strong accuracy when attack labels and attack coverage are adequate [3]. In practice, labelled container-attack data are scarce and attack campaigns evolve quickly. Supervised models trained on a closed set of attack classes therefore face two structural limits: (i) limited generalisation beyond the labelled attack set, and (ii) expensive, continuously refreshed labelled corpora.

Unsupervised anomaly detection addresses these limits by modelling *normal* behaviour and treating large deviations as suspicious [4], [5]. Autoencoders (AEs) are a widely used realisation of this idea: they learn a compressed representation of benign inputs and flag examples with high reconstruction error [6], [7]. For sequential telemetry, attention-based sequence models—most notably Transformers—can capture temporal dependencies without recurrent bottlenecks [8]. Combining an attention-enhanced sequence autoencoder with reconstruction-error scoring therefore offers a principled route to container misuse detection when only benign windows are available for training.

A second practical challenge is **distribution drift**. Container workloads, traffic mixes, and attack campaigns change over time. A detector whose decision boundary is frozen at training time may suffer rising false-positive rates or missed detections when the benign score distribution shifts [9], [10]. Drift monitoring (for example via population stability indices or distributional distance statistics) and adaptive decision thresholds are therefore relevant operational complements to a static offline model.

This Module 4 project develops and evaluates **MDC vNext**: an unsupervised, reconstruction-error-based detector for container network-flow windows derived from the Misuse Detection in Containers (MDC) dataset of Sever and Doğan [11], extended with a drift-aware simulated streaming layer (adaptive thresholding, PSI-based monitoring, and optional incremental fine-tuning). The implemented pipeline uses CICFlowMeter-style network flow features only; multi-modal CPU, memory, system-call, or file-access inputs mentioned in earlier proposal wording are **not** present in the final window artefacts.

---

## 1.2 Aim and Objectives

### Aim

To design, implement, and evaluate an unsupervised attention-enhanced autoencoder for detecting container misuse from network-flow telemetry, using reconstruction error as the primary anomaly signal, and to assess drift-aware monitoring mechanisms under leakage-aware preprocessing and integrity-gated evaluation.

### Objectives

1. Construct a leakage-aware preprocessing pipeline that converts raw MDC flow records into fixed-length scaled windows suitable for sequence modelling.
2. Train a Transformer bottleneck autoencoder exclusively on benign windows and score anomalies via reconstruction error.
3. Establish robust offline evaluation using validation-tuned thresholds, multi-seed training, and hyperparameter optimisation (HPO).
4. Compare the detector against classical and deep unsupervised baselines on the same windows.
5. Implement and evaluate a drift-aware monitoring layer comprising sliding-window benign buffering, adaptive thresholding, statistical drift signals (PSI / KS-proxy), and incremental fine-tuning under **simulated streaming replay**, with validated improvement on the evaluation path.
6. Document, with evidence, which proposed novelties are fully realised, partially realised, or insufficiently evaluated.

---

## 1.3 Proposed Solution Overview

### Users

| User group | Role relative to Module 4 |
|------------|---------------------------|
| Researchers / dissertation evaluators | Reproduce notebooks, inspect metrics and freeze artefacts |
| Security analysts (intended consumers) | Consume anomaly scores and alerts (integration demo path) |
| Developers / integrators | Load the exported HPO checkpoint for inference |

There is no standalone production API inside `module4/`; the research system is notebook-driven, with an optional integration handover model for demonstration.

### Inputs

| Input | Description |
|-------|-------------|
| Raw MDC CSV | CIC-style container network flows (Kaggle dataset) |
| Preprocessed windows | Tensors of shape `(B, 10, 163)` — 10 timesteps × 163 bucketed features |
| Optional | Drift baseline quantiles; offline threshold from `metrics_vnext.json` |

### Outputs

| Output | Description |
|--------|-------------|
| Continuous anomaly score | Per-window reconstruction-error-derived score (higher = more anomalous after score protocol) |
| Binary alert | Score compared with a fixed or adaptive threshold |
| Evaluation artefacts | ROC/PR metrics, confusion counts, stream logs, PSI timelines |
| Model checkpoint | `checkpoint_vnext.pt` / standalone HPO `.pt` for integration |

### Process (high level)

```text
Raw CIC flows
    → sessionisation & leakage-safe split
    → hygiene, scaling, 15 s bucketing
    → sliding windows (T=10, stride=2)
    → benign-only SequenceBottleneckAE training
    → reconstruction-error scoring + validation threshold
    → offline test evaluation (+ multi-seed / HPO)
    → simulated timestamp-ordered stream
         (fixed thr | adaptive thr | PSI | optional fine-tune)
```

### Technologies

| Layer | Technology |
|-------|------------|
| Language | Python 3.11–3.12 (Kaggle runtime) |
| Deep learning | PyTorch (`SequenceBottleneckAE`) |
| Classical ML | scikit-learn (`StandardScaler`, `VarianceThreshold`, Isolation Forest) |
| HPO | Optuna (TPE; 12 trials in final config) |
| Numerics / IO | NumPy, Pandas, joblib |
| Platform | Kaggle notebooks (canonical final track) |

### Main features

- Benign-only unsupervised training (no attack labels at train time)
- Attention-enhanced sequence autoencoder (Transformer encoder–bottleneck–decoder)
- Reconstruction-error anomaly scoring with auto score-orientation safety
- Validation-tuned primary operating point (`f1_optimal`)
- Multi-seed robustness and Optuna HPO
- Drift-aware simulated streaming (adaptive threshold, PSI, optional fine-tune)
- Baseline comparison (Dense AE, Isolation Forest)

### System requirements

| Item | Requirement (as evidenced) |
|------|----------------------------|
| Training | GPU recommended (Kaggle GPU / RTX-class); exact freeze GPU not pinned in artefacts |
| Inference | CPU feasible; median latency on the order of ~1.8–3.4 ms/window in reported runs |
| Dependencies | Unpinned `pip install` style in notebooks; no `requirements.txt` lockfile under `module4/` |
| Data | Access to MDC Kaggle dataset `yigitsever/misuse-detection-in-containers-dataset` |

---

## 1.4 Dissertation Structure

This Module 4 final-report documentation is organised as five companion Markdown files that mirror a conventional academic dissertation structure for the module:

| File | Title | Role |
|------|-------|------|
| `01_introduction.md` | Introduction | Background, aim, objectives, solution overview, structure |
| `02_literature_review.md` | Literature Review | Related work, strengths/limitations, research gap |
| `03_technology_adapted.md` | Technologies and Techniques Adopted | Techniques used and why they fit Module 4 |
| `04_approach_analysis_and_design.md` | Approach, Analysis and Design | Combined method, architecture, novelty verification |
| `05_implementation_and_discussion.md` | Implementation, Discussion and Evaluation | Implementation detail, experiments, results, limitations |

Cross-chapter consistency is maintained as follows: claims about what the system *does* are grounded in `module4/notebook/final/`; quantitative claims cite frozen JSON/CSV artefacts under `output-metrics/`; proposed novelties are classified as fully, partially, or insufficiently evidenced rather than assumed from proposal text alone.

---

## References

[1] D. Bernstein, “Containers and cloud: From LXC to Docker to Kubernetes,” *IEEE Cloud Computing*, vol. 1, no. 3, pp. 81–84, 2014.  
[2] M. Souppaya, J. Morello, and K. Scarfone, “Application container security guide,” NIST SP 800-190, 2017.  
[3] A. L. Buczak and E. Guven, “A survey of data mining and machine learning methods for cyber security intrusion detection,” *IEEE Communications Surveys & Tutorials*, vol. 18, no. 2, pp. 1153–1176, 2016.  
[4] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” *ACM Computing Surveys*, vol. 41, no. 3, pp. 1–58, 2009.  
[5] M. Ahmed, A. N. Mahmood, and J. Hu, “A survey of network anomaly detection techniques,” *Journal of Network and Computer Applications*, vol. 60, pp. 19–31, 2016.  
[6] M. Sakurada and T. Yairi, “Anomaly detection using autoencoders with nonlinear dimensionality reduction,” in *Proc. MLSDA*, 2014.  
[7] R. Chalapathy and S. Chawla, “Deep learning for anomaly detection: A survey,” arXiv:1901.03407, 2019.  
[8] A. Vaswani et al., “Attention is all you need,” in *Proc. NeurIPS*, 2017.  
[9] J. Gama et al., “A survey on concept drift adaptation,” *ACM Computing Surveys*, vol. 46, no. 4, pp. 1–37, 2014.  
[10] G. I. Webb, R. Hyde, H. Cao, H. L. Nguyen, and F. Petitjean, “Characterizing concept drift,” *Data Mining and Knowledge Discovery*, vol. 30, no. 4, pp. 964–994, 2016.  
[11] Y. Sever and A. Doğan, Misuse Detection in Containers (MDC) dataset, ITU Journal / Kaggle, 2023.
