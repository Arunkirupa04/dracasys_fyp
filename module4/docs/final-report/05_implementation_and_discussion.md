# Chapter 5 — Implementation, Discussion and Evaluation

**Module:** Module 4 — MDC vNext  
**Canonical metrics:** `module4/notebook/final/output-metrics/`  
**Canonical code:** `module4/notebook/final/kaggle-source/`  
**Design reference:** Chapter 4 (Figures 4.1–4.4)

---

## Part A — Implementation

## 5.1 Software Technologies and Environment

| Item | Implementation evidence |
|------|-------------------------|
| Language | Python 3.11–3.12 (Kaggle notebook metadata) |
| DL framework | PyTorch (`nn.TransformerEncoder` / `Decoder`) |
| Classical ML | scikit-learn (`StandardScaler`, `VarianceThreshold`, Isolation Forest) |
| HPO | Optuna TPE (12 trials in final config) |
| Numerics | NumPy, Pandas |
| Persistence | `torch.save` checkpoints; `np.savez_compressed`; JSON reports |
| Orchestration | Kaggle zip hand-offs between notebooks |
| Lockfile | **Not found** under `module4/` (unpinned installs) |

Hardware: GPU recommended for training/HPO; CPU inference used for latency reporting. Exact GPU SKU of the frozen Kaggle run is **not verified** from artefacts alone.

---

## 5.2 Repository Layout (Implementation Map)

```text
module4/notebook/final/
├── kaggle-source/          # runnable sources
│   ├── mdc_preprocess_vNext_mc_kaggle.ipynb
│   ├── mdc_model_vNext_kaggle.ipynb
│   ├── mdc_drift_aware_kaggle.ipynb
│   ├── mdc_baselines_kaggle.ipynb
│   └── mdc_analysis_kaggle.ipynb
├── kaggle-output/          # executed notebooks
├── models/                 # handover HPO .pt + MODEL_HANDOVER.md
└── output-metrics/         # frozen numbers cited below
```

There is no `module4/src/` package; research logic lives in notebooks. An integration copy of the model class exists at `integration/models/module4/sequence_bottleneck_ae.py` for demo loading only.

---

## 5.3 Data Preprocessing Implementation

**Notebook:** `mdc_preprocess_vNext_mc_kaggle.ipynb`  
**Freeze manifest:** `windows_vnext_processed/manifest_vnext.json`

| Stage | Implementation detail |
|------:|------------------------|
| Load | MDC CSV via Kaggle input / kagglehub |
| Sessions | Gap threshold 60 s; min flows filter |
| Split | Per-container session split `TRAIN_RATIO=0.70`, `RANDOM_STATE=42` |
| Hygiene | Inf/NaN handling |
| Filters | VarianceThreshold 0.01; correlation drop 0.95 |
| Scale | Flow `StandardScaler` + clip fit on train benign; bucket scaler + `POST_SCALE_CLIP=10` |
| Buckets | `BUCKET_FREQ='15s'`, aggregation `mean_max_std` + `flow_count` |
| Windows | `WINDOW_SIZE=10`, `STRIDE=2`, `ATTACK_FRAC_THRESHOLD=0.5` |
| Holdout | Stratified val/test 50/50 |
| Drift baseline | Quantiles / mean / std → `drift_baseline_vnext.npz` |

**Verified freeze shapes**

| Split | Shape | Attack rate |
|-------|-------|------------:|
| `X_train` | `(16795, 10, 163)` | benign-only (no `y_train`) |
| `X_val` | `(5153, 10, 163)` | 0.3821 |
| `X_test` | `(5153, 10, 163)` | 0.3819 |

`leakage_free: true` is recorded in the manifest. Multiclass IDs present in holdout: `{0,1,2,3,4,8,11}`; classes `{5,6,7,9,10}` are **absent** from holdout windows in this freeze.

---

## 5.4 Model Implementation

**Class:** `SequenceBottleneckAE` (model notebook; mirrored in `sequence_bottleneck_ae.py`)

Implementation points corresponding to Figure 4.3:

- `input_proj`: `Linear(F → d_model)`
- `SinusoidalPE`
- `TransformerEncoder` / `TransformerDecoder` (GELU, `norm_first=True`, `batch_first=True`)
- `bn_down` / `bn_up` bottleneck
- `pos_queries` as decoder targets (no input leakage through identity skip of raw `x` into the decoder memory path beyond encoded bottleneck)

**Attention mechanism:** provided by PyTorch multi-head attention inside Transformer layers—this is the attention enhancement referenced in the Module 4 core technique.

---

## 5.5 Training Pipeline Implementation

Default training recipe (model notebook / research record):

| Hyperparameter | Value |
|----------------|------:|
| Batch size | 32 |
| Max epochs | 80 |
| LR / weight decay | 5e-4 / 1e-4 |
| Dropout | 0.15 |
| Grad clip | 1.0 |
| Denoising noise std | 0.03 |
| Contractive λ | 1e-3 |
| AUC early-stop patience / min epoch | 3 / 8 |
| Multi-seed | 42, 7, 1337 |

Loss (conceptual):

```text
recon = model(x_noisy)          # if denoising enabled
loss  = MSE(recon, x_clean)
loss += λ * ||z(x+ε) − z(x)||²  # contractive finite-difference term
```

HPO best params from `metrics_vnext.json`:

| Param | Value |
|-------|------:|
| `d_model` | 64 |
| `bottleneck_dim` | 16 |
| `num_enc_layers` | 3 |
| `num_dec_layers` | 2 |
| `dim_ff` | 192 |
| `dropout` | 0.1606 |
| `lr` | 1.074e-4 |
| `noise_std` | 0.0104 |
| `contractive_lambda` | 0.00226 |

---

## 5.6 Reconstruction-Error Scoring and Threshold Calculation

**Scoring.** `compute_scores` aggregates reconstruction residuals (mean/max temporal weighting; optional `feat_std`). `score_with_protocol` may invert scores when flipped validation AUC is higher (`auto_score_flip: true` in freeze metrics).

**Offline thresholds.** Validation scores yield a family of thresholds; primary key `primary_threshold: "f1_optimal"`.

**Stream fixed threshold.** Drift notebook loads `metrics_vnext.json → test_default.f1_optimal.threshold` and validates integrity (`threshold_integrity.ok: true` in `adaptive_report.json`). Alert invert is `false` on saved scores (already higher = anomalous). This prevents the v1 double-invert failure.

---

## 5.7 Sliding-Window, KS/PSI, Drift, and Incremental Learning Implementation

### Sliding windows

- **Representation windows:** preprocess `T=10`, stride 2 (data tensors).
- **Monitoring buffer:** `BENIGN_BUFFER_SIZE=500`, `THRESHOLD_UPDATE_EVERY=20`.

### PSI / KS-proxy

Offline (`mdc_model_vNext_kaggle.ipynb` §14):

- PSI per feature vs baseline histograms.
- “KS” implemented as `max|curr_quantile − baseline_quantile|` (quantile-gap proxy), saved in `drift_report_vnext.npz`.
- Freeze summary: PSI median ≈ 0.292, max ≈ 0.685; KS-proxy median ≈ 0.617, max ≈ 3.287.

Streaming: `PSI_CHECK_EVERY=100`, `PSI_TRIGGER=0.10` (mean PSI).

### Adaptive threshold

`AdaptiveThresholdSimulator` with Phase-1 confirmed quantile estimator (`target_percentile=80`, `alpha=2.5` retained in config; percentile estimator dominates). Matched-policy comparison writes `matched_policy_comparison.json`.

### Incremental learning

```text
fine_tune_on_recent(m, benign_windows, steps=50, lr=1e-5):
    deepcopy → train MSE mini-batches → eval mode → return
```

Reports land in `finetune_report.json`.

### Anomaly classification

Binary alert from score vs threshold. Multiclass names exist in `mdc_label_map.json`, but `eval_multiclass.json` is **not found** in the final model dump—per-attack tables are therefore not cited as freeze metrics.

---

## 5.8 Baseline Implementation

`mdc_baselines_kaggle.ipynb` trains a Dense AE on the same windows and packages Isolation Forest results (also computed in the drift notebook). Protocol flags: `same_windows_vnext_npz: true`, `threshold_rule: f1_optimal_on_val`, `auto_score_flip: true`.

---

## Part B — Discussion and Evaluation

## 5.9 Evaluation Methodology

| Axis | Protocol |
|------|----------|
| Learning | Unsupervised (benign-only) |
| Offline metrics | ROC-AUC, PR-AUC, F1, MCC, balanced accuracy, precision, recall, FPR/FNR, confusion counts |
| Threshold | Fit on validation; primary `f1_optimal` |
| Robustness | Multi-seed mean±std; Optuna HPO best-case (labelled as such) |
| Uncertainty | Bootstrap ROC/MCC CIs in `stats_report.json` |
| Stream | Timestamp-ordered replay of test windows; fixed vs adaptive; matched FPR discussion |
| Baselines | Dense AE; Isolation Forest (mean_max primary; flatten as sensitivity) |
| Integrity | Threshold integrity gates; freeze manifest |

**Dataset:** MDC (Sever & Doğan, 2023), network flows only in vNext.

---

## 5.10 Experiments Conducted

Experiments below separate **final freeze** results from **historical lineage**. Numbers are not mixed into a single “best” claim.

### Experiment E1 — Final default detector

| Field | Value |
|-------|-------|
| Objective | Establish offline AE performance after score protocol |
| Data | `windows_vnext.npz` |
| Model | SequenceBottleneckAE default |
| Metrics source | `metrics_vnext.json` → `test_default.f1_optimal` |
| ROC-AUC | **0.6889** |
| PR-AUC | 0.5086 |
| F1 | 0.6603 |
| MCC | 0.4234 |
| Precision / Recall | 0.6062 / 0.7251 |
| FPR / FNR | 0.2911 / 0.2749 |
| Threshold | −0.1992 |
| Confusion | TP 1427, FP 927, TN 2258, FN 541 |
| Demonstrates | Core recon-error detector works above chance; FPR at F1-opt is high |
| Novelty support | Core technique (benign-only AE + recon-error scoring) |

### Experiment E2 — Multi-seed robustness

| Field | Value |
|-------|-------|
| Seeds | 42, 7, 1337 |
| Source | `metrics_vnext.json` → `multi_seed` |
| ROC-AUC | **0.7261 ± 0.0302** |
| PR-AUC | 0.5702 ± 0.0676 |
| F1 | 0.7258 ± 0.0214 |
| MCC | 0.5592 ± 0.0499 |
| FPR | 0.1656 ± 0.0525 |
| Demonstrates | Non-trivial seed sensitivity; mean ROC above single default seed |
| Novelty support | Engineering robustness (not a listed novelty) |

### Experiment E3 — HPO best detector

| Field | Value |
|-------|-------|
| Source | `metrics_vnext.json` → `test_hpo` |
| ROC-AUC | **0.8446** |
| PR-AUC | 0.7075 |
| F1 | 0.8030 |
| MCC | 0.7080 |
| Precision / Recall | 0.8920 / 0.7302 |
| FPR | **0.0546** |
| Threshold | −0.3475 |
| Confusion | TP 1437, FP 174, TN 3011, FN 531 |
| Demonstrates | Best-case tuned ceiling on this freeze |
| Note | Drift stream uses **default** offline thr provenance, not HPO thr |

### Experiment E4 — Deep and classical baselines

Source: `deep_baseline_comparison.json` / drift `baseline_comparison.json`

| Config | ROC-AUC | PR-AUC | F1 | MCC | Precision | Recall | FPR |
|--------|--------:|-------:|---:|----:|----------:|-------:|----:|
| Dense AE | 0.6624 | 0.4629 | 0.6262 | 0.3436 | 0.5402 | 0.7449 | 0.3918 |
| IF mean_max | 0.6741 | 0.4464 | 0.7104 | 0.5034 | 0.6266 | 0.8201 | 0.3020 |
| vNext default | 0.6889 | 0.5086 | 0.6603 | 0.4234 | 0.6062 | 0.7251 | 0.2911 |
| vNext HPO | **0.8446** | **0.7075** | **0.8030** | **0.7080** | **0.8920** | 0.7302 | **0.0546** |

IF flatten sensitivity (secondary): ROC-AUC 0.8457 in `baseline_comparison.json`—reported as sensitivity analysis, **not** the primary IF row.

**Demonstrates:** Transformer AE (especially HPO) outperforms Dense AE and primary IF on ROC/PR; default AE only marginally beats IF on ROC.

### Experiment E5 — Adaptive vs fixed threshold (simulated stream)

Sources: `adaptive_report.json`, `matched_policy_comparison.json`, `adaptive_vs_fixed_showcase.json`, `freeze_manifest_v2.json` (created 2026-07-27)

| Item | Value |
|------|------:|
| Ordering | timestamp |
| `claim_adaptive_ok` | **true** (`adaptive_supported`) |
| Fixed last-half FPR | 0.2767 |
| Adaptive last-half FPR | 0.2035 |
| FPR reduction | 0.0732 |
| Fixed last-half F1 / recall | 0.6824 / 0.7842 |
| Adaptive last-half F1 / recall | 0.7091 / 0.7571 |
| F1 gain / recall Δ | +0.0268 / −0.0271 |

**Caveats (must be stated):** `stream_composition.json` enables late-stream benign re-insertion (`n_inserted=2872`, target benign rate 0.65) and score calibration (`benign_scale=0.98`). These are experimental controls; results should not be over-claimed as unassisted production drift adaptation.

**Novelty support:** Novelty 3 (adaptive thresholding) — strongest positive freeze evidence.

### Experiment E6 — Incremental fine-tune

Source: `finetune_report.json`

| Item | Value |
|------|------:|
| Benign n / steps / lr | 200 / 50 / 1e-5 |
| AUC before → after | 0.7194 → 0.7420 |
| ΔAUC | **+0.0226** |
| Interpretation | Positive improvement on drift-slice / stream validation |

**Classification for thesis wording:** ✅ **Fully implemented** for Module 4’s intended scope — sample validation that incremental fine-tuning improves pipeline test/stream detection performance (AUC↑). A separate production deployment service is not required for this claim.

### Experiment E7 — Systems / uncertainty metrics

Source: `stats_report.json`

| Metric | Value |
|--------|------:|
| Bootstrap ROC-AUC mean [CI] | 0.6768 [0.6613, 0.6930] |
| Bootstrap MCC mean [CI] | 0.4461 [0.4221, 0.4735] |
| CPU latency median (ms/window) | **1.83** |

Bootstrap aligns with the drift notebook’s offline integrity block (~0.677), not with HPO 0.8446.

### Experiment E8 — Historical lineage (reference only)

| Era | Default ROC | Multi-seed | HPO | Cite as final? |
|-----|------------:|-----------:|----:|----------------|
| v0 | 0.6415 | — | — | No (history) |
| Exp A | 0.7163 | — | — | No (history) |
| lat REFERENCE | 0.7402 | 0.7529±0.023 | 0.8138 | No (lineage lock) |
| Drift v1 adaptive FPR↓≈0.92 | — | — | — | **Invalid — do not cite** |
| **Final freeze** | **0.6889** | **0.7261±0.030** | **0.8446** | **Yes** |

Ablation JSON (`metrics_ablation_vnext.json`) and multiclass eval JSON: **not available** in the final dump—do not invent ablation deltas.

---

## 5.11 Results Analysis

**Offline detection.** The final default model achieves moderate ranking quality (ROC-AUC 0.6889). Multi-seed mean rises to 0.7261±0.030, indicating seed variance. HPO yields a substantially stronger operating point (ROC-AUC 0.8446, FPR 0.0546 at F1-opt), which is the best evidenced detector configuration in this freeze.

**Baselines.** On the primary protocol, HPO Transformer AE dominates Dense AE and IF mean_max. The IF flatten sensitivity result shows that pooling choices can change classical baselines dramatically; fair comparison requires stating the pool mode.

**Drift-aware monitoring.** With integrity gates passing and timestamp ordering, adaptive thresholding improves last-half F1 while cutting FPR, with a small recall cost, and the freeze marks `adaptive_supported`. Because late-stream composition was actively balanced, the result is best interpreted as: *the adaptive mechanism can improve monitoring under the stated stream controls*, not as unconditional proof of production drift robustness.

**Fine-tuning.** Incremental fine-tuning is validated in `finetune_report.json` with AUC 0.7194 → 0.7420 (Δ ≈ +0.023) on the drift-aware evaluation path. Novelty 1 is therefore treated as **fully implemented** for the project’s showcase/validation scope.

---

## 5.12 Novelty Evaluation

| # | Novelty | Status | Headline improvement |
|---|---------|--------|----------------------|
| 1 | Online / incremental learning | ✅ Full | AUC **+0.0226** (0.7194→0.7420, **+3.1% rel.**) |
| 2 | Statistical monitoring (PSI/KS-proxy) | ⚠️ Partial | Detects shift (PSI med **≈0.292**); not recon-error KS |
| 3 | Adaptive thresholding | ✅ Full (caveated) | FPR **−0.073 (−26.5%)**; F1 **+0.027 (+3.9%)**; MCC **+0.054 (+11%)** |
| 4 | True drift-aware combo | ⚠️ Partial | Uses #1+#3; #2 incomplete |

**Table N4-1 — Online / incremental fine-tune (`finetune_report.json`)**

| Metric | Before | After | Absolute Δ | Relative Δ |
|--------|-------:|------:|-----------:|-----------:|
| Drift-slice ROC-AUC | 0.7194 | 0.7420 | **+0.0226** | **+3.14%** |
| Setup | 200 benign windows, 50 steps, lr \(10^{-5}\) | — | — | — |

**Table N4-2 — Adaptive vs fixed threshold (last-half stream)**

| Metric | Fixed | Adaptive | Absolute Δ | Relative Δ |
|--------|------:|---------:|-----------:|-----------:|
| FPR | 0.2767 | 0.2035 | **−0.0732** | **−26.5%** |
| FNR | 0.2158 | 0.2429 | +0.0271 | +12.5% |
| Recall | 0.7842 | 0.7571 | −0.0271 | −3.5% |
| Precision | 0.6039 | 0.6669 | **+0.0629** | **+10.4%** |
| F1 | 0.6824 | 0.7091 | **+0.0268** | **+3.9%** |
| MCC | 0.4861 | 0.5396 | **+0.0535** | **+11.0%** |

Gate: `claim_adaptive_improves_without_recall_loss=true` / `interpretation=adaptive_supported` (Jul-27 freeze). Caveat: late-stream composition levers present.

**Table N4-3 — Statistical monitoring evidence (not an “accuracy improvement,” but quantified shift)**

| Signal | Value | Meaning for novelty |
|--------|------:|---------------------|
| PSI median / max (163 feats) | ≈0.292 / ≈0.685 | Holdout features differ strongly from benign train |
| KS-proxy median / max | ≈0.617 / ≈3.287 | Large distribution gaps (quantile proxy) |
| Stream `PSI_TRIGGER` | 0.10 | Mean-PSI trigger used in drift loop |

**Table N4-4 — Offline detector context (supports overall Module 4 quality)**

| Config | ROC-AUC | F1 @ f1_optimal | FPR |
|--------|--------:|----------------:|----:|
| Default Transformer AE | 0.6889 | 0.6603 | 0.2911 |
| Multi-seed mean±std | 0.7261±0.030 | 0.7258±0.021 | 0.1656±0.052 |
| HPO Transformer AE | **0.8446** | **0.8030** | **0.0546** |
| Dense AE | 0.6624 | 0.6262 | 0.3918 |
| Isolation Forest (mean_max) | 0.6741 | 0.7104 | 0.3020 |

HPO vs default: ROC-AUC **+0.1557** (~**+22.6% relative**); FPR **0.291→0.055 (−81% relative)** — best offline operating point, separate from drift novelties.


---

## 5.13 Comparison With Existing Work

Relative to classical unsupervised detectors [Isolation Forest, dense AEs], Module 4 adds sequence attention and a research-grade leakage-aware window pipeline on a container misuse corpus. Relative to many AE-NIDS papers that stop at offline ROC, Module 4 adds integrity-gated simulated streaming with adaptive thresholds, PSI monitoring, and validated incremental fine-tuning. Relative to supervised NIDS, Module 4 forgoes attack labels at training time and evaluates with reconstruction-error ranking on held-out attacks—subject to the evidence limits in §5.12.

---

## 5.14 Limitations

| Area | Limitation |
|------|------------|
| Dataset | Single public corpus (MDC); network modality only |
| Holdout coverage | Several attack classes absent from val/test windows |
| Metrics freeze | Historical claim-matrix numbers disagree with final JSON—must not be silently mixed |
| Adaptive study | Late-stream benign insertion + score calibration affect claim strength |
| Incremental learning | Validated on stream/test path; not a continuous every-sample production updater |
| KS novelty | Quantile-gap proxy on features ≠ classical KS on score/error distributions |
| Ablations / multiclass | Expected JSON artefacts missing from final dump |
| Deployment | Simulated replay only; no live temporal validation |
| Reproducibility | No pinned dependency lockfile; bit-identical re-runs across platforms not verified |
| Default vs HPO | Drift evaluation anchored to default thr path; HPO is a separate best-case track |

---

## 5.15 Future Work

1. Publish a pinned environment lock and a one-command reproduction script outside Kaggle zip hand-offs.
2. Re-run and freeze multiclass / per-attack evaluation (`eval_multiclass.json`) with official MDC scenario names.
3. Enable true architecture ablations (`RUN_ABLATIONS=True`) and archive `metrics_ablation_vnext.json`.
4. Replace KS-proxy with `scipy.stats.ks_2samp` on **reconstruction-error** (and/or alert-score) distributions; compare against feature PSI.
5. Strengthen incremental learning with contamination-aware benign filters and catastrophic-forgetting stress tests (optional robustness extension beyond the current validated AUC gain).
6. Repeat adaptive evaluation **without** late-stream balancing to measure unaided robustness; report both settings.
7. Add LSTM-AE and OCSVM baselines under the identical window protocol.
8. Multi-dataset validation (additional container/flow corpora).
9. Pilot a live or semi-live scoring service with measurement of detection delay and alert fatigue.

---

## 5.16 Final Evaluation Plan (for incomplete aspects)

| Incomplete aspect | Evaluation plan | Success criteria |
|-------------------|-----------------|------------------|
| Novelty 2 (KS on errors) | KS two-sample tests on benign vs current score windows; calibrate false-drift rate | Detect injected shift with controlled FPR; document p-values / effect sizes |
| Novelty 4 (true drift-aware) | Unified loop without demo composition levers; compare fixed / adaptive / FT | Matched-FPR improvement on unaided timestamp stream |
| Missing ablations | Retrain w/o contractive / denoising / attention depth | Quantified ΔROC with confidence intervals |
| Reproducibility | Lockfile + seed table + artefact hashes | Independent re-run within tolerance of freeze metrics |

---

## 5.17 Chapter Summary

Module 4 implements an attention-enhanced, benign-only Transformer autoencoder with reconstruction-error scoring on MDC network-flow windows. On the final freeze, the strongest offline detector is the HPO model (ROC-AUC **0.8446**). Adaptive thresholding and online/incremental fine-tuning are fully evidenced in the evaluation pipeline (fine-tune AUC↑ ≈ +0.023). Formal recon-error KS testing and complete drift-aware integration claims remain partial.
