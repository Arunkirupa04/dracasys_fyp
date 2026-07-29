# Chapter 4 — Approach, Analysis and Design

**Module:** Module 4 — MDC vNext  
**Evidence basis:** `module4/notebook/final/` (kaggle-source + output-metrics)  
**Companion chapters:** Technologies (Ch. 3); Implementation & Evaluation (Ch. 5)

---

## Part A — Your Approach

## 4.1 Overall Approach

Module 4 solves container misuse detection as an **unsupervised sequence anomaly detection** problem with an optional **drift-aware monitoring** layer. The approach is:

1. Learn a model of *benign* container network behaviour from leakage-safe windows.
2. Score every window by reconstruction error through an attention-enhanced bottleneck autoencoder.
3. Convert scores to alerts using a validation-tuned offline threshold.
4. Under simulated streaming replay, adapt the threshold from a rolling benign buffer, monitor feature drift (PSI), and apply incremental fine-tuning with validated AUC improvement on the evaluation stream.

This matches the implemented vNext pipeline. It does **not** assume multi-modal CPU/memory/syscall inputs—the final artefacts use **163 CIC-style network features** per 15 s bucket.

### Proposed versus implemented (core technique)

| Aspect | Proposed | Implemented evidence |
|--------|----------|----------------------|
| Attention-enhanced AE | Yes | `SequenceBottleneckAE` with Transformer encoder/decoder layers |
| Train on benign only | Yes | `X_train` benign-only; no `y_train` in NPZ |
| Reconstruction error as primary signal | Yes | `compute_scores` / MSE-based scoring |
| Multi-modal telemetry | Mentioned in older proposal text | **Not found** in final `feat_names` / NPZ |

**Verdict:** The proposed core technique is **implemented** for network-flow windows.

---

## 4.2 Users, Inputs, and Outputs

### Users

- **Researcher:** runs the five Kaggle-source notebooks in order and inspects freeze JSON/figures.
- **Integrator:** loads `model_hpo_best_vnext.pt` with `SequenceBottleneckAE` for demo inference.
- **Analyst (intended):** consumes score/alert streams (via integration UI, outside `module4/` package scope).

### Inputs

| Stage | Input | Shape / format |
|-------|-------|----------------|
| Preprocess | Raw MDC CSV | Tabular CIC-style flows |
| Model / Drift / Baselines | `windows_vnext.npz` | `X_*`: `(N, 10, 163)` |
| Drift | `checkpoint_vnext.pt`, `scores_vnext.npz`, `metrics_vnext.json` | PyTorch + NumPy + JSON |
| Drift (optional) | `drift_baseline_vnext.npz` | Quantiles / stats per feature |

### Outputs

| Stage | Output | Consumer |
|-------|--------|----------|
| Preprocess | Windows, manifest, label map, drift baseline | All downstream notebooks |
| Model | Checkpoint, scores, `metrics_vnext.json`, PSI/KS report | Drift, baselines, analysis |
| Drift | Adaptive reports, stream log, freeze manifest, figures | Thesis evaluation |
| Baselines | Dense AE + comparison table | Results discussion |

---

## 4.3 Processing Workflow

```text
[1] Load MDC flows
[2] Filter containers; sessionise (60 s gap)
[3] Per-container session split 70% train / 30% holdout
[4] Hygiene → variance/correlation filters → clip bounds
[5] Flow StandardScaler (fit benign train) + clip
[6] 15 s bucketing (mean/max/std + flow_count)
[7] Bucket StandardScaler + clip (±10)
[8] Sliding windows T=10, stride=2; attack-frac ≥ 0.5 labels
[9] Holdout → stratified val/test 50/50
[10] Train SequenceBottleneckAE on benign X_train
[11] Score val/test; auto score-flip; thresholds on val
[12] Evaluate offline (default, multi-seed, HPO)
[13] Replay test windows in timestamp order
[14] Fixed thr vs adaptive thr; PSI checks; optional fine-tune
[15] Compare Isolation Forest / Dense AE baselines
```

Steps 1–9 are preprocess; 10–12 are the core detector; 13–14 are the drift-aware layer; 15 is comparative evaluation.

---

## 4.4 Model Architecture

The detector is a **sequence bottleneck Transformer autoencoder**:

```text
Input x ∈ R^{B×10×163}
    → Linear to d_model + sinusoidal PE
    → TransformerEncoder (L_enc layers, H heads)
    → Bottleneck Linear (d_model → bn) + LN + GELU
    → Upsample (bn → d_model)
    → TransformerDecoder (L_dec layers) with learned position queries
    → Linear to 163
    → Reconstruction x̂
```

**Default config (model notebook):** `d_model=64`, `nhead=4`, `bottleneck_dim=24`, encoder/decoder 2/2, `dim_ff=256`, dropout 0.15.  
**HPO best (freeze):** `bottleneck_dim=16`, enc/dec 3/2, `dim_ff=192`, dropout ≈ 0.161, LR ≈ 1.07e-4, noise_std ≈ 0.010, contractive_λ ≈ 0.00226.

Attention is provided by the Transformer layers (multi-head self-/cross-attention inside PyTorch encoder/decoder layers).

---

## 4.5 Training Process

| Item | Design choice |
|------|----------------|
| Data | Benign windows only |
| Loss | MSE(`x̂`, `x`) + optional denoising on input + contractive term on bottleneck |
| Optimiser | AdamW; grad clip 1.0 |
| Early stopping | Validation AUC patience (default path) |
| Checkpointing | AUC-best weights retained |
| Robustness | Multi-seed `{42, 7, 1337}`; Optuna 12 trials for HPO path |

Training deliberately avoids teaching the model to reconstruct attacks, which historically caused “attack reconstruction creep” and AUC decay in earlier Exp A peak-then-decay behaviour.

---

## 4.6 Detection Process and Reconstruction-Error Calculation

1. Forward pass → `x̂`.
2. Per-timestep / per-feature errors aggregated with mean and max temporal pooling (weighted).
3. Optional feature weighting via `feat_std` estimated on benign validation errors when an AUC gate passes.
4. Auto score-flip: compare validation ROC-AUC with and without negating scores; keep the better orientation.
5. Alerts: `score >= threshold` in alert space (higher = more anomalous).

Offline primary threshold: validation **`f1_optimal`**, then applied to test scores. Stream evaluation loads this threshold from `metrics_vnext.json` rather than recomputing under a conflicting invert convention (the v1 failure mode).

---

## 4.7 Thresholding Mechanisms

| Mechanism | Where | Role |
|-----------|-------|------|
| `youden_j`, `p95_benign`, `p99_benign`, `f1_optimal`, `fbeta_0.5` | Model notebook | Offline operating-point family |
| Fixed stream threshold | Drift notebook | Frozen offline `f1_optimal` thr |
| Adaptive threshold | `AdaptiveThresholdSimulator` | Rolling benign buffer (size 500), update every 20 steps, quantile estimator (pctl 80 in confirmed Phase-1) |

Adaptive updates are the implementation of proposed Novelty 3.

---

## 4.8 Drift Detection, Incremental Learning, and Sliding Windows

### Sliding windows (data representation)

Preprocess sliding windows (`T=10`, stride 2) create the model’s examples. Separately, the drift layer uses a **sliding benign score buffer** (capacity 500) for threshold adaptation—two different “sliding window” concepts serving representation vs monitoring.

### Drift detection

- Offline: PSI + KS-proxy per feature vs `drift_baseline_vnext.npz`.
- Online (simulated): mean PSI every 100 steps; trigger level 0.10.
- Stream periods: first ~70% labelled stable analysis slice; last ~30% drift analysis slice (with optional late-stream benign balancing).

### Incremental learning

`fine_tune_on_recent` performs short MSE fine-tuning on recent benign windows when triggered/configured. It is integrated into the orchestration loop and validated with a positive AUC delta on the evaluation stream (Chapter 5).

---

## 4.9 Interaction Between Components

```text
Preprocess ──windows──► Model ──scores/ckpt/metrics──► Drift-aware
                │                      │
                │                      ├── fixed threshold path
                │                      ├── adaptive threshold path
                │                      ├── PSI monitor
                │                      └── optional fine-tune
                │
                └──────────► Baselines (Dense AE, IF on same windows)
                                   │
Analysis ←──── all artefacts ──────┘
```

No module inside Module 4 feeds another project module (M1–M3); Module 4 is an independent detector. Integration dashboards may display M4 scores alongside other modules without pipelining.

---

## Part B — Analysis and Design

## 4.10 System Architecture

**Figure 4.1: Module 4 Drift-Aware Anomaly Detection Architecture**

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Module 4 (MDC vNext)                      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐ │
│  │ Preprocess   │    │ Detector         │    │ Drift-Aware    │ │
│  │              │    │ SequenceBottleneck│    │ Monitor        │ │
│  │ CSV→sessions │───►│ AE (Transformer) │───►│ Stream replay  │ │
│  │ scale/bucket │    │ Recon-error score│    │ Adaptive thr   │ │
│  │ windows T=10 │    │ Val thresholds   │    │ PSI / FT       │ │
│  └──────────────┘    └────────┬─────────┘    └───────┬────────┘ │
│                               │                      │          │
│                               ▼                      ▼          │
│                     metrics / checkpoint      adaptive_report   │
│                     scores_vnext.npz          freeze_manifest   │
│                                                                  │
│  ┌──────────────┐                                               │
│  │ Baselines    │◄── same windows_vnext.npz                     │
│  │ Dense AE, IF │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

Figure 4.1 shows the major components and artefact hand-offs. Preprocess produces windows consumed by the detector and baselines; the detector’s checkpoint and scores feed the drift-aware monitor.

---

## 4.11 Data-Flow Design

**Figure 4.2: End-to-End Data Flow**

```text
MDC CSV
  → sessions (gap 60s)
  → train sessions / holdout sessions
  → scaled buckets (15s, 163 feats)
  → windows_vnext.npz
        ├─ X_train (benign) → AE training
        ├─ X_val, y_val → score protocol + thresholds
        └─ X_test, y_test, ts_test → offline test + timestamp stream
```

Figure 4.2 emphasises leakage control: scalers and clip bounds are fit on train benign data; holdout windows are transformed only.

---

## 4.12 Model Architecture Diagram

**Figure 4.3: SequenceBottleneckAE Internal Structure**

```text
x (B,T,F)
   │
   ▼
Input proj + Sinusoidal PE
   │
   ▼
Transformer Encoder × L_enc   ← multi-head self-attention
   │
   ▼
Bottleneck (B,T,bn)
   │
   ▼
Upsample to d_model
   │
   ▼
Transformer Decoder × L_dec   ← attention over memory + pos queries
   │
   ▼
Output proj → x̂ (B,T,F)
   │
   ▼
‖x − x̂‖ → anomaly score
```

Figure 4.3 is the design counterpart of `SequenceBottleneckAE` in the model notebook / `sequence_bottleneck_ae.py`.

---

## 4.13 Drift-Aware Workflow Design

**Figure 4.4: Simulated Streaming Orchestration**

```text
For each window in timestamp order:
  1. Obtain score (saved scores or model forward)
  2. Compare with fixed thr → fixed alert
  3. If presumed benign: push score into buffer (size ≤ 500)
  4. Every 20 steps: refresh adaptive thr from buffer quantile estimator
  5. Compare score with adaptive thr → adaptive alert
  6. Every 100 steps: compute mean PSI vs baseline; record drift_active
  7. Optionally fine-tune on recent benign when configured
  8. Append row to drift_aware_log
```

Figure 4.4 documents the control loop realised by `drift_aware_eval` in `mdc_drift_aware_kaggle.ipynb`. Ordering mode in the final freeze is **`timestamp`** (`n_windows=5153`). This is **simulated streaming replay**, not live deployment.

---

## 4.14 Algorithms

### Algorithm 4.1 — Offline scoring and thresholding

```text
Input: trained model M, X_val, y_val, X_test, y_test
s_val_raw, s_test_raw ← compute_scores(..., invert=False)
if AUTO_SCORE_FLIP and AUC(−s_val_raw) > AUC(s_val_raw):
    invert ← True
s_val, s_test ← compute_scores(..., invert)
thr ← argmax_t F1(y_val, s_val ≥ t)   # f1_optimal
metrics ← evaluate(y_test, s_test, thr)
Output: thr, metrics, s_val, s_test
```

### Algorithm 4.2 — Adaptive threshold update (alert space)

```text
Input: benign_buffer B, target_percentile p, envelope constraints
estimate ← quantile(B, p)   # Phase-1 confirmed estimator
thr_adaptive ← apply_envelope(estimate, thr_fixed, δ)
Output: thr_adaptive
```

### Algorithm 4.3 — Incremental fine-tune (as implemented)

```text
Input: model M, benign_windows X_b, steps S, lr η
M' ← deepcopy(M)
for s in 1..S:
    take one mini-batch from X_b
    minimise MSE(M'(x), x) with AdamW(η)
Output: M'
```

---

## 4.15 Component Responsibility Table

| Component | Responsibility | Key artefacts |
|-----------|----------------|---------------|
| Preprocess | Leakage-safe windows + drift baseline | `windows_vnext.npz`, `manifest_vnext.json` |
| Detector | Benign AE train, score, HPO | `checkpoint_vnext.pt`, `metrics_vnext.json` |
| Drift monitor | Stream policies + gates | `adaptive_report.json`, `freeze_manifest_v2.json` |
| Baselines | Comparative unsupervised detectors | `deep_baseline_comparison.json` |
| Analysis | Synthesis plots/tables | `mdc_analysis_outputs/` |

---

## 4.16 Novelty Verification Against Design

| # | Novelty | Status | Improvement evidence (metrics) |
|---|---------|--------|--------------------------------|
| 1 | Online / incremental learning | ✅ | AUC **0.7194 → 0.7420** (**+0.0226**, ~**+3.1%**) on drift-slice validation |
| 2 | Statistical monitoring (PSI/KS-proxy) | ⚠️ | PSI median **≈0.292** (major-shift regime); not formal recon-error KS |
| 3 | Adaptive thresholding | ✅ | Last-half FPR **−26.5% rel.** (0.277→0.204); F1 **+3.9% rel.** (0.682→0.709); MCC **+11.0% rel.** |
| 4 | True drift-aware pipeline | ⚠️ | Integrates #1+#3 gains in one simulator; #2 still partial |

Full numeric tables are in Chapter 5 §5.12.


---

## 4.17 Design Constraints and Non-Goals

| Constraint / non-goal | Rationale from repository |
|-----------------------|---------------------------|
| Notebook-only research system | No `module4/src` package found |
| Simulated stream only | Explicit notes in drift artefacts |
| Network flows only | Final feature set is CIC-style |
| Do not cite v1 “~92% FPR reduction” | Documented threshold double-invert bug |
| Do not mix metric eras | Final freeze ≠ historical claim-matrix lock (0.7402 / 0.8138) |

---

## 4.18 Summary

The Module 4 approach combines a Transformer bottleneck autoencoder, reconstruction-error scoring, and a drift-aware monitoring layer. Figures 4.1–4.4 define the architecture and workflows that Chapter 5 implements and evaluates. Novelty claims are intentionally stratified: online/incremental fine-tuning and adaptive thresholding are fully evidenced in the evaluation pipeline; KS-on-reconstruction-error monitoring and full drift-aware integration claims remain partial.
