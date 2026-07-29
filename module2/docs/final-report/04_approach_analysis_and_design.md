# Chapter 4 — Approach, Analysis and Design

**Module:** Module 2 — Drift-Aware Short-Term Resource Prediction  
**Evidence basis:** `module2/final_notebook/`  
**Companion chapters:** Technologies (Ch. 3); Implementation & Evaluation (Ch. 5)

---

## Part A — Your Approach

## 4.1 Overall Approach

Module 2 solves **short-term multivariate regression** of container CPU and memory. The approach is:

1. Convert raw per-metric CSVs into per-container aligned series.
2. Inject synthetic bursts (for stress evaluation) while retaining a clean variant.
3. Normalize with train-only statistics and engineer 27 features.
4. Build variability-driven lookback windows in [500, 1000].
5. Train a residual GRU per horizon so predictions start from persistence.
6. Evaluate statically against Persistence and SES under clean/injected/spike/normal regimes.
7. Replay the injected test stream with EWMA/z-score drift detection and error-triggered fine-tuning.

This is the **as-built** approach. The final pipeline forecasts CPU and memory only.

---



## 4.2 Inputs and Outputs



### Inputs


| Stage      | Input                                                           |
| ---------- | --------------------------------------------------------------- |
| Phase 1    | AIOpsArena `complex_case1` container KPI CSVs                   |
| Model      | `(B, L, 27)` float32 windows, L\in[500,1000]                    |
| Drift loop | Chronological injected test windows + frozen static checkpoints |




### Outputs


| Stage     | Output                                           |
| --------- | ------------------------------------------------ |
| Phase 1   | Features, windows, masks, manifest, spike events |
| Phase 3   | `gru_h{1,2,3}_static.pt`, `all_metrics.json`     |
| Phase 4   | Static vs adaptive stream MAPE, debug log, plots |
| Inference | 4 de-normalized forecasts per chosen horizon     |


---



## 4.3 Data Flow and Preprocessing

```text
Long-format KPI rows
  → pivot (timestamp, cmdb_id, case) to 7 wide metrics
  → artifact check on natural step-changes
  → per-container chronological 70/15/15 + 10-row embargo
  → synthetic burst injection (train/val/test seeds 101/202/303)
  → train-only z-score (stats from injected train)
  → lag diffs (1,2,3) + rolling mean/std(3) on targets → 27 features
  → adaptive lookback windows [anchor_end, L]
```

Split membership is decided by **target row**, not window start, so lookback may cross a split boundary as context while the prediction target does not leak across splits (embargo protects shared targets).

---



## 4.4 Feature Preparation


| Block           | Columns                                                                         |
| --------------- | ------------------------------------------------------------------------------- |
| Raw (7)         | cpu_usage, cpu_system, cpu_user, mem_usage, mem_working_set, mem_rss, mem_cache |
| Engineered (20) | For each of 4 targets: DIFF_1/2/3, ROLLING_MEAN_3, ROLLING_STD_3                |
| Targets (4)     | Indices `[0, 3, 4, 5]` inside the 27-vector                                     |


Exact names are frozen in `feature_cols.json`.

---



## 4.5 GRU Model Architecture

**Figure 4.3** (detailed below) summarises:

```text
Packed sequence (B, L, 27)
  → 2-layer GRU (hidden 128, dropout 0.2)
  → h_n[-1]
  → Dropout → Linear(128→64) → ReLU → Linear(64→4)
  → + last observed targets
  → ŷ (B, 4) in normalized space
```

Training uses MSE in normalized space; reported metrics de-normalize with frozen mean/std.

---



## 4.6 Training and Forecasting Process


| Item              | Design                     |
| ----------------- | -------------------------- |
| Horizons          | 1, 2, 3 steps (15/30/45 s) |
| Optimiser         | Adam lr=0.001              |
| Epochs / patience | 30 max / early stop 7      |
| Batch             | 128 train / 512 eval       |
| Checkpoint        | Best validation MSE        |


Forecasting: load horizon-specific weights → forward → inverse z-score per target.

---



## 4.7 Adaptive Sliding-Window Mechanism

Lookback L is set once per window from train-period CPU variability (Chapter 3 formula). Length tables feed every later phase; **51.6%** of train windows are shortened below 1000, showing the rule is active. Phase 4 reuses these lengths rather than recomputing them each chunk.

---

## 4.8 Error Monitoring, Drift Detection, Incremental Learning

**Figure 4.4** orchestration — the detailed drift-aware novelty loop:

```text
For each chronological chunk of test windows:
  predict with current model
  compute chunk error
  AdaptiveThreshold.update (confidence band log)
  if DriftMonitor.update(error) == True:
      model ← OnlineAdapter.adapt(model, seen_windows)
  log metrics / checksums
```

| Drift-aware approach | Implementation | Validation role |
|----------------------|----------------|-----------------|
| Adaptive lookback windows | Phase 1 length rule \(L\in[500,1000]\) | Variability-aware history for GRU inputs |
| Error monitoring | `DriftMonitor` EWMA + z-score | Detects sustained error rise on stream |
| Online / incremental learning | `OnlineAdapter` fine-tune on recent windows | Improves Adaptive vs Static MAPE (Chapter 5) |
| Adaptive confidence band | `AdaptiveThreshold` percentiles | Logged with stream debug trace |

`DriftMonitor` = EWMA + z-score vs warmup reference.  
`OnlineAdapter` = short MSE fine-tune on recent windows when drift is sustained.

For Module 2’s intended scope, this stack is **fully implemented**: Phase 4 provides sample validation that error-triggered retraining improves pipeline stream/test performance. A production autoscaler service is not required for the claim.

---



## 4.9 Burst Handling

Burst handling is a **pipeline protocol**, not a separate network module:

1. Inject controlled events (CPU permanent bump; memory temporary bump).
2. Train primarily on injected trajectories.
3. Score injected overall and spike-masked subsets.
4. Allow drift-triggered updates when burst-driven error exceeds the statistical gate.

---



## 4.10 Multivariate CPU and Memory Outputs

The model jointly outputs four CPU/memory series from a shared GRU state. This is standard multivariate forecasting, not a separate novelty claim in this report.

---



## Part B — Analysis and Design



## 4.11 Top-Level Architecture

**Figure 4.1: Module 2 Drift-Aware Resource Forecasting Architecture**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         Module 2 (as-built)                         │
│                                                                     │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────┐  │
│  │ Phase 1      │   │ Phase 2         │   │ Phase 3              │  │
│  │ Load/split   │──►│ model_defs.py   │──►│ Static train H1–H3   │  │
│  │ Burst inject │   │ AdaptiveGRU     │   │ Persistence / SES    │  │
│  │ Feats+windows│   │ DriftMonitor    │   │ Clean/injected eval  │  │
│  └──────────────┘   │ OnlineAdapter   │   └──────────┬───────────┘  │
│         │           └────────┬────────┘              │              │
│         │                    │                       ▼              │
│         │                    │            ┌──────────────────────┐  │
│         └────────────────────┴───────────►│ Phase 4 Streaming    │  │
│                                           │ Static vs Adaptive   │  │
│                                           │ Drift → Fine-tune    │  │
│                                           └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

Figure 4.1 shows the four-phase hand-off. Phase 2 contributes shared code; Phase 3 produces static checkpoints; Phase 4 evaluates drift-aware behaviour on the injected stream.

---



## 4.12 Data-Flow Design

**Figure 4.2: End-to-End Data Flow**

```text
kpi_*.csv (long)
   → wide 7 metrics (n_rows≈223,830; 27 containers)
   → clean_raw / inj_raw
   → features_clean.npy / features_injected.npy  (N, 27)
   → windows_* .npy  (N_win, 2)=[end, L]
   → WindowDataset → padded batches
   → AdaptiveGRUModel → ŷ
```

Figure 4.2 emphasises parallel clean/injected artefacts used throughout evaluation.

---



## 4.13 Model Architecture Diagram

**Figure 4.3: AdaptiveGRUModel Internal Structure**

```text
x (B,L,27), lengths
        │
        ▼
 pack_padded_sequence
        │
        ▼
 GRU × 2 (h=128)
        │
        ▼
 h_n[-1] → Drop → FC 128→64 → ReLU → FC 64→4
        │
        ▼
 + x[i, L_i-1, {0,3,4,5}]
        │
        ▼
 ŷ_norm (B,4) → de-normalize → real units
```

Figure 4.3 is the design counterpart of `AdaptiveGRUModel` in `model_defs.py`.

---



## 4.14 Drift-Aware Streaming Workflow

**Figure 4.4: Phase-4 Drift Orchestration**

```text
Injected test stream (chronological chunks)
   │
   ├─► Static model path (frozen weights) ──► MAPE static
   │
   └─► Adaptive path
         DriftMonitor (EWMA/z)
              │ trigger
              ▼
         OnlineAdapter fine-tune
              │
              ▼
         MAPE adaptive + debug_log
```

Figure 4.4 isolates the novelty-relevant control loop. `AdaptiveThreshold` runs in parallel for logging only.

---



## 4.15 Algorithms



### Algorithm 4.1 — Adaptive lookback assignment

```text
Input: per-container train rows, cpu_usage series
ref ← median(rolling_std_120(train_cpu))
for each valid anchor end:
    r ← rolling_std_120 at end
    L ← clip(1000 - 250*(r/ref - 1), 500, 1000)
    emit window (end, L)
```



### Algorithm 4.2 — DriftMonitor.update

```text
ewma ← α·err + (1-α)·ewma
if |ref| < warmup: append err; return False
z ← (ewma - mean(ref)) / std(ref)
hits ← hits+1 if z > z_thr else 0
return hits ≥ sustain
```



### Algorithm 4.3 — OnlineAdapter.adapt

```text
recent ← last min(4096, seen) windows
for 1 epoch:
    minimise MSE(model(X), y) with Adam(lr=1e-4)
return model
```

---



## 4.16 Component Responsibility Table


| Component      | Responsibility                   | Artefacts                       |
| -------------- | -------------------------------- | ------------------------------- |
| Phase 1        | Data, bursts, features, windows  | `kagglephase1_output/`          |
| Phase 2        | Shared model/drift/baseline code | `model_defs.py`                 |
| Phase 3        | Train + static metrics           | `all_metrics.json`, checkpoints |
| Phase 4        | Streaming adaptation proof       | `phase4_results.json`           |
| Handover model | Integration inference            | `production_model.pt`           |


---



## 4.17 Novelty Verification Against Design

| # | Novelty | Status | Improvement evidence (metrics) |
|---|---------|--------|--------------------------------|
| 1 | Drift-aware short-term prediction | ✅ | Adaptive overall MAPE: H1 **0.132→0.118 (−10.6% rel.)**, H2 **0.213→0.200 (−6.1% rel.)**, H3 flat at 0.262 |
| 1a | Adaptive sliding windows | ✅ | 51.6% of train windows shortened below L=1000 (rule active) |
| 1b | Error monitoring (EWMA+z) | ✅ | Drift triggers drive updates (e.g. H3 large z before first adapt) |
| 1c | Online / incremental learning | ✅ | **3/9/9** updates; spike MAPE↓ **−0.010 / −0.012 / −0.014** pp at H1/H2/H3 |
| 1d | Adaptive confidence band | ✅ | Logged each chunk with stream debug |
| 2 | Burst-aware forecasting | ✅ (CPU caveat) | Memory MAPE vs Persistence: up to **−37% relative** (H3 mem_usage 0.494→0.311); CPU still near/behind Persistence |

Sources: `phase4_results.json`, `all_metrics.json` / per-target tables (Chapter 5).


---



## 4.18 Design Constraints and Non-Goals


| Constraint                    | Rationale                                        |
| ----------------------------- | ------------------------------------------------ |
| Single case (`complex_case1`) | Cross-case scale mismatch destroyed earlier runs |
| Forecasting metrics only      | Not an anomaly detector (no ROC/F1)              |
| CPU + memory targets          | Final KPI set used in the pipeline               |
| Notebook pipeline             | No `module2/src` package in final path           |
| Modest adaptive gains         | Mechanism real; accuracy lift small              |


---



## 4.19 Summary

Module 2’s approach centres on a residual GRU for short-horizon CPU/memory forecasting, with a detailed drift-aware stack (adaptive windows, EWMA/z monitoring, error-triggered online fine-tuning, confidence-band logging) and synthetic burst evaluation. Figures 4.1–4.4 define the architecture that Chapter 5 implements and measures. Drift-aware / online learning and burst-aware forecasting are treated as **fully implemented** under the project’s stream/test validation scope.