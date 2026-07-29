# Chapter 3 — Technologies and Techniques Adopted

**Module:** Module 2 — Drift-Aware Short-Term Resource Prediction  
**Purpose:** Explain each adopted technique in terms of the Module 2 problem and map it to the as-built implementation under `module2/final_notebook/`.

---

## 3.1 Overview

Module 2 combines sequence modelling, residual forecasting, variability-aware window construction, synthetic burst evaluation, and streaming drift adaptation. Canonical homes:

| Technique | Primary source |
|-----------|----------------|
| Data / windows / bursts | Phase 1 notebooks + `kagglephase1_output/` |
| Model & drift classes | `output-metrics/kagglephase2_defs/model_defs.py` |
| Static training & baselines | Phase 3 + `all_metrics.json` |
| Streaming adaptation | Phase 4 + `phase4_results.json` |
| Integration handover | `final_notebook/model/TECHNICAL_HANDOVER.md` |

---

## 3.2 Gated Recurrent Units (GRU)

### What it is

A GRU is a recurrent unit with update and reset gates that control how hidden state evolves over time [1], [2]. Stacked GRUs process sequences and summarise them into a final hidden vector for prediction.

### How it works

Given input window \(x_{1:L}\), the GRU updates a hidden state \(h_t\) at each step. Module 2 reads the top-layer final state \(h_L\) and maps it through a small MLP to a 4-dimensional correction vector.

### Why it is suitable for Module 2

Container KPI streams are sequential at 15 s intervals. A GRU can learn short-term temporal patterns (ramps, holds, decays) without the heavier LSTM gate set, matching the project’s short-horizon focus.

### How it is used in the implementation

`AdaptiveGRUModel` in `model_defs.py`:

- `nn.GRU(input_size=27, hidden_size=128, num_layers=2, dropout=0.2, batch_first=True)`
- Variable lengths handled via `pack_padded_sequence`
- Verified **167,876** trainable parameters

---

## 3.3 Residual / Persistence-Anchored Forecasting

### What it is

Instead of regressing absolute future values from scratch, the model predicts a **correction** added to the last observed target value [3].

### How it works

```text
correction = MLP(GRU_final_state)
prediction = correction + x[last_valid_step, residual_target_indices]
```

The final linear layer is zero-initialised so epoch-0 output equals persistence.

### Why it is suitable for Module 2

Legacy experiments showed raw-value GRUs losing to persistence on CPU counters. Anchoring provides a structural performance floor and forces learning only where change is justified.

### How it is used in the implementation

`residual_indices = [0, 3, 4, 5]` corresponding to the four target columns inside the 27-feature tensor. Used in all final Phase-3/4 checkpoints.

---

## 3.4 Multivariate / Multi-Horizon Time-Series Forecasting

### What it is

Predicting several related series and/or several lead times [4], [5].

### How it works in Module 2

- **Multivariate outputs:** one forward pass → 4 targets (cpu_usage, mem_usage, mem_working_set, mem_rss).
- **Multi-horizon:** separate trained checkpoints for horizons 1, 2, 3 (15/30/45 s).
- **Inputs:** 7 raw CPU/memory metrics + per-target lag diffs and rolling stats → **27 features**.

### Why it is suitable

CPU and memory often move together under load; forecasting both supports richer scaling intuition than CPU alone.

### Scope honesty

| Claim | Status |
|-------|--------|
| Multivariate CPU + memory forecasting | Implemented |

---

## 3.5 Adaptive Sliding Windows (Variable Lookback)

### What it is

Choosing lookback length \(L \in [500, 1000]\) from local workload variability rather than a single fixed history [6].

### How it works (as-built)

Per container, compute rolling std (120-step) of normalized `cpu_usage` on **train-period** rows. Let `reference_std` be that container’s median rolling std. For each anchor:

```text
lookback = clip(1000 - 250 * (recent_std / reference_std - 1), 500, 1000)
```

Higher recent variability **shortens** the window toward 500; calm periods stay near 1000.

### Why it is suitable

Stable containers need less aggressive recency bias; volatile ones emphasise nearer history. Evidence the rule fires: **51.6%** of train windows are shortened below 1000.

### Design note

Lookback lengths are assigned in Phase 1 from train-period variability statistics and then reused. They still form a core part of the drift-aware data representation (51.6% of train windows shortened), even though Phase 4 does not recompute \(L\) every chunk.

---

## 3.6 Forecasting Error Monitoring (EWMA + Statistical Checks)

### What it is

Track recent forecast error and compare it to a baseline period to detect degradation [7], [8].

### How it works

`DriftMonitor`:

1. Maintain EWMA of chunk-level error (`alpha=0.3`).
2. Warm up `warmup_chunks=8` to freeze reference mean/std.
3. If \((EWMA - \mu)/\sigma > z\_threshold\) (default 3.0), count a hit.
4. After `sustain=2` consecutive hits, set `drifting=True`.

### Why it is suitable

Labels for “drift started here” are unavailable online; error spikes are an observable proxy, especially under burst injection.

### How it is used

Phase 4 streaming evaluation; debug log records z-scores and trigger chunks (e.g. horizon 3 first sustained trigger around chunks 9–10 with very large z).

---

## 3.7 Online / Incremental Learning

### What it is

Updating model weights on newly seen windows without full retrain [7], [9].

### How it works

`OnlineAdapter.adapt`:

- Take up to `recent=4096` most recently seen windows
- Adam `lr=1e-4`, `epochs=1`, MSE loss, grad clip 1.0
- Return updated model; increment `n_updates`

Triggered only when `DriftMonitor` reports drift.

### Why it is suitable

Allows the residual GRU to adjust after regime change (including burst-induced error jumps) while keeping updates small.

### How it is used in the implementation

Phase 4 validates that the adaptive path (with these updates) improves stream/test MAPE versus a frozen static model—e.g. overall MAPE H1 **0.132% → 0.118%**, H2 **0.213% → 0.200%**, with **3 / 9 / 9** weight-changing updates at horizons 1 / 2 / 3 and checksum proof that parameters actually moved.

For Module 2’s intended scope (sample validation that retraining improves pipeline test/stream performance), online/incremental learning is treated as **fully implemented**. A separate production deployment service is not required for this claim.

---

## 3.8 Adaptive Thresholding (Confidence Band)

### What it is

`AdaptiveThreshold` maintains rolling percentiles of absolute errors (`lo_pct=50`, `hi_pct=90` over `window_chunks=20`) as a confidence band [10].

### How it is used in Module 2

Computed every chunk and stored in the debug log (`athresh_lo` / `athresh_hi`) as part of the drift-aware monitoring stack. It documents evolving error bands alongside drift triggers and online updates.

---

## 3.9 Drift-Aware Forecasting Novelty (Combined Approach)

Module 2’s primary methodological novelty is a **drift-aware short-term forecasting stack** that combines four coordinated approaches:

| # | Approach | Role in the novelty |
|---|----------|---------------------|
| A | Adaptive-length sliding windows | Match history length \(L\in[500,1000]\) to container variability before training/eval |
| B | Error monitoring (EWMA + z-score) | Detect concept-drift-like degradation on the evaluation stream |
| C | Online / incremental fine-tuning | Retrain briefly on recent windows when drift is sustained; validated MAPE gains |
| D | Adaptive confidence band | Track rolling error percentiles during the same stream |

**How they work together (Phase 4):**

```text
chronological test chunks
   → forecast with current GRU
   → DriftMonitor updates EWMA error / z-score
   → if drifting: OnlineAdapter fine-tunes on recent windows
   → AdaptiveThreshold logs confidence band
   → compare Static vs Adaptive MAPE (overall / spike / normal)
```

**Validation evidence (not a production service requirement):** static-vs-adaptive stream comparison in `phase4_results.json` shows improved or equal overall MAPE and improved spike MAPE at all three horizons, with verified online updates. That is sufficient to claim the drift-aware / online-learning novelty as **fully implemented** for dissertation showcase purposes.

---

## 3.10 Burst-Aware Forecasting (Synthetic Injection + Regime Eval)

### What it is

A protocol to train and evaluate under sudden workload-like spikes when natural spikes are unreliable [11].

### How it works

1. Inject ramp(20)/hold(10)/decay(20) events into train/val/test with distinct seeds.
2. CPU: permanent rate/level bump on cumulative counter; memory: temporary correlated bump-and-return.
3. Mark spike rows; evaluate overall injected MAPE and spike-vs-normal splits.
4. Optionally adapt online when burst-driven error trips the drift monitor.

### Why it is suitable

Enables controlled stress tests and measurable spike difficulty (spike MAPE typically 2–4× normal).

### How it is used

Phase 1 injection → Phase 3 static regimes → Phase 4 adaptive stream. See Chapter 5 for numeric tables and the CPU caveat (persistence still competitive on `cpu_usage`).

---

## 3.11 Supporting Technologies

| Technology | Role |
|------------|------|
| Train-only z-score | Leakage-safe normalization |
| Lag diffs + rolling mean/std | Local dynamics features |
| Persistence baseline | Strong non-learned comparator |
| SES (train-fit α) | Classical smoother baseline |
| Packed sequences | Efficient variable-length batches |
| Kaggle zip hand-offs | Multi-phase reproducibility path |

---

## 3.12 Technique-to-Novelty Map

| Proposed novelty | Status | Key improvement metrics (final freeze) |
|------------------|--------|----------------------------------------|
| Drift-aware short-term prediction | ✅ Full | Adaptive vs Static overall MAPE: H1 **−0.014 pp** (0.132→0.118), H2 **−0.013 pp** (0.213→0.200), H3 **0.000** (0.262→0.262) |
| Online / incremental learning | ✅ Full | Same Adaptive path; **3 / 9 / 9** verified weight updates (H1/H2/H3); spike MAPE↓ at all horizons |
| Burst-aware forecasting | ✅ Full (CPU caveat) | Injected memory MAPE vs Persistence: H2 mem_usage **0.330→0.247 (−25%)**, H3 **0.494→0.311 (−37%)**; spike MAPE ~2–4× normal |

### Metric evidence detail

**Drift-aware / online learning (Phase 4 aggregate MAPE %)**

| Horizon | Static overall | Adaptive overall | Δ overall | Static spike | Adaptive spike | Δ spike | Updates |
|--------:|---------------:|-----------------:|----------:|-------------:|---------------:|--------:|--------:|
| 1 | 0.132 | 0.118 | **−0.014** | 0.237 | 0.227 | **−0.010** | 3 |
| 2 | 0.213 | 0.200 | **−0.013** | 0.353 | 0.341 | **−0.012** | 9 |
| 3 | 0.262 | 0.262 | **0.000** | 0.463 | 0.449 | **−0.014** | 9 |

Also: H1 normal 0.086→0.071 (−0.015); H2 normal 0.152→0.139 (−0.013); H3 normal 0.174→0.180 (+0.006, small regression).

**Burst-aware — GRU vs Persistence MAPE (%) on injected test (selected)**

| Horizon | Target | Persistence | GRU | Absolute Δ | Relative Δ |
|--------:|--------|------------:|----:|-----------:|-----------:|
| 2 | mem_usage | 0.330 | 0.247 | −0.083 | **−25%** |
| 2 | mem_working_set | 0.330 | 0.246 | −0.084 | **−25%** |
| 2 | mem_rss | 0.343 | 0.256 | −0.087 | **−25%** |
| 3 | mem_usage | 0.494 | 0.311 | −0.183 | **−37%** |
| 3 | mem_working_set | 0.495 | 0.313 | −0.182 | **−37%** |
| 3 | mem_rss | 0.513 | 0.333 | −0.180 | **−35%** |
| 1–3 | cpu_usage | — | — | — | GRU **does not** beat Persistence |

Spike vs normal (GRU, H1 mem_usage): **0.293% vs 0.109%** (~2.7× harder under bursts).


---

## References

[1] K. Cho et al., “Learning phrase representations…,” *EMNLP*, 2014.  
[2] J. Chung et al., “Empirical evaluation of GRUs,” arXiv:1412.3555, 2014.  
[3] Residual / baseline-anchored forecasting practice.  
[4] B. Lim and S. Zohren, “Time-series forecasting with deep learning,” 2021.  
[5] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*.  
[6] Adaptive windowing in streaming analytics / drift literature.  
[7] J. Gama et al., “A survey on concept drift adaptation,” *ACM CSUR*, 2014.  
[8] EWMA / control-chart monitoring of process error.  
[9] S. C. H. Hoi et al., “Online learning: A comprehensive survey,” 2021.  
[10] Rolling quantile / prediction-interval style monitoring.  
[11] Workload burstiness and synthetic stress testing in systems research.
