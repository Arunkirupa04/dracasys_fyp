# Module 4 — Root Cause Analysis & Fix Plan
## Adaptive Threshold + Online/Incremental Learning (Drift-Aware Layer)

**Scope:** This document covers **only** the two drift-adaptation mechanisms in
`kaggle_drift_aware_2607.ipynb` — (A) the benign-buffer adaptive threshold, and
(B) the incremental benign-only fine-tune. It does not re-litigate PSI/KS drift
*detection* (which works) or the base detector (see
`MODULE4_NOVELTY_VALIDATION_REPORT.md` for that).

**Evidence base:** direct inspection of
`module4/notebook/final/output-metrics/drift_aware_outputs/{adaptive_report.json, finetune_report.json, freeze_manifest_v2.json, drift_aware_summary.json, drift_aware_log.csv}`,
cross-checked against the executed cells in
`module4/notebook/final/kaggle-output/kaggle_drift_aware_2607.ipynb`. All numbers
below were recomputed directly from `drift_aware_log.csv` (5153 rows, one per
stream step), not copied from prior summaries.

---

## Executive summary

| Mechanism | Proposal claim | Current state | Severity |
|---|---|---|---|
| **Adaptive threshold** | "Adaptive thresholding to dynamically adjust anomaly decision boundary" | **Fires zero alerts across all 5,153 stream steps** — not "degrades under drift," it never once triggers, from step 1 to step 5153, at any of the four swept α values (1.5/2.0/2.5/3.0). | **Critical — mechanism is non-functional, not merely sub-optimal** |
| **Online/incremental learning** | "Online/incremental learning for continuous model adaptation" | Two independent failure modes: (1) in this freeze, the fine-tune never actually runs on genuinely recent data — it silently substitutes year-old windows; (2) on the one occasion it *did* run on real recent data (a historical run), it made detection **worse** (ΔAUC ≈ −0.06, catastrophic forgetting). | **Critical — mechanism either doesn't execute as designed, or actively harms accuracy when it does** |

Both are fixable with moderate, well-scoped engineering changes (no new data collection, no architecture change). The fix plan below is ordered by effort vs. impact.

---

## Part A — Adaptive Threshold

### A.1 What the code does today

`kaggle_drift_aware_2607.ipynb`, `AdaptiveThresholdSimulator` + `compute_adaptive_threshold()`:

```python
ADAPTIVE_ALPHA         = 2.5
BENIGN_BUFFER_SIZE     = 500
THRESHOLD_UPDATE_EVERY = 20

def compute_adaptive_threshold(benign_alert_scores, alpha=2.5, min_samples=5):
    buf = np.asarray(benign_alert_scores, dtype=np.float64)
    if len(buf) < min_samples:
        return float(np.median(buf)) if len(buf) else 0.0
    return float(np.median(buf) + alpha * np.std(buf))   # <-- the failure point
```

Every 20 stream steps, once ≥5 benign windows have been observed, the threshold
is recomputed as `median(recent ≤500 benign scores) + α · std(recent benign scores)`.
This is a classic **statistical process-control** rule (the same family as
Shewhart/3-sigma control charts), and it is the correct family **only** when
the calibration data is roughly symmetric / light-tailed.

### A.2 What actually happens — measured directly from `drift_aware_log.csv`

```
Global score range (all 5,153 windows):  min = -3.5343   max = -0.0404
Benign score stats:  median = -0.361   mean = -0.528   std = 0.504
Attack  score stats:  median = -0.162   mean = -0.327   std = 0.379
```

Benign reconstruction-error scores are **heavy-tailed**: most benign windows
score around −0.2 to −0.6, but a minority score as low as −3.5 (the AE finds
some ordinary benign sessions much harder to reconstruct than others). This
tail inflates `std` to ≈0.50 — almost as large as the *entire* usable score
range up to the ceiling (−0.04).

Consequence, traced step-by-step in the log:

| Stream step | % into stream | `threshold_raw` | Global score ceiling | What happened |
|---:|---:|---:|---:|---|
| 0–118 | 0–2% | −0.13 to −0.20 | −0.04 | Still below ceiling; a few alerts would still be theoretically reachable |
| 139–158 | 3% | −0.118 | −0.04 | A short run of unusually low-scoring (but genuinely benign) windows enters the 500-buffer |
| **159** | **3.1%** | **−0.004** | **−0.04** | **Threshold crosses above the highest score any window (benign or attack) achieves anywhere in the dataset** |
| 250–5152 | 5–100% | climbs to +0.18 → **+1.53**, settles ≈ +0.50 | −0.04 (fixed) | Threshold is now permanently unreachable by any real score |

**Result: `adaptive_report.json` → `adaptive_global = {f1: 0.0, mcc: 0.0, recall: 0.0}`. Total alerts fired in the entire 5,153-step stream: `df['alert'].sum() == 0`.**

The α-sweep in the same run (`alpha_sweep` in `adaptive_report.json`) tested
α ∈ {1.5, 2.0, 2.5, 3.0} and **every single value produced recall = 0, FPR = 0,
F1 = 0** in the last-half comparison. This rules out "α is a bit too
conservative" — the mechanism is broken by construction for this score
distribution, not merely mistuned.

### A.3 Root cause

> **`median + α·σ` is the wrong estimator family for a heavy-tailed, one-sided-bounded score distribution.** Because a handful of ordinary (non-attack) windows produce disproportionately extreme reconstruction error, the sample standard deviation of the benign calibration buffer is inflated to roughly the same magnitude as the median itself. `median + 2.5σ` (and even `median + 1.5σ`) therefore overshoots the score space's hard ceiling — a ceiling that exists because the scoring formula (`0.3·mean_err + 0.7·max_err`, inverted) can only ever produce values in a bounded range. Once the threshold crosses that ceiling, **no future window of either class can ever satisfy `score ≥ threshold` again**, and this happens after seeing only ~150–300 calibration samples (3–6% of the stream) — long before any real "drift period" begins. This is a **distributional-shape bug**, not a drift-robustness bug: it would fail identically even on a perfectly stationary, non-drifting stream, which is why `adaptive_first_half` and `adaptive_last_half` are *both* zero, not just the drift half.

Two compounding design gaps make it worse:
1. **No sanity governor.** Nothing in `AdaptiveThresholdSimulator` checks the proposed threshold against the range of scores actually observed so far. A one-line assertion (`assert candidate_thr <= running_max_score_seen`) would have caught this in development.
2. **No smoothing.** The threshold is fully recomputed from scratch every update (no exponential moving average, no capped step size), so one volatile 20-step window can move the operating point by more than a full unit in score-space in a single update (observed: −0.118 → −0.004 in one update).

### A.4 Fix plan — ranked by effort vs. impact

| # | Fix | Effort | Impact | Description |
|---|---|---|---|---|
| **1** | **Switch to a robust scale estimator (MAD instead of σ)** | Low (1 line) | High | Replace `np.std(buf)` with `1.4826 * median_abs_deviation(buf)`. MAD is far less sensitive to the outlier tail that currently inflates σ. This alone should keep the threshold inside the achievable score range for most of the stream. |
| **2** | **Add a hard sanity governor** | Low (3–5 lines) | High (safety net) | Clip every candidate threshold to `min(candidate_thr, running_max_score_seen_so_far)`, and separately never let it fall below the 1st percentile of the buffer. This makes the "threshold escapes the achievable range" failure mode structurally impossible, independent of fix #1. |
| **3** | **Recalibrate to a target-FPR quantile instead of a parametric rule** | Low–Medium | High | Replace `median + α·σ` with `np.quantile(benign_buf, target_percentile)` where `target_percentile` is chosen to match the *operating regime* the fixed threshold already lives in (FPR ≈ 25–30% at F1-optimal, per `metrics_vnext.json`) rather than a "rare-outlier" assumption borrowed from low-base-rate fault detection. A quantile of a finite sample is, by definition, always inside the sample's own range — this removes the overshoot failure mode entirely, and removes the need for fix #2 as a safety net (though keep #2 anyway, defense in depth). |
| **4** | **Smooth threshold updates (EWMA)** | Low | Medium | `thr = 0.85 * thr_prev + 0.15 * candidate_thr` (or cap `|Δthr|` per update to a fraction of the observed score range). Prevents any single volatile 20-step window from swinging the operating point by more than a bounded amount, which also softens the impact of true drift once it occurs. |
| **5** | **Champion/challenger gate before adopting a new threshold** | Medium | High | Every update, evaluate the *candidate* threshold against a short lookback window of already-revealed labels (available in this simulated-replay setting) before committing it. If the candidate's recall on that lookback window would drop below, say, 50% of the current threshold's recall, keep the old threshold and log the rejected candidate instead. This directly prevents the "silently stop alerting forever" failure mode regardless of which estimator (#1/#3) is used. |
| **6** | **Re-run the α-sweep only after #1–#3 are in place** | — | Validation | The existing `RUN_ALPHA_SWEEP` infrastructure and `matched_policy_comparison.json` claim gate are good and should be kept as-is — they correctly caught this failure. Re-run them post-fix as the acceptance test. |

**Minimal patch (fixes #1 + #2 + #3 together, ~15 lines):**

```python
def compute_adaptive_threshold(benign_alert_scores, target_percentile=72.0,
                                 running_max_seen=None, min_samples=5):
    buf = np.asarray(benign_alert_scores, dtype=np.float64)
    if len(buf) < min_samples:
        return float(np.median(buf)) if len(buf) else 0.0
    # Quantile-based (robust to heavy tails; always within observed range).
    thr = float(np.quantile(buf, target_percentile / 100.0))
    # Governor: never exceed the highest score observed anywhere so far.
    if running_max_seen is not None:
        thr = min(thr, running_max_seen)
    return thr
```

`target_percentile≈72` is a starting point chosen to land near the fixed
threshold's empirical FPR (~28%) on this dataset — tune it against
`metrics_vnext.json`'s `test_default.f1_optimal.fpr` rather than picking a
generic "outlier" percentile like 95/99.

### A.5 How to validate the fix

1. Re-run §5 of the drift-aware notebook with the patched estimator.
2. Confirm `adaptive_report.json → adaptive_global.recall > 0` and, ideally,
   comparable to `fixed_global.recall` (0.725).
3. Confirm the α/percentile-sweep no longer produces an all-zero row for every
   setting swept — at least one setting should show a **non-trivial**
   trade-off curve (some FPR reduction *with* a bounded recall cost), which is
   what §5's `matched_policy_comparison.json` is designed to report.
4. Only then revisit whether `claim_adaptive_improves_without_recall_loss`
   can honestly flip to `true` — do not force this; a well-instrumented
   negative result is still a valid result if the mechanism is at least
   *functioning* (non-zero alerts) but still doesn't beat the fixed baseline.

---

## Part B — Online / Incremental Learning (Benign-Only Fine-Tune)

### B.1 What the code does today

`kaggle_drift_aware_2607.ipynb` §6, `fine_tune_on_recent()`:

```python
DRIFT_SLICE_FRAC   = 0.30     # last 30% of the timestamp-ordered stream
FINETUNE_STEPS     = 50
FINETUNE_LR        = 1e-5
FINETUNE_BENIGN_N  = 200

X_drift = X_stream[split_idx:]                  # last 30% of the stream
y_drift = y_stream[split_idx:]
benign_drift_idx = np.where(y_drift == 0)[0]

if len(benign_drift_idx) > 0:
    X_ft = X_drift[benign_drift_idx[-FINETUNE_BENIGN_N:]]
    ft_source = 'drift_slice'
else:
    # falls back to OLD data from the first 70% ("stable") slice
    X_ft = X_stream[:split_idx][benign_stable_idx[-FINETUNE_BENIGN_N:]]
    ft_source = 'stable_slice_fallback'
```

Fine-tuning is a plain unsupervised MSE update (AdamW, weight_decay=1e-4, grad
clip 1.0) on whichever 200 benign windows it finds, then the resulting model
is evaluated against `y_drift` (the last-30% slice) before/after.

### B.2 What actually happens

**Failure mode 1 — data starvation (this freeze):**

```
finetune_report.json:
  finetune_benign_source : "stable_slice_fallback"
  auc_before             : NaN
  auc_after              : NaN
  auc_delta              : NaN
```

The chronological last-30% slice (`X_drift`, 1,546 windows) is **100% attack**
— `len(benign_drift_idx) == 0`. This is confirmed by the PSI-over-time trace
computed independently in `kaggle_analysis_final.ipynb` §4: buckets 14–18
(2024-04-25 to 2024-04-26) show `mean_psi` between 1.79 and 2.75 — a sustained,
extreme distributional spike — consistent with a **dedicated attack-only
recording block** at the end of the dataset's timeline, not gradual concept
drift. Two consequences:

- `y_drift` is single-class, so ROC-AUC is mathematically undefined
  (`auc_before`/`auc_after` = NaN) — not a bug in the AUC computation, a
  genuine consequence of the data.
- The `stable_slice_fallback` path then fine-tunes on windows from the
  **first 70%** of the stream — i.e., **not recent data at all**. The
  mechanism silently does the opposite of what "online adaptation to recent
  behaviour" means, while still logging a value (`"interpretation":
  "positive_delta"`) that reads as if something informative happened.

**Failure mode 2 — catastrophic forgetting (historical run, real recent data):**

The one run in the project's history where genuine recent-benign fine-tune
data *was* available (`output-legacy-experiments`, drift v1) shows:

```
Drift-slice AUC before fine-tune: 0.8508
Drift-slice AUC after  fine-tune: 0.7902
Δ = −0.0605
```

This is not noise — it is the expected outcome of unregularized benign-only
MSE fine-tuning: minimizing reconstruction error on recent benign traffic
necessarily makes the model reconstruct *anything nearby in feature space*
better, including attack windows that share feature-space territory with that
recent benign traffic (compressed further by the upstream variance/correlation
filtering and ±10 clipping). The attack/benign error gap the whole detector
depends on shrinks as a direct result of the very training step meant to help.

### B.3 Root cause

There are **two distinct, independent root causes** — both must be fixed;
fixing only one leaves the mechanism unreliable.

1. **Data-definition root cause (why it doesn't even run on real data in this freeze):** the "drift slice" is defined as a *fixed* percentage (last 30%) of the replay. The MDC dataset's attack scenarios were evidently recorded in **calendar blocks** (specific CVE tests run in dedicated sessions), so a fixed percentage cutoff can land entirely inside an attack-only block, starving the fine-tune of any real "recent benign" to learn from. This is a property of how the dataset was collected, not something a smarter fine-tuning *algorithm* can fix — the **slice-selection logic** needs to change instead.
2. **Objective/regularization root cause (why it hurts even when it does run on real data):** plain MSE fine-tuning has no mechanism to preserve the original decision boundary. There is no anchor loss, no frozen sub-network, no held-out check before the fine-tuned weights are trusted. This is a standard, well-documented failure mode (catastrophic forgetting in continual learning) with standard, well-documented mitigations that this project's own thesis notes correctly identify as *future work* — they should be promoted to first attempts, since they are cheap relative to the damage they prevent.

### B.4 Fix plan — ranked by effort vs. impact

| # | Fix | Effort | Impact | Description |
|---|---|---|---|---|
| **1** | **Source fine-tune data from the adaptive threshold's rolling benign buffer, not the fixed 30% slice** | Low (reuse existing buffer) | High | The adaptive-threshold subsystem (Part A) already maintains an up-to-500-window rolling buffer of the *most recently observed* benign windows, regardless of where the arbitrary 70/30 split falls. Point `fine_tune_on_recent()` at that buffer instead of `X_drift[benign_drift_idx]`. This guarantees "recent" always means recent-in-stream-order and can never be starved by an attack-only block, without inventing new infrastructure. |
| **2** | **Freeze most of the network; fine-tune only a small adapter** | Medium | High | Freeze `input_proj`, `encoder`, `decoder` weights; unfreeze only `bn_down`/`bn_up` (the bottleneck projection) or add a small LoRA-style adapter on the output projection. This bounds how far reconstruction behaviour can drift, directly limiting collateral damage to attack separability while still tracking benign-level drift. |
| **3** | **Add an anchor/EWC penalty to the fine-tune loss** | Medium | High | `loss = mse(recon, x) + λ * ||θ - θ0||²` (L2-SP), or a proper Fisher-weighted EWC term if time permits. Anchoring to the offline-trained weights `θ0` is the direct, textbook fix for catastrophic forgetting and is explicitly named as future work in `thesis_drift_aware_notes.md` §4 — implement the simplest version (L2-SP) first; it is a 2-line loss change. |
| **4** | **Champion/challenger gate before adopting the fine-tuned model** | Medium | Critical (safety net) | Before `drift_aware_eval()` swaps `m = model_ft`, evaluate both `model_base` and `model_ft` on a short lookback window of already-revealed labels (mixed benign+attack, causal — no future peeking). Only swap if `model_ft`'s AUC/F1 on that check-slice is not worse than `model_base`'s by more than a small tolerance (e.g. 0.02). This makes the −0.06 AUC regression structurally unable to reach the live scoring path, regardless of whether #2/#3 fully solve the underlying forgetting problem. |
| **5** | **Early-stop the fine-tune loop on a proxy separability metric** | Low | Medium | Track reconstruction-error separation (e.g., AUC on a small frozen check-set containing both classes) every few steps during the 50-step loop; stop as soon as it stops improving or starts degrading — mirroring the `auc_early_stop` pattern the offline training already uses successfully. |
| **6** | **Lock one fine-tune narrative after re-running with #1–#5** | — | Reporting | The project currently has three conflicting fine-tune outcomes across its history (−0.06 legacy, +0.0319 in the 2026-07-13 `run-final` audit, NaN in this freeze). Once the mechanism is fixed and re-run, report exactly one number in the thesis and retire the others as historical. |

**Minimal patch for the highest-leverage fix (#1, data-source change):**

```python
# Instead of sourcing from the fixed 30% slice:
#   X_ft = X_drift[benign_drift_idx[-FINETUNE_BENIGN_N:]]
#
# Source from the adaptive threshold's own rolling benign buffer, which is
# always populated with the most recently *observed* benign windows:
recent_benign_indices = sim.benign_window_index_buf[-FINETUNE_BENIGN_N:]  # requires
                                                                            # storing indices
                                                                            # alongside scores
                                                                            # in AdaptiveThresholdSimulator
X_ft = X_stream[recent_benign_indices]
ft_source = 'adaptive_buffer_rolling'   # replaces 'drift_slice' / 'stable_slice_fallback'
```

(Requires a small addition to `AdaptiveThresholdSimulator.observe()` to also
push the window's stream index into a parallel buffer — a few lines, no
architectural change.)

### B.5 How to validate the fix

1. Confirm `finetune_report.json → finetune_benign_source == "adaptive_buffer_rolling"` and `finetune_benign_n == 200` with **no** fallback path taken, on the current dataset.
2. Confirm `auc_before`/`auc_after` are finite (not NaN) because the evaluation slice is no longer forced to be single-class by construction — or, if it still is, evaluate on a slightly widened window rather than exactly the last 30%.
3. Re-run with #2 (adapter-only) and #3 (anchor loss) individually and combined; report `auc_delta` for each configuration in an ablation table (the project already has the `ablation_table_stream.csv` convention — extend it with a `finetune_variant` column).
4. Accept the mechanism as "supported" only if at least one configuration shows `auc_delta >= 0` on a genuinely recent, mixed-class evaluation slice, without cherry-picking; otherwise keep reporting the negative result honestly (as the project already does well), but now backed by a mechanism that at least *executes as designed*.

---

## Combined priority order (both mechanisms)

| Priority | Fix | Est. effort | Unblocks |
|---|---|---|---|
| 1 | Adaptive threshold: quantile-based calibration + max-seen governor (A.4 #1–#3) | ~1 hour | Makes adaptive thresholding functional at all — currently 0 alerts in 5,153 steps |
| 2 | Fine-tune: source from rolling benign buffer instead of fixed 30% slice (B.4 #1) | ~1–2 hours | Fixes NaN/starvation; makes the mechanism actually run on recent data |
| 3 | Fine-tune: champion/challenger gate (B.4 #4) | ~1–2 hours | Prevents catastrophic forgetting from ever reaching the reported/live result |
| 4 | Fine-tune: L2-SP anchor loss (B.4 #3) | ~30 min | Reduces the *magnitude* of forgetting when fine-tuning does run |
| 5 | Adaptive threshold: EWMA smoothing + champion/challenger (A.4 #4–#5) | ~1 hour | Hardens against future volatility once the estimator itself is fixed |
| 6 | Fine-tune: adapter-only / frozen-backbone variant (B.4 #2) | ~2–3 hours | Highest-quality fix for forgetting, but the most implementation work |
| 7 | Re-run α-sweep, ablations, and lock one fine-tune narrative (A.4 #6, B.4 #6) | ~1 hour + Kaggle GPU time | Final validation + thesis-ready numbers |

Total estimated effort to get both mechanisms from **non-functional** to
**functional and honestly evaluated**: roughly **one Kaggle GPU session**
(6–8 hours of wall-clock work including re-runs), no new data and no
architecture change required.

---

*Root cause figures in this document were recomputed directly from
`module4/notebook/final/output-metrics/drift_aware_outputs/drift_aware_log.csv`
(5,153 rows) and cross-checked against `adaptive_report.json` and
`finetune_report.json` in the same directory — see inline code excerpts above
for the exact recomputation.*
