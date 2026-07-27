# Adaptive Threshold — Robustness Fix Plan (Phase 3: why the win didn't hold)

**Status:** Planning only — no code changed yet.
**Scope constraint (unchanged):** only `module4/notebook/final/kaggle-source/mdc_drift_aware_kaggle.ipynb` gets touched when this is implemented.
**Builds on:** `MODULE4_ADAPTIVE_THRESHOLD_FIX_PLAN.md` (bug fix — confirmed still holding) and `MODULE4_ADAPTIVE_THRESHOLD_IMPROVEMENT_PLAN.md` (Phases 1–4 — implemented, but the win didn't reproduce).

---

## 1. What actually happened (the evidence)

Two back-to-back executions of the identical patched notebook, against what should be the same frozen model/thresholds, produced opposite verdicts:

| Run | Selected `target_percentile` | Validation `fpr_reduction` | **Test** `fpr_reduction` | `claim_ok` |
|---|---:|---:|---:|---|
| p2 (local replay, prior session) | 60.0 | — | +0.0063 | **True** |
| p3 (live Kaggle re-run) | 55.0 | +0.0176 | **−0.0397** | **False** |

The validation sweep said pctl=55 was a winner (+0.0176 FPR reduction, within tolerance). Applied to test, it did the opposite — adaptive FPR ended up *higher* than fixed's in both halves. The sign flipped between validation and test.

## 2. Root cause: the selection rule optimizes for "barely positive," which is the least stable thing to optimize for

Look at the p3 validation sweep:

```
pctl=50  fpr_red=-0.0400   (fails)
pctl=55  fpr_red=+0.0176   (qualifies -- barely)
pctl=60  fpr_red=+0.0400   (bigger margin, but recall_delta=-0.0963 -- FAILS the recall tolerance)
pctl=65+ ...................(fails recall tolerance, progressively worse)
```

Only **one** grid point (55) both qualifies and stays within the recall/F1 tolerance. The selection rule (`sweep_adaptive_percentile_on_val`) picks it because it's the only option — not because it's a robust choice. A percentile sitting alone at the edge between "just barely qualifies" and "the next grid point over already fails" is, almost by definition, the least stable value on the whole curve: a small shift in the underlying score distribution (which is exactly what happens between a separately-drawn validation set and the test stream on this block-structured dataset) is enough to push a `+0.0176` into a `-0.0397`.

Two compounding issues:

1. **No margin requirement.** The qualifying filter is `fpr_reduction_last_half > 0.0` — mathematically positive is treated the same as "robustly positive." A `+0.0176` and a `+0.15` pass the identical bar.
2. **Coarse grid (step of 5).** The sweep only tested {50, 55, 60, 65, ...}. Whatever the *actual* local optimum is, it may sit between 55 and 60 where recall stays inside tolerance *and* the margin is bigger than 0.0176 — the coarse grid can't see it.
3. **Single validation draw, no internal check.** The selection trusts one pass over the validation set with no check for whether that result would hold up on a *different* slice of data before committing to it on test — exactly the kind of check that would have caught this before it ever reached the test stream.

---

## 3. Fix plan — four changes, all inside `sweep_adaptive_percentile_on_val` / `sweep_adaptive_smoothing_on_val` and their call sites

### 3.1 (Highest priority) Require a real margin, not just `> 0`

```python
# New config (cell 4): ADAPTIVE_MIN_FPR_MARGIN = 0.02  (was: any positive value qualified)

qualifying = [
    r for r in rows
    if r["fpr_reduction_last_half"] >= min_fpr_margin      # was: > 0.0
    and r["recall_delta_last_half"] >= -recall_tol
    and r["f1_gain_last_half"] >= -f1_tol
]
```

If nothing clears a 2-point margin, that's the honest answer — report `status = "no_candidate_cleared_margin"` and fall back to the fixed threshold (`ADAPTIVE_TARGET_PCTL` stays at whatever produces zero deviation, i.e., effectively disable adaptation for this run) rather than shipping a coin-flip winner.

### 3.2 Finer grid around the promising region

Replace the 5-point-step grid with a two-stage search: coarse pass (current 8 points) to find the promising neighborhood, then a fine pass (step 1) around the best coarse candidate(s):

```python
ADAPTIVE_PCTL_SWEEP_VALUES       = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0)  # coarse, unchanged
ADAPTIVE_PCTL_FINE_HALF_WIDTH    = 4.0   # fine pass searches [best-4, best+4] in steps of 1
```

This costs a handful of extra `evaluate_stream_strategies` calls (cheap — pure array arithmetic, not model inference) and directly addresses "the real optimum might be sitting between two coarse grid points."

### 3.3 (Most important structurally) Nested validation: tune on one half, confirm on the other

Split the *validation* stream itself in two (chronologically, same `stream_order_indices` logic already used everywhere else). Select the candidate using only the tune-half; then require it to **also** clear the margin on the confirm-half before accepting it:

```python
def sweep_adaptive_percentile_on_val(..., nested_confirm=True, ...):
    order = stream_order_indices(len(labels_val), timestamps=timestamps_val)
    y_v, s_v = labels_val[order], scores_val[order]
    tune_end = len(y_v) // 2
    y_tune, s_tune = y_v[:tune_end], s_v[:tune_end]
    y_confirm, s_confirm = y_v[tune_end:], s_v[tune_end:]

    # ... sweep + select best on (y_tune, s_tune) exactly as today ...

    if nested_confirm:
        confirm_rep = evaluate_stream_strategies(y_confirm, s_confirm, fixed_threshold_raw, invert, sim_at_selected)
        if confirm_rep["fpr_reduction_last_half"] < min_fpr_margin:
            status = "failed_nested_confirmation"
            # fall back to no adaptation (percentile that reproduces the fixed threshold)
```

This is the single change most likely to have caught the p3 failure *before* it ever touched the test set — the entire point of a tune/confirm split is to ask "does this candidate's apparent win survive being checked against data it wasn't chosen using," which is precisely what the val→test transition later asked, just one step earlier and cheaper to iterate on.

### 3.4 Consider tuning against the test stream's own first half instead of (or alongside) the separate validation set

This dataset has calendar-block structure (different attack types dominate different real-world recording sessions — established during the RCA), so a separately-drawn validation set may simply have a different score-distribution shape than the test stream, no matter how well-tuned. The test stream's **own first half** is causally available (it strictly precedes the second half that `claim_ok` is scored on — no leakage) and is by definition drawn from the same underlying process as the half being scored. Add this as a second candidate data source and report both:

```python
ADAPTIVE_TUNE_SOURCE = "validation"  # or "test_first_half" -- compare both, pick whichever passes 3.1-3.3
```

This isn't guaranteed to help (first-half of the test stream showed its own recall=0 anomaly earlier in the RCA — a genuinely hard, non-stationary dataset), but it's a legitimate second opinion that costs little to compute since the infrastructure is already in place.

---

## 4. Rollout order

1. **3.1 (margin requirement) first, alone.** Re-run. If no candidate clears the margin on the existing coarse grid, that's already useful information — proceed to 3.2.
2. **3.2 (fine grid)** on top of 3.1. Re-run.
3. **3.3 (nested tune/confirm split)** on top of 3.1+3.2 — this is the one that directly targets the observed failure mode. Re-run.
4. **3.4 (test-first-half as an alternate tuning source)** as a final comparison, reported alongside the validation-based result either way.

Track all four stages in one ablation table (extending the existing `pctl_sweep`/`smoothing_sweep` report keys) so the thesis narrative is "here is the progressively more rigorous validation discipline we applied, and here's what survived."

## 5. Honest expectation, stated up front

Sections 3.1–3.3 are about **not shipping a coin-flip result** — they make the selection process trustworthy, but they cannot manufacture a real effect that doesn't exist in the data. Given the ROC-flatness analysis from the original improvement plan (a narrow but real window near FPR≈20–30%) and now two data points showing the margin available there is on the order of ±0.02–0.04 (smaller than typical run-to-run noise on this dataset), the realistic outcomes are:

- **A smaller number of percentile candidates survive 3.1–3.3**, and if one does, it's a genuinely more trustworthy "adaptive beats fixed" result — worth reporting as such.
- **Nothing survives nested confirmation**, in which case the honest, final, and still scientifically valuable conclusion is: *"adaptive thresholding, calibrated with proper margin and nested-validation discipline, could not be shown to reliably beat the static threshold on this dataset — the available improvement margin is smaller than the dataset's own val↔test variability."* That is a legitimate thesis finding, not a failure to deliver — and it directly explains, mechanistically, why the naive (single-validation-pass) tuning in the previous phase looked like a win and then wasn't.

Do not respond to a second failed run by loosening the margin or dropping the nested check — that repeats exactly the mistake this plan exists to fix.
