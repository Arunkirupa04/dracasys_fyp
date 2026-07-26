# Thesis Notes — Drift-Aware Extension (v2)

**Last updated:** 2026-07-11  
**Source notebooks:** `notebook/mdc_drift_aware.ipynb`, `notebook/mdc_eval_utils.py`  
**Reference run (after Task 3 re-run):** `notebook/run/mdc_drift_aware_output_v2.ipynb`  
**Legacy invalid run:** `notebook/run/mdc_drift_aware_output.ipynb` (v1 thr bug — do not cite adaptive FPR)

---

## 1. Headline metrics (lock for Abstract / Results)

| Metric | Value | Use in thesis |
|--------|-------|---------------|
| Default test ROC-AUC | **0.7402** | Methods baseline |
| Multi-seed ROC-AUC | **0.7529 ± 0.023** | Robustness claim |
| HPO test ROC-AUC | **0.8138** | Best-case performance |
| CPU latency | **~3.3 ms/window** | Deployment feasibility |

**Score convention:** `SCORE_INVERT=True` → raw scores negative; alerts use **alert space** (higher = more anomalous).

---

## 2. Simulated streaming (Methods §5.x)

### Terminology (required)

Use **"simulated streaming replay"** or **"ordered holdout evaluation"**.  
Do **not** claim real-time temporal deployment validation.

### Ordering modes (automatic in pipeline)

| Mode | When | Interpretation |
|------|------|----------------|
| `timestamp` | `ts_test` in `windows_vnext.npz` | True temporal replay by `window_end_ts` |
| `container_lexsort` | Legacy npz without timestamps | Pseudo-chronological proxy |
| `identity` | No container IDs | Raw test index order |

### Stable / drift periods

- First **70%** of replay = stable period (analysis label)
- Last **30%** = drift period (analysis label)
- This is **not** calendar drift unless `timestamp` mode is active
- Train/val/test split remains random stratified by `(container × label)` upstream

### Suggested Methods paragraph

> We evaluate drift-aware monitoring offline via simulated streaming replay of the held-out test set. Windows are ordered by per-window end timestamps when available (`window_end_ts` exported at preprocess); otherwise by container ID as a pseudo-chronological proxy. The replay exercises adaptive thresholding, PSI-based drift signals, and optional incremental fine-tuning in a single pass. This assesses monitoring logic under ordered replay; it does not constitute a live deployment test.

### Figure caption template

> *Figure X: Simulated streaming replay of the test set (ordering: {timestamp|container}). Red shading marks attack windows. Adaptive threshold trace shown in raw score space; calibration performed in alert space.*

---

## 3. Adaptive threshold (Results §6.x)

### Score / threshold convention (Task 1 — critical)

| Symbol | Meaning |
|--------|---------|
| `SCORE_INVERT` | How `compute_scores()` was produced (`auto_score_flip` in metrics) |
| `ALERT_INVERT` | **False** on saved `scores_vnext.npz` (already higher=anomalous) |
| `thr_fixed` | Loaded from `metrics_vnext.json` → `test_default.f1_optimal.threshold` (≈ **−0.1434**) |

**v1 failure mode (do not cite):** recomputed thr with `invert=SCORE_INVERT` → thr≈−0.0923 + double invert → stream fixed FPR ≈ **97%** → fake `fpr_reduction≈0.92`.

**Integrity gate:** `validate_threshold_integrity` must print `THRESHOLD INTEGRITY: PASS` before any adaptive claim.

### Design (alert-space adaptive)

On saved scores (`ALERT_INVERT=False`):

```
s_alert = s_raw                          # already higher = more anomalous
threshold_alert = median(benign) + α · std(benign)
alert = s_alert >= threshold_alert
```

### Metrics to report (Task 2 — matched operating points)

| Metric | Fixed (last half) | Adaptive (last half) |
|--------|-------------------|----------------------|
| FPR | from `adaptive_report` | from `adaptive_report` |
| Recall | ✓ | ✓ |
| F1 | ✓ | ✓ |
| MCC | ✓ | ✓ |

Also export `matched_policy_comparison.json`:
- ROC recall/F1 @ 5%/10%/25% FPR (score quality)
- Alpha-matched row (adaptive α closest to fixed FPR)
- `claim_adaptive_improves_without_recall_loss` boolean

### Honest framing

- If adaptive **reduces FPR** without sacrificing recall → support adaptation claim (`claim_ok=True`)
- If adaptive **trades** recall for lower FPR → report trade-off
- If no improvement → valid negative result; threshold-only adaptation insufficient
- **Never** cite v1 `fpr_reduction_last_half = 0.92`

---

## 4. Incremental fine-tune (Discussion §7.x)

### Observed result (v1 run — expect similar after v2)

| Metric | Before FT | After FT | Δ |
|--------|-----------|----------|---|
| Drift-slice AUC | 0.8508 | 0.7977 | **−0.053** |

### Interpretation (catastrophic forgetting)

1. **Procedure:** 50 steps, lr=1e−5, MSE on 200 recent **benign** windows from last 30% of replay.
2. **Mechanism:** Benign-only MSE adaptation shifts the reconstruction manifold toward recent normal traffic, eroding the attack separation boundary learned during offline training.
3. **Drift-slice AUC > full-test AUC:** Last 30% may be easier to separate due to ordering/label mix — not proof the period is inherently harder.
4. **Conclusion:** Drift **detection** (PSI) is supported; naive drift **adaptation** via unsupervised fine-tune is **not** supported.

### Suggested Discussion paragraph

> We implemented incremental benign fine-tuning as proposed. On the drift-period replay slice, ROC-AUC decreased by 0.053 after 50 MSE steps (AUC 0.851 → 0.798), indicating catastrophic forgetting of the offline decision boundary. This suggests that unsupervised adaptation requires stronger regularization (e.g., elastic weight consolidation, replay buffers, or adapter layers) than naive MSE fine-tuning. For this dataset, **threshold recalibration** is a safer operational response than **weight updates**.

### What you CAN claim

- Implemented and evaluated incremental fine-tune transparently
- Demonstrates distinction between drift detection vs adaptation

### What you CANNOT claim

- "Fine-tuning improves drift-period detection"
- "Drift-aware system adapts successfully online"

---

## 5. Isolation Forest baseline (Results §6.x)

### v2 methodology

| Setting | Value | Rationale |
|---------|-------|-----------|
| Training data | Benign `X_train` only | Matches AE paradigm |
| Features | mean + max pool → `(N, 2F)` | Preserves bursts vs flatten |
| Contamination | `'auto'` | Avoids injecting test attack rate |
| Direction | Val ROC picks sign | Same principle as AE flip |

### Reporting

- Primary row: `Isolation Forest (mean+max)`
- Footnote: flattened sensitivity run in `baseline_comparison.json`
- Compare against vNext default (0.7402) and Exp A (0.7163)

---

## 6. Ablation study (Results §4.4)

### From saved metrics (no GPU)

- Full vNext HPO, default, multi-seed — `metrics_vnext.json`

### From drift notebook (no retrain)

- Fixed vs adaptive threshold on stream
- Fine-tune before/after on drift slice

### Optional retrain (`mdc_model_vNext.ipynb` §18b)

Set `RUN_ABLATIONS = True`:

| Ablation | Change |
|----------|--------|
| `no_contractive` | `contractive_lambda=0` |
| `no_early_stop` | `auc_early_stop=False` |

Output: `metrics_ablation_vnext.json` → auto-loaded in drift notebook §9.

---

## 7. Re-run checklist (Colab) — Task 3

1. Upload **both** `mdc_eval_utils.py` + `mdc_drift_aware.ipynb` to `/content/` (utils first)
2. Ensure run artifacts present: `checkpoint_vnext.pt`, `scores_vnext.npz`, `metrics_vnext.json`
3. (Optional) Re-run `mdc_preprocess_vNext.ipynb` for `ts_test` in npz
4. Run full `mdc_drift_aware.ipynb` top → bottom
5. Confirm §4 prints `THRESHOLD INTEGRITY: PASS` and `thr_fixed ≈ -0.1434`
6. Confirm §5 writes `matched_policy_comparison.json` + `fig07b_matched_operating_points.png`
7. Save output as `notebook/run/mdc_drift_aware_output_v2.ipynb`
8. Copy `drift_aware/` + `thesis_figures_300dpi/` to Drive

### Validation gates (Task 3)

| Check | Expected |
|-------|----------|
| Model AUC check | test ≈ **0.7402** |
| `thr_fixed` | ≈ **−0.1434** (from metrics, not −0.0923) |
| `ALERT_INVERT` | **False** |
| Fixed stream FPR | **≪ 0.50** (≈ offline ~0.14) |
| Fixed stream F1 | ≈ **0.74** (matches offline) |
| `THRESHOLD INTEGRITY: PASS` | Required |
| `matched_policy_comparison.json` | Present |
| IF ROC-AUC | > **0.55** (ideally 0.60+) |
| `stream_ordering.ordering_mode` | `timestamp` or `container_lexsort` |
| Adaptive claim | Only if `claim_ok=True`; else report trade-off |

---

## 8. Claim matrix (submission-ready)

**Canonical locked matrix:** [`docs/claim_evidence_matrix.md`](claim_evidence_matrix.md) (Task 5).

| Claim | Supported? |
|-------|------------|
| Transformer AE detector (ROC 0.74–0.81) | ✅ |
| Leakage-free preprocess + drift baseline | ✅ |
| PSI drift detection on holdout | ✅ |
| Simulated streaming replay + logging | ✅ |
| CPU inference < 100 ms | ✅ |
| Adaptive threshold reduces FPR | ⚠️ Only if `claim_ok=True` after Task 1–3 re-run; else trade-off |
| Incremental fine-tune improves detection | ❌ Report as negative result |
| Live temporal deployment validated | ❌ Use simulated framing |
| IF baseline inferior to AE | ✅ After v2 IF fix |
| Fixed stream FPR ~97% / FPR↓0.92 | ❌ **Invalid v1 artifact — do not cite** |

### Locked headline usage (do not swap)

| Number | Where |
|--------|-------|
| **0.7529 ± 0.023** | Abstract (robustness) |
| **0.7402** | Methods default reference |
| **0.8138** | Results best-case (HPO) |

---

## 9. Task 4–10 status

| Task | Status | Action remaining |
|------|--------|------------------|
| 4 Freeze v2 | Infra ready | Colab re-run → `mdc_drift_aware_output_v2.ipynb` |
| 5 Claim lock | **DONE** | Use `claim_evidence_matrix.md` |
| 6 Timestamps | **CODED** | Re-export both npz files |
| 7 Retrain ablations | **HARNESS READY** | `RUN_ABLATIONS=True` on GPU |
| 8 Ablation tables | **DONE** | Fill Table A after Task 7 |
| 9 Results/Discussion | **DONE (draft)** | `docs/thesis_results_discussion.md` |
| 10 Attack names | **DONE (code)** | Re-run model §19 for named figures |

**Chapter draft:** [`thesis_results_discussion.md`](thesis_results_discussion.md)  
**Label map:** `notebook/mdc_label_map.py`
