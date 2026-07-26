# Claim–Evidence Matrix (Locked for Thesis)

**Status:** LOCKED for headline metrics (Task 5)  
**Last updated:** 2026-07-11  
**Evidence freeze target:** `notebook/run/mdc_drift_aware_output_v2.ipynb`  
**Legacy (do not cite adaptive FPR):** `notebook/run/mdc_drift_aware_output.ipynb`

---

## 1. Locked headline metrics

| Metric | Value | Thesis location | Source artifact |
|--------|-------|-----------------|-----------------|
| Default test ROC-AUC | **0.7402** | Methods / Results baseline | `metrics_vnext.json` → `test_default.f1_optimal` |
| Multi-seed ROC-AUC | **0.7529 ± 0.023** | Abstract (robustness) | `metrics_vnext.json` → `multi_seed` |
| HPO test ROC-AUC | **0.8138** | Results best-case | `metrics_vnext.json` / HPO section |
| Exp A baseline | **0.7163** | Methods comparison | lineage table |
| Isolation Forest | **~0.674** | Results baseline | drift-aware §8 |
| CPU latency | **~3.36 ms/window** | Deployment feasibility | drift-aware §10 |
| Offline F1 @ f1_optimal | **~0.741** | Results operating point | `metrics_vnext.json` |
| Offline thr (raw) | **≈ −0.1434** | Methods threshold | `test_default.f1_optimal.threshold` |

**Usage rule:** Never swap these roles. Do not put 0.8138 in the Abstract as the only number without also stating multi-seed robustness.

---

## 2. Claim matrix

| ID | Claim | Status | Evidence | Figure / Table | Limitation |
|----|-------|--------|----------|-------------|------------|
| C1 | Transformer AE detects container network anomalies on MDC | **SUPPORTED** | Default ROC 0.7402; multi-seed 0.7529±0.023; HPO 0.8138 | ROC/PR fig; main results table | Single dataset |
| C2 | Preprocessing is leakage-free (session split, benign-only scaler) | **SUPPORTED** | Preprocess gates PASS; manifest `leakage_free=true` | Methods §preprocess | — |
| C3 | Holdout shows distribution shift vs benign train (PSI/KS) | **SUPPORTED** | Drift monitor: many features PSI>0.25 | PSI figure / §14 | Shift expected by split design |
| C4 | System supports simulated streaming replay with logging | **SUPPORTED** | `drift_aware_log.csv`; orchestration loop | Score/PSI timelines | Not live deployment |
| C5 | CPU inference is deployment-feasible (<100 ms/window) | **SUPPORTED** | ~3.36 ms/window | Latency table | Hardware-specific |
| C6 | Isolation Forest is a weaker but valid unsupervised baseline | **SUPPORTED** | IF ROC ~0.674 < AE 0.7402 | Baseline table | One classical baseline |
| C7 | Adaptive thresholding improves monitoring under drift | **CONDITIONAL** | Only if `claim_ok=True` in `matched_policy_comparison.json` after v2 freeze | fig07 / fig07b | May be FPR↔recall trade-off |
| C8 | Incremental benign fine-tune improves drift-period detection | **NOT SUPPORTED** | AUC dropped (~−0.06) | fig10 finetune | Catastrophic forgetting |
| C9 | Live real-time temporal deployment is validated | **NOT SUPPORTED** | N/A | — | Simulated replay only |
| C10 | v1 stream “92% FPR reduction” is a valid result | **INVALID** | Threshold double-invert bug | — | **Do not cite** |
| C11 | Contractive reg / early-stop ablations quantified | **PENDING GPU** | `metrics_ablation_vnext.json` after `RUN_ABLATIONS=True` | Table A | Smoke mode not usable |
| C12 | Per-attack results use official MDC CVE/scenario names | **SUPPORTED (code)** | `mdc_label_map.py` + §19 export | `eval_multiclass.json` | Re-run §19 for named plots |

---

## 3. Required wording snippets

### Abstract (template)

> We present a leakage-aware Transformer autoencoder for container network anomaly detection on the MDC dataset, achieving **0.7529 ± 0.023** ROC-AUC across seeds (best HPO **0.8138**; default reference **0.7402**). We evaluate drift monitoring (PSI/KS) and drift-aware mechanisms under **simulated streaming replay**. Benign-only fine-tuning degraded drift-slice AUC, indicating catastrophic forgetting; adaptive thresholding is reported at matched operating points.

### Methods — threshold

> The primary operating point is the validation-tuned F1-optimal threshold from the model notebook (`threshold ≈ −0.1434`), applied to scores that are already oriented higher=anomalous. Stream evaluation loads this threshold from `metrics_vnext.json` and does not recompute it under a different invert convention.

### Methods — streaming (Task 6)

> Windows are replayed in timestamp order when `ts_test` is present in `windows_vnext.npz`; otherwise container-lexsort is used as a pseudo-chronological proxy and stated as a limitation.

### Discussion — fine-tune

> Unsupervised benign fine-tuning reduced drift-period ROC-AUC, so we do **not** claim successful online adaptation. Threshold recalibration is the safer operational response on this dataset.

---

## 4. Evidence pack checklist (Task 4 freeze)

Freeze only when `freeze_manifest_v2.json` reports `freeze_ready=true`:

- [ ] `THRESHOLD INTEGRITY: PASS`
- [ ] `ALERT_INVERT=False`
- [ ] Fixed stream FPR < 0.50
- [ ] `matched_policy_comparison.json` present
- [ ] Required figures present
- [ ] Executed notebook saved as `mdc_drift_aware_output_v2.ipynb`
- [ ] Legacy v1 notebook retained but marked do-not-cite for adaptive FPR

---

## 5. Task 6 timestamp dependency

| Artifact | Notebook | Keys | Used by |
|----------|----------|------|---------|
| `windows_vnext.npz` | `mdc_preprocess_vNext.ipynb` | `ts_val`, `ts_test` | **Drift-aware streaming** |
| `windows_vnext_mc.npz` | `mdc_preprocess_vNext_mc.ipynb` | `ts_val`, `ts_test` + multiclass | Model §19 + future temporal+MC |

Re-run **both** preprocess notebooks after the Task 6 code update, then re-run drift-aware and confirm `ordering_mode=timestamp`.
