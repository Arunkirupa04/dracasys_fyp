# 04 — Final Model, Results, and Reproducibility

**Canonical metrics source:** `module4/notebook/final/output-metrics/model_vnext_runs/metrics_vnext.json`  
**Canonical code:** `module4/notebook/final/kaggle-source/`  
**Executed notebooks:** `module4/notebook/final/kaggle-output/`  
**Related:** [01](01_project_overview_and_architecture.md) · [02](02_data_pipeline_and_methodology.md) · [03](03_experiments_and_research_history.md) · [05](05_model_integration_and_developer_guide.md)

This file documents the **final/ freeze only** unless a subsection is explicitly labelled as a previous experiment / reference.

---

## 1. Final architecture

| Item | Value | Source |
|------|-------|--------|
| Class | `SequenceBottleneckAE (no temporal mean-pool)` | `metrics_vnext.json` → `architecture` |
| Input shape | `(B, 10, 163)` | `train_shapes` / manifest |
| Default bottleneck | 24 | Checkpoint `config` (model notebook) |
| HPO bottleneck | 16 | `hpo_best_params` |
| Encoder / decoder layers (default) | 2 / 2 | Model notebook config |
| HPO encoder / decoder | 3 / 2 | `hpo_best_params` |
| `d_model` / heads / ff (default) | 64 / 4 / 256 | Model notebook |
| Training data | Benign-only `X_train` | Design + shapes |
| Created at (metrics) | `2026-07-25T20:10:42Z` | `metrics_vnext.json` |

### 1.1 Training recipe (default path)

| Hyperparameter | Value | Source |
|----------------|-------|--------|
| Seeds (multi-seed) | 42, 7, 1337 | Model notebook |
| Batch size | 32 | Model notebook |
| Max epochs | 80 | Model notebook |
| LR / weight decay | 5e-4 / 1e-4 | Model notebook |
| Dropout | 0.15 | Model notebook |
| Grad clip | 1.0 | Model notebook |
| Denoising noise std | 0.03 | Model notebook |
| Contractive λ | 1e-3 | Model notebook |
| AUC early-stop patience / min epoch | 3 / 8 | Model notebook |

### 1.2 HPO best params (this freeze)

Source: `metrics_vnext.json` → `hpo_best_params`

| Param | Value |
|-------|------:|
| `d_model` | 64 |
| `bottleneck_dim` | 16 |
| `num_enc_layers` | 3 |
| `num_dec_layers` | 2 |
| `dim_ff` | 192 |
| `dropout` | 0.16059582234952724 |
| `lr` | 0.00010739354622074816 |
| `noise_std` | 0.010435561573088638 |
| `contractive_lambda` | 0.0022593854685372225 |

Optuna trial count in source config: 12. Exact trial-by-trial log in freeze dump: **Not found** as a separate artifact.

---

## 2. Evaluation methodology

1. Train on benign windows only.
2. Score val/test with `score_with_protocol()`:
   - Compute raw scores with `invert=False`
   - If `AUTO_SCORE_FLIP`, choose invert when flipped val AUC is higher
   - Optionally fit `feat_std` on benign-val errors when AUC gate passes
3. Fit thresholds on **validation** scores.
4. **Primary threshold:** `f1_optimal` (`primary_threshold` in metrics JSON).
5. Report test metrics at each threshold family; cite `f1_optimal` as primary.
6. Multi-seed: retrain with seeds `[42,7,1337]`, report mean±std.
7. HPO: Optuna search → retrain best → `test_hpo` block.
8. Drift-aware: replay test windows in **timestamp** order with fixed offline thr; evaluate adaptive under matched-policy gates.

Frozen flags: `auto_score_flip: true`, `primary_threshold: "f1_optimal"`.

---

## 3. Final metrics — default configuration

**Experiment label:** Final freeze — vNext default (`f1_optimal`)  
**Source:** `module4/notebook/final/output-metrics/model_vnext_runs/metrics_vnext.json` → `test_default.f1_optimal`

| Metric | Value | What it measures | Interpretation in this run |
|--------|------:|------------------|----------------------------|
| ROC-AUC | 0.6889160636111856 | Ranking quality attack vs benign | Moderate separation above chance |
| PR-AUC | 0.5086027803151792 | Precision–recall under imbalance | Modest; attack rate ~38% |
| F1 | 0.6603424340583064 | Harmonic precision/recall at thr | Primary operating-point quality |
| MCC | 0.4233525383108315 | Correlation of preds vs labels | Moderate positive association |
| Balanced accuracy | 0.7170249103393702 | Mean of TPR and TNR | Reasonable balance |
| Precision | 0.6062022090059473 | TP / predicted positives | ~61% of alerts true |
| Recall | 0.7251016260162602 | TP / actual attacks | ~73% attacks caught |
| FPR | 0.2910518053375196 | FP / actual benign | High false-alarm rate at F1-opt |
| FNR | 0.27489837398373984 | Miss rate | ~27% attacks missed |
| Threshold | −0.19924625754356384 | Decision boundary on saved score space | Negative due to score invert |

**Confusion matrix @ default F1-opt:** TP=1427, FP=927, TN=2258, FN=541.

### 3.1 Other thresholds (same default model scores)

Same ROC-AUC (score ranking unchanged); operating points differ.

| Threshold rule | Threshold | F1 | MCC | Recall | FPR |
|----------------|----------:|---:|----:|-------:|----:|
| youden_j | −0.192682 | 0.660697 | 0.428521 | 0.712398 | 0.274411 |
| p95_benign | −0.111698 | 0.013189 | −0.105690 | 0.007114 | 0.044270 |
| p99_benign | −0.078027 | 0.002996 | −0.050415 | 0.001524 | 0.010047 |
| fbeta_0.5 | −0.189202 | 0.661754 | 0.432624 | 0.707317 | 0.265934 |
| **f1_optimal (primary)** | **−0.199246** | **0.660342** | **0.423353** | **0.725102** | **0.291052** |

---

## 4. Final metrics — multi-seed robustness

**Experiment label:** Final freeze — multi-seed mean±std  
**Source:** `metrics_vnext.json` → `multi_seed`

| Metric | Mean | Std |
|--------|-----:|----:|
| ROC-AUC | 0.7261136849987024 | 0.03024798571831183 |
| PR-AUC | 0.5701745228920215 | 0.06759783813699523 |
| F1 | 0.7257865324748707 | 0.02140209966728127 |
| MCC | 0.5592444779510907 | 0.04989553361843328 |
| Balanced accuracy | 0.7774803448583937 | 0.018172205411595695 |
| FPR | 0.16556776556776556 | 0.052466380684461154 |
| FNR | 0.27947154471544716 | 0.016244275775554644 |

**Meaning:** Across three seeds, mean ROC-AUC (~0.726) exceeds the single default seed run (~0.689), indicating seed sensitivity; ±0.030 is non-trivial variance.

**Do not confuse with** historical lock `0.7529±0.023` (lat / claim matrix era).

---

## 5. Final metrics — HPO / optimised configuration

**Experiment label:** Final freeze — HPO best  
**Source:** `metrics_vnext.json` → `test_hpo`

| Metric | Value |
|--------|------:|
| ROC-AUC | 0.8446074395987289 |
| PR-AUC | 0.7075289458954636 |
| F1 | 0.8030176026823135 |
| MCC | 0.7080336529836047 |
| Balanced accuracy | 0.8377759218133782 |
| Precision | 0.8919925512104283 |
| Recall | 0.7301829268292683 |
| FPR | 0.054631083202511775 |
| FNR | 0.2698170731707317 |
| Threshold | −0.3474787175655365 |
| TP / FP / TN / FN | 1437 / 174 / 3011 / 531 |

**Meaning:** Best-case tuned detector in this freeze; substantially lower FPR than default F1-opt. Report as **HPO best-case**, not as the integrity baseline used by the drift stream (drift uses default `f1_optimal` thr from `test_default`).

Checkpoint stores HPO weights under `model_best` inside `checkpoint_vnext.pt` (see [05](05_model_integration_and_developer_guide.md)).

---

## 6. Classical and deep baselines (final freeze)

**Source:** `module4/notebook/final/output-metrics/deep_baseline_comparison/deep_baseline_comparison.json`

| Config | ROC-AUC | PR-AUC | F1 | MCC | Precision | Recall | FPR | Source field |
|--------|--------:|-------:|---:|----:|----------:|-------:|----:|--------------|
| Dense AE | 0.662446 | 0.462948 | 0.626228 | 0.343583 | 0.540162 | 0.744919 | 0.391837 | `dense_ae` / table |
| Isolation Forest (mean_max) | 0.674149 | 0.446369 | 0.710387 | 0.503425 | 0.626553 | 0.820122 | 0.302041 | table + drift `baseline_comparison.json` |
| vNext default | 0.688916 | 0.508603 | 0.660342 | 0.423353 | 0.606202 | 0.725102 | 0.291052 | matches metrics |
| vNext HPO | 0.844607 | 0.707529 | 0.803018 | 0.708034 | 0.891993 | 0.730183 | 0.054631 | matches metrics |

**Protocol note (JSON):** `same_windows_vnext_npz: true`, `threshold_rule: f1_optimal_on_val`, `auto_score_flip: true`.

**Unified analysis CSV:** `mdc_analysis_outputs/unified_results_table.csv` (duplicates the above rows).

IF flatten sensitivity (higher ROC in some drift outputs): present as sensitivity analysis in drift baseline JSON — **not** the primary IF row in the deep comparison table. Exact flatten row values: see `baseline_comparison.json` (secondary).

---

## 7. Statistical / systems metrics (final freeze)

**Source:** `drift_aware_outputs/stats_report.json`

| Metric | Value |
|--------|------:|
| Bootstrap ROC-AUC mean | 0.6890674840886751 |
| Bootstrap ROC-AUC CI | [0.6735404468735563, 0.70514050287927] |
| Bootstrap MCC mean | 0.4232011545823313 |
| Bootstrap MCC CI | [0.39918014080355035, 0.45033354145011417] |
| CPU latency median (ms) | 2.5313620000133596 |

Bootstrap aligns with default offline ROC (~0.689), not HPO.

---

## 8. Drift-aware outcomes (final freeze)

### 8.1 Stream setup

**Source:** `drift_aware_summary.json`

| Item | Value |
|------|-------|
| Version | `drift_aware_v2` |
| Ordering | `timestamp` |
| n_windows | 5153 |
| Note | Simulated streaming replay — not live deployment |
| Fixed thr provenance | `metrics_vnext.json:test_default.f1_optimal.threshold` = −0.199246… |
| Threshold integrity | PASS on test_set and stream_order (`ok: true`) |
| Adaptive α / buffer / update_every | 2.5 / 500 / 20 |

### 8.2 Adaptive claim gate

**Sources:** `matched_policy_comparison.json`, `freeze_manifest_v2.json`, `mdc_analysis_outputs/drift_aware_findings_summary.json`

| Item | Value |
|------|-------|
| `claim_adaptive_ok` / `claim_ok` | **false** |
| Last-half fixed | FPR 0.4651, recall 0.7329, F1 0.7783 |
| Last-half adaptive | FPR 0.0, recall 0.0, F1 0.0 |
| Natural FPR reduction | 0.4651 (with catastrophic recall loss) |
| Interpretation | `tradeoff_or_negative — report F1/recall/MCC; do not claim FPR reduction alone` |

**Do not cite** adaptive as an improvement in this freeze.

### 8.3 Fine-tune

**Source:** `finetune_report.json`

| Item | Value |
|------|-------|
| Drift slice frac | 0.3 |
| Benign source | `stable_slice_fallback` |
| Benign n / steps / lr | 200 / 50 / 1e-5 |
| auc_before / after / delta | **NaN** |
| interpretation field | `positive_delta` (string present despite NaN — **do not treat as numeric proof**) |
| freeze `do_not_cite` | includes `"fine-tune improves detection"` |

### 8.4 Freeze manifest caveats

**Source:** `freeze_manifest_v2.json`

| Gate | Value |
|------|-------|
| `freeze_ready` | true |
| `claim_adaptive_ok` | false |
| `thr_fixed_near_offline` | false (despite same thr provenance — gate inconsistency to note) |
| `ordering_is_timestamp` | true |
| `headline_metrics_locked` | **0.7402 / 0.7529±0.023 / 0.8138** from `docs/claim_evidence_matrix.md` |

**Conflict:** Locked headlines inside the freeze manifest **do not match** this freeze’s `metrics_vnext.json`. For research-record purposes, cite §3–§5 of **this file** for final offline numbers.

`do_not_cite` list also includes: v1 FPR reduction ~0.92; v1 fixed stream FPR ~0.97; live temporal deployment validated.

---

## 9. Previous experiment / reference metrics (separate — do not mix)

| Label | Default ROC | Multi-seed | HPO | Where |
|-------|------------:|-----------:|----:|-------|
| REFERENCE lat | 0.7402 | 0.7529±0.023 | 0.8138 | `output-legacy-experiments/mdc_model_vNext_lat_output.ipynb` |
| Jul-24 archive | 0.7042 | (printed 0.7529±0.023) | (printed 0.8138) | `output-legacy-2026-07-24/` |
| Exp A | 0.7163 | — | — | `mdc_model_v3_output_4.ipynb` |
| 25-07 buggy | 0.7316 default; **invalid** multi-seed/HPO | 0.2739 | 0.1554 | `output-legacy-kaggle-2026-07-25/...25-07.ipynb` |
| **Final freeze** | **0.6889** | **0.7261±0.030** | **0.8446** | **`final/output-metrics/.../metrics_vnext.json`** |

---

## 10. Artifacts checklist (final freeze)

### Present under `output-metrics/`

| Path | Role |
|------|------|
| `windows_vnext_processed/windows_vnext.npz` | Data |
| `windows_vnext_processed/windows_vnext_mc.npz` | Identical MC alias |
| `windows_vnext_processed/drift_baseline_vnext.npz` | PSI baseline |
| `windows_vnext_processed/preproc_vnext.pkl` | Provenance scalers |
| `windows_vnext_processed/manifest_vnext.json` | Metadata |
| `windows_vnext_processed/mdc_label_map.json` | Labels |
| `model_vnext_runs/checkpoint_vnext.pt` | Model weights |
| `model_vnext_runs/scores_vnext.npz` | Saved scores |
| `model_vnext_runs/metrics_vnext.json` | Metrics |
| `model_vnext_runs/drift_report_vnext.npz` | Feature PSI/KS |
| `model_vnext_runs/eval_vnext.png` | Eval figure |
| `drift_aware_outputs/*` | Stream reports + figures |
| `deep_baseline_comparison/*` | Baseline table |
| `mdc_analysis_outputs/*` | Synthesis |

### Not found in final model dump

| Expected by some cells/docs | Status |
|-----------------------------|--------|
| `metrics_ablation_vnext.json` | **Not found** |
| `eval_multiclass.json` | **Not found** |
| `per_attack_type_vs_timeline.csv` | **Not found** (analysis skipped without multiclass eval) |
| Separate `checkpoint_hpo.pt` | **Not found** (HPO inside `checkpoint_vnext.pt` as `model_best`) |
| Packed `.zip` files beside unpacked dirs | **Not found** under `output-metrics/` (unpacked contents only) |
| `requirements.txt` with pinned versions | **Not found** |

---

## 11. Exact execution procedure (reproduce final pipeline)

Platform: **Kaggle** (final track). Colab/Drive is historical.

```text
Step 1 → Create Kaggle notebook from
         module4/notebook/final/kaggle-source/mdc_preprocess_vNext_mc_kaggle.ipynb
         Attach MDC dataset yigitsever/misuse-detection-in-containers-dataset
         Run All → download / keep windows_vnext_processed.zip (Output)

Step 2 → Notebook from mdc_model_vNext_kaggle.ipynb
         Add Data → Step 1 notebook output (or uploaded zip dataset)
         Run All → model_vnext_runs.zip
         Verify metrics_vnext.json exists

Step 3 → Notebook from mdc_drift_aware_kaggle.ipynb
         Add Data → Step 1 + Step 2 outputs
         Run All → drift_aware_outputs.zip
         Verify freeze_manifest_v2.json gates; expect claim_adaptive_ok=false

Step 4 → Notebook from mdc_baselines_kaggle.ipynb
         Add Data → Step 1 (+ Step 2 metrics, Step 3 baseline_comparison if available)
         Run All → deep_baseline_comparison.zip

Step 5 (optional) → mdc_analysis_kaggle.ipynb
         Add Data → all four outputs
         Run All → mdc_analysis_outputs.zip
```

**GPU:** Recommended for model / baselines / drift fine-tune cells (proposal: RTX 3060-class). Exact GPU used in the frozen Kaggle run: **Not verified** from artifacts alone.

**Seeds / randomness:** `RANDOM_STATE=42` in preprocess; model seeds as above. Bit-identical re-runs across platforms: **Not verified**.

---

## 12. Reproducibility gaps and known limitations

1. No pinned dependency lockfile.
2. Metric era conflict (claim matrix / freeze_manifest locked headlines vs `metrics_vnext.json`).
3. Ablation and multiclass eval JSON absent from this freeze dump.
4. Adaptive and fine-tune not claimable as improvements.
5. Stream is simulation, not live deployment.
6. `preproc_vnext.pkl` unused by downstream notebooks.
7. Default ROC in final freeze is **below** Exp A (0.7163) and lat REFERENCE (0.7402) — must be discussed honestly if comparing eras (ISS-05 pattern).

---

## 13. Metric meaning quick reference

| Metric | Relevance to this research |
|--------|----------------------------|
| ROC-AUC | Primary ranking metric for unsupervised detector quality |
| PR-AUC | More informative under class imbalance than ROC alone |
| F1 @ f1_optimal | Operating-point quality used as primary threshold rule |
| MCC | Robust single-number summary with imbalance |
| FPR / Recall | Security trade-off; central to adaptive claim gates |
| Multi-seed std | Stability / reproducibility of training |
| HPO metrics | Optimistic tuned ceiling — label clearly |
| Bootstrap CI | Uncertainty on default operating metrics |
| Latency ms | Feasibility for near-real-time scoring |
