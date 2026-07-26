# 04 — Final Model, Results & Reproducibility

> **Scope**: the currently-deployed model only. Every number in this file comes from the seed-42 model trained by `module3_pipeline/train_vae.ipynb` and evaluated by `module3_pipeline/vae_eval.ipynb`, unless a table explicitly says otherwise (multi-seed / baseline / HPO tables are clearly separated per Section 4). Do not blend these with any number from `03`'s experiment history.

---

## 1. Final architecture

```text
Input (26, whitened PCA) -> Linear(26,64) -> ReLU -> Linear(64,32) -> ReLU
                          -> [fc_mu: Linear(32,32)]  [fc_logvar: Linear(32,32), clamped to [-10,10]]
                          -> reparameterize (mu + exp(0.5*logvar)*eps, training only)
                          -> Linear(32,32) -> ReLU -> Linear(32,64) -> ReLU -> Linear(64,26)  [reconstruction]
```

| Hyperparameter | Value | Chosen by |
|---|---|---|
| `hidden1` | 64 | fixed (not ablated) |
| `hidden2` | 32 | fixed (not ablated) |
| `latent_dim` | 32 | ablation over {8, 16, 32} at fixed `beta_max=0.01`, selected by lowest val loss — 8→16 gave +25% val-loss improvement, 16→32 gave only +4% (diminishing but still positive) |
| `beta_max` (KL weight) | 0.01 | search over {1.0, 0.1, 0.01, 0.001} at fixed `latent_dim=16`; both 1.0 and 0.1 caused full posterior collapse (final KL < 0.05) under the corrected KL formula; 0.01 is the largest non-collapsing value |
| KL warmup | 10 epochs, linear ramp 0→`beta_max` | fixed |
| Weight init | Kaiming/He normal (`nonlinearity='relu'`), zero bias | fixed |
| Optimizer | Adam, `lr=1e-3` | fixed |
| Batch size | 512 | fixed |
| Max epochs / patience | 300 / 20 (early stopping on val loss) | fixed |
| Input clip | `±20` (post-whitening safety margin) | fixed |
| Parameters | **10,778** | measured directly (`sum(p.numel() for p in model.parameters())`) |
| Model file size | **46.7 KB** (`vae_cc1.pt`, state_dict only) | measured directly |

**Anomaly score** = `mean((decoder(mu(x)) - x)^2)` — reconstruction MSE using the **deterministic encoder mean** (`mu`), not a stochastic sample. This is used at every evaluation/inference step; the reparameterization trick's random sampling is used only during training.

**KL-divergence formula** (the corrected form — see `03` for the bug this replaced):
```python
kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
```
Sum over the latent dimension, then mean over the batch — **not** a flat `torch.mean()` over both.

---

## 2. Training process (exact, reproducible)

Source: `module3_pipeline/train_vae.ipynb`, run with the `python39-pytorch` Jupyter kernel (**required** — the default `python3` kernel resolves to a Python 3.11 install that crashes on `import torch` with `WinError 1114` on this machine).

1. Load `data/processed/windows_cc1/X_{cc1_train,cc1_val,cc1_test,drift_sc1,drift_sc2,drift_cc2}.npy` (+ corresponding `y_*.npy`).
2. **Data-sanity guard, fails fast**: assert consistent dims across splits, no NaN/Inf anywhere, and `cc1_train`'s labels are 100% zero. Clip all arrays to `±20`.
3. Beta search: train 4 candidates (`beta_max ∈ {1.0, 0.1, 0.01, 0.001}`) at `latent_dim=16`, 40 epochs, patience 8. Pick the largest `beta_max` whose final KL exceeds 0.05 (non-collapsed).
4. Latent-dim ablation: train `latent_dim ∈ {8, 16, 32}` at the chosen `beta_max`, same short budget. Pick lowest val loss.
5. Full training: `latent_dim=32`, `beta_max=0.01`, up to 300 epochs, patience 20, restoring the best-val-loss checkpoint at the end (this run did **not** early-stop — it used all 300 epochs, with val loss still improving at epoch 300: 0.2335 → the reported best of 0.2320 was actually reached at a slightly earlier epoch within the run, restored via the "keep best state_dict" mechanism).
6. Compute `mu_train`/`sigma_train` (train-set reconstruction-MSE mean/std) — the reference statistics every downstream threshold is built from.
7. Save `models/vae_cc1.pt` (state_dict) and `models/vae_cc1_meta.pkl` (architecture config + `mu_train`/`sigma_train` + full beta-search/ablation results, for provenance).

**Training-time results actually observed** (verified from the saved `vae_cc1_meta.pkl` and the notebook's cell outputs):

| Quantity | Value |
|---|---|
| Final train KL | 15.9926 (non-trivial — confirms no collapse) |
| Final best val loss | 0.2320 |
| `mu_train` (train MSE mean) | 0.022176 |
| `sigma_train` (train MSE std) | 0.032012 |
| `cc1_val` MSE mean/std | 0.02305 / 0.01954 (close to train — both 100% normal, as expected) |
| Anomaly/normal MSE ratio (directional sanity check only, not a formal metric) | cc1_test 16.80x, drift_sc1 4.48x, drift_sc2 3.78x, drift_cc2 6.29x |

---

## 3. Evaluation methodology

Source: `module3_pipeline/vae_eval.ipynb`. Every threshold used for a reported precision/recall/F1 number is calibrated **exclusively from `cc1_val`** (100% normal — zero anomaly-label leakage into threshold selection) or from `mu_train`/`sigma_train` directly. **Test and drift-set labels are used only to compute the final reported metrics, never to choose a threshold or hyperparameter.**

Three thresholds are computed and reported side-by-side (not just one, to show sensitivity to this choice):
- `val_p99` — 99th percentile of `cc1_val`'s MSE distribution (**the deployed/primary operating point**)
- `k=2` — `mu_train + 2·sigma_train`
- `k=3` — `mu_train + 3·sigma_train`

**Primary metric: PR-AUC (average precision), not ROC-AUC.** Justification (directly evidenced, not just asserted): under this project's severe class imbalance (0.57% anomaly rate on `cc1_test`, 0.94% on `drift_cc2`), ROC-AUC can look deceptively strong while precision-recall behavior is poor — e.g. `drift_sc1` (a set excluded from final reporting, but used as evidence for this exact point) had ROC-AUC=0.8515 (looks good) alongside AUC-PR of only 0.0181 (barely above what a random classifier would score at that base rate) — a textbook case of ROC-AUC masking a genuinely weak precision-recall trade-off. PR-AUC does not have this blind spot.

Also computed and saved (see full definitions/interpretation in the notebook itself):
- **Oracle-ceiling F1** — best possible F1 at *any* threshold for this exact trained model (via `sklearn.metrics.precision_recall_curve`), used purely as a diagnostic to separate "the threshold is leaving performance on the table" from "the model itself is the limit."
- **Per-fault-type recall** (`cpu`, `memory`, `pod-failure` separately).
- **Detection delay** — wall-clock time from a fault's groundtruth onset to the first flagged window, computed only for `drift_cc2` (the set with available, currently-in-scope groundtruth timing).
- **KS-test** comparing `drift_cc2`'s normal-window MSE distribution against `cc1_train`'s, to empirically characterize drift (not to trigger anything in this notebook — that consumption happens in `incremental_learning.ipynb`).

---

## 4. Final metrics — clearly separated by configuration

### 4.1 Default configuration (single seed=42, `val_p99` threshold) — **the deployed operating point**

| Set | PR-AUC (primary) | ROC-AUC | F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|---|
| `cc1_test` (in-distribution) | **0.6014** | 0.8763 | 0.618 | 0.630 | 0.605 | 0.0021 |
| `drift_cc2` (sole reported drift set) | **0.4089** | 0.8812 | 0.276 | 0.168 | 0.772 | 0.0362 |

At `k=3` instead of `val_p99` (higher precision, lower recall trade-off — reported for completeness, **not** the deployed threshold): `cc1_test` F1=0.671 (precision 0.808, recall 0.574); `drift_cc2` F1=0.355 (precision 0.234, recall 0.738).

**Oracle-ceiling comparison** (same seed-42 model, best-possible threshold):

| Set | Deployed F1 (`val_p99`) | Oracle-best F1 | Gap | Oracle precision | Oracle recall |
|---|---|---|---|---|---|
| `cc1_test` | 0.6175 | **0.6857** | 0.0682 | 0.878 | 0.562 |
| `drift_cc2` | 0.2757 | **0.5147** | 0.2391 | 0.520 | 0.510 |

**Per-fault-type recall** (at `val_p99`):

| Set | cpu | memory | pod-failure |
|---|---|---|---|
| `cc1_test` | 0.580 (n=88) | 0.763 (n=76) | 0.500 (n=92) |
| `drift_cc2` | 0.908 (n=207) | 0.713 (n=414) | 0.737 (n=99) |

**Detection delay** (`drift_cc2` only, `val_p99` threshold): **12/12 groundtruth events detected (100%)**. Mean delay 88s (1.5 min), median 75s (1.2 min). By fault type: cpu 75.0s mean (n=3), memory 80.0s mean (n=6), pod-failure 115.0s mean (n=3).

**KS-test drift characterization**: `drift_cc2` vs. `cc1_train` — KS statistic 0.4097, p≈0.00 → statistically significant distributional shift confirmed (not merely assumed from context).

### 4.2 Multi-seed robustness (3 seeds: 42, 7, 123 — `experiments/multi_seed_variance.ipynb`)

| Set | AUC-ROC (mean±std) | AUC-PR (mean±std) | F1 (mean±std) |
|---|---|---|---|
| `cc1_test` | 0.8739 ± 0.0112 | 0.5859 ± 0.0173 | 0.5448 ± 0.0662 |
| `drift_cc2` | 0.8900 ± 0.0062 | 0.4206 ± 0.0087 | 0.2047 ± 0.0505 |

Extended to 5 seeds for oracle-ceiling specifically (`experiments/multi_seed_oracle_ceiling.ipynb`; see `03` Phase 6b for the full table): oracle F1 on `cc1_test` ranges **0.626–0.699** across seeds {7, 123, 999, 42, 2024}. Seed 42 (deployed) is not the single best seed by this metric but is the only fully leak-free selection (lowest `cc1_val` loss) and sits within 1.9 points of the best.

**How to read these two tables together**: the "default configuration" table (4.1) is what the deployed model actually does. The multi-seed tables (4.2) answer "how much would this number move if retrained" — they are not a better or more official estimate of the deployed model's performance, they are a variance/robustness characterization of the *method*.

### 4.3 Classical baselines (`module3_pipeline/baseline_comparison.ipynb`, single-seed VAE)

| Set | Method | AUC-ROC | AUC-PR | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| `cc1_test` | Gaussian (whitened-distance, 0 learned params) | 0.8798 | **0.6554** | **0.664** | 0.694 | 0.637 |
| `cc1_test` | Isolation Forest | 0.7584 | 0.1135 | 0.171 | 0.138 | 0.227 |
| `cc1_test` | **VAE** | 0.8763 | 0.6014 | 0.618 | 0.630 | 0.605 |
| `drift_cc2` | Gaussian | 0.8755 | 0.1430 | 0.161 | 0.090 | 0.739 |
| `drift_cc2` | Isolation Forest | 0.8348 | 0.2507 | 0.220 | 0.146 | 0.443 |
| `drift_cc2` | **VAE** | **0.8812** | **0.4089** | **0.276** | 0.168 | 0.772 |

**Reading this honestly** (see `03` Phase 5 and `results_baseline_discussion.md` for the full discussion): the VAE does **not** beat the simplest possible baseline in-distribution — a zero-parameter Gaussian distance measure in the same whitened PCA space wins on `cc1_test`. The VAE's advantage is concentrated, consistently, on `drift_cc2` — the genuinely separate deployment. The defensible claim for this project is *"the VAE earns its complexity specifically under drift, not in-distribution,"* not an unqualified superiority claim.

### 4.4 Ablation study — VAE alone vs. +Adaptive Threshold vs. Full Model (`module3_pipeline/final_comparison.ipynb`)

| Set | Config | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| `cc1_test` | VAE alone | 0.6014 | 0.8763 | 0.618 | 0.630 | 0.605 |
| `cc1_test` | + Adaptive Threshold | 0.6014 | 0.8763 | 0.623 | 0.676 | 0.578 |
| `cc1_test` | Full Model (+ incremental learning) | 0.6052 | 0.8851 | 0.634 | 0.701 | 0.578 |
| `drift_cc2` | VAE alone | 0.4089 | 0.8812 | 0.276 | 0.168 | 0.772 |
| `drift_cc2` | + Adaptive Threshold | 0.4089 | 0.8812 | 0.347 | 0.223 | 0.786 |
| `drift_cc2` | Full Model (+ incremental learning) | **0.4218** | **0.9020** | **0.391** | 0.262 | 0.768 |

**A conceptual point this table demonstrates directly, not just asserts**: "VAE alone" and "+Adaptive Threshold" have **identical** PR-AUC/ROC-AUC per set — thresholding cannot change the underlying score ranking, only where you cut it. Only "Full Model" (incremental learning) moves PR-AUC/ROC-AUC, because it is the only stage that changes the reconstruction-error scores themselves.

**Drift-adaptation retention** (`drift_cc2`, relative to `cc1_test`'s "before drift" baseline):

| Metric | Before drift (`cc1_test`) | After drift alone | After adaptation (Full Model) | Retention, drift only | Retention, adapted |
|---|---|---|---|---|---|
| AUC-PR | 0.6014 | 0.4089 | 0.4218 | 68.0% | 70.1% |
| F1 | 0.6175 | 0.2757 | 0.3910 | 44.6% | 63.3% |
| Recall | 0.6055 | 0.7722 | 0.7681 | 127.5% | 126.9% |

F1 retention improves substantially with adaptation (44.6% → 63.3%) — the clearest single number for the "drift-aware" claim.

### 4.5 Incremental-learning control vs. treatment (`module3_pipeline/incremental_learning.ipynb`)

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Control (incremental learning OFF) | 0.223 | 0.786 | 0.347 |
| Treatment (incremental learning ON) | 0.261 | 0.771 | **0.390** |

15 fine-tune events fired across the 76,977-window `drift_cc2` stream (every 5,000-window check, all 15 checks triggered) — see `03` Phase 4 for the caveat that this means the model never stopped registering as "drifted" relative to its fixed reference, an open question this project did not resolve further.

### 4.6 "Lightweight and real-time" — measured, not asserted

| Quantity | Value |
|---|---|
| Parameters | 10,778 |
| Model file size | 46.7 KB |
| Single-window inference latency (CPU) | 0.4525 ms |
| Batched (1000 windows) latency | 5.64 ms total (0.0056 ms/window) |
| Theoretical single-call throughput | ~2,210 windows/sec |

---

## 5. Model persistence — full artifact inventory (`models/`)

| File | Format | Contents | Produced by |
|---|---|---|---|
| `vae_cc1.pt` | PyTorch state_dict | trained VAE weights | `train_vae.ipynb` |
| `vae_cc1_meta.pkl` | pickle | `input_dim, hidden1, hidden2, latent_dim, beta_max, clip, mu_train, sigma_train`, full beta-search + ablation history | `train_vae.ipynb` |
| `cc1_pca.pkl` | **joblib** (not plain pickle — loading with `pickle.load` raises `UnpicklingError`) | fitted `PCA` object, `feature_cols`, `window_size`, `stride`, `n_components` | `windowing_pca.ipynb` |
| `cc1_scaler.pkl` | joblib | fitted `RobustScaler`, `feature_cols`, `clip` | `clean_and_split.ipynb` |
| `vae_cc1_eval.pkl` | pickle | thresholds, AUC, precision/recall/F1, oracle ceiling, per-fault recall, detection delay, KS-drift, summary table — **currently CC2-only scoped** (see `03`'s inconsistency note) | `vae_eval.ipynb` |
| `vae_cc1_adaptive_eval.pkl` | pickle | naive adaptive-threshold results (superseded; still a real dependency of `adaptive_threshold_blended.ipynb`) | `adaptive_threshold.ipynb` (experiments/) |
| `vae_cc1_adaptive_blended_eval.pkl` | pickle | blended adaptive-threshold results (deployed thresholding method) | `adaptive_threshold_blended.ipynb` |
| `incremental_learning_eval.pkl` | pickle | lightweight-validation numbers, control/treatment F1, fine-tune events | `incremental_learning.ipynb` |
| `baseline_comparison.pkl` | pickle | Gaussian / Isolation Forest / VAE comparison | `baseline_comparison.ipynb` |
| `final_comparison_results.pkl` | pickle | consolidated baseline + ablation + drift-adaptation results | `final_comparison.ipynb` |
| `multi_seed_variance.pkl` | pickle | 3-seed per-seed results + mean/std summary | `multi_seed_variance.ipynb` (experiments/) |
| `multi_seed_oracle_ceiling.pkl` | pickle | 5-seed oracle-ceiling comparison | `multi_seed_oracle_ceiling.ipynb` (experiments/) |
| `maxpool_scoring_eval.pkl` | pickle | pooling-method comparison (negative result) | `maxpool_scoring.ipynb` (experiments/) |
| various `*.png` | image | training curves, PR/ROC curves, distribution histograms, per-fault bar charts, dashboards | respective notebooks |

**Reproducibility caveat, stated plainly**: several of these `.pkl` files (`baseline_comparison.pkl`, `vae_cc1_adaptive_eval.pkl`, `vae_cc1_adaptive_blended_eval.pkl`) were last written **before** `vae_eval.ipynb`'s CC2-only rescoping and still contain `drift_sc1`/`drift_sc2` keys that the *current* `vae_cc1_eval.pkl` no longer has. Their own producing notebooks (`baseline_comparison.ipynb`, `adaptive_threshold_blended.ipynb`) would raise a `KeyError` if re-executed today without first either regenerating `vae_cc1_eval.pkl` with all 3 drift sets, or editing those notebooks' `DRIFT_SETS` lists down to `['drift_cc2']`. This is a real, present gap — not resolved as part of writing this documentation.

---

## 6. Exact reproduction procedure

```text
Prerequisite: Jupyter kernel "python39-pytorch" registered and available
              (Python 3.9 + PyTorch 2.8.0+cpu + scikit-learn + pandas/numpy/
              matplotlib/joblib). On this machine, the default python3 kernel
              is NOT usable for any notebook that imports torch.

Step 1 -> data/raw/**  (already present; AIOpsArena benchmark data)
Step 2 -> [historical: merge_and_normalize.ipynb + EDA/preprocessing.ipynb
           produced data/processed/all_cases_labeled.csv — this input file
           already exists on disk; there is no current notebook that
           regenerates it from data/raw/ directly under the CC1-only design.
           Not found / not verified: an exact single current notebook that
           reproduces all_cases_labeled.csv from raw KPI files under today's
           conventions.]
Step 3 -> module3_pipeline/clean_and_split.ipynb
          Input:  data/processed/all_cases_labeled.csv,
                  data/labeled/single_case1_labeled.csv
          Output: data/processed/{all_cases_cleaned,cc1_train,cc1_val,
                  cc1_test,drift_single_case1,drift_single_case2,
                  drift_complex_case2}.csv, models/cc1_scaler.pkl
Step 4 -> module3_pipeline/windowing_pca.ipynb
          Input:  the 6 split CSVs from Step 3
          Output: data/processed/windows_cc1/{X,y,ft}_<split>.npy,
                  models/cc1_pca.pkl
Step 5 -> module3_pipeline/train_vae.ipynb
          Input:  windows_cc1/X_{cc1_train,cc1_val}.npy
          Output: models/vae_cc1.pt, models/vae_cc1_meta.pkl
Step 6 -> module3_pipeline/vae_eval.ipynb
          Input:  models/vae_cc1.pt, vae_cc1_meta.pkl, windows_cc1/*.npy,
                  data/processed/drift_complex_case2.csv (for detection delay),
                  data/raw/Complex Case-2/Case-2/groundtruth/groundtruth.json
          Output: models/vae_cc1_eval.pkl + 6 PNG figures
Step 7 -> module3_pipeline/adaptive_threshold_blended.ipynb
          Input:  models/vae_cc1.pt, vae_cc1_meta.pkl, vae_cc1_eval.pkl,
                  vae_cc1_adaptive_eval.pkl [STALE DEPENDENCY — see Section 5]
          Output: models/vae_cc1_adaptive_blended_eval.pkl
Step 8 -> module3_pipeline/incremental_learning.ipynb
          Input:  models/vae_cc1.pt, vae_cc1_meta.pkl, vae_cc1_eval.pkl
                  (thresholds only), cc1_pca.pkl, data/processed/
                  drift_complex_case2.csv
          Output: models/incremental_learning_eval.pkl
Step 9 -> module3_pipeline/baseline_comparison.ipynb
          Input:  windows_cc1/*.npy, vae_cc1_eval.pkl [STALE DEPENDENCY]
          Output: models/baseline_comparison.pkl
Step 10-> module3_pipeline/final_comparison.ipynb
          Input:  models/vae_cc1.pt, vae_cc1_meta.pkl, cc1_pca.pkl,
                  vae_cc1_eval.pkl (thresholds only — NOT stale-dependent),
                  all 4 split CSVs
          Output: models/final_comparison_results.pkl,
                  models/final_comparison_summary.png

Execution command pattern used throughout this project (nbconvert, headless):
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=python39-pytorch <notebook>.ipynb
```

**To reproduce Step 7 and Step 9 exactly as originally run** (given the stale-dependency issue in Section 5), either: (a) temporarily regenerate `vae_cc1_eval.pkl` with `DRIFT_SETS = ['drift_sc1','drift_sc2','drift_cc2']` in `vae_eval.ipynb` before running them, then re-run `vae_eval.ipynb` again with `DRIFT_SETS = ['drift_cc2']` to restore the current scope; or (b) edit `adaptive_threshold_blended.ipynb` and `baseline_comparison.ipynb`'s own `DRIFT_SETS` lists down to `['drift_cc2']` and re-run — this would drop the historical SC1/SC2 comparison rows shown in `03`, which are only preserved by not touching those notebooks' code.

---

## 7. Known limitations of the final model (stated directly)

- Oracle-ceiling F1 on `cc1_test` (0.686) has not been pushed above 0.70 by any method tried (seed search, feature extension, alternative pooling — see `03` Phase 6). This is documented as a likely-real ceiling for this architecture/feature combination, not a gap awaiting an easy fix.
- `drift_sc1`'s oracle ceiling (0.060) is barely above its deployed F1 (0.040) — for this specific drift set, the signal itself, not the threshold, is the limiting factor (this set is excluded from the currently-reported metrics, per project scope decision, but the underlying data/windows still exist).
- Incremental learning fired at every single periodic check throughout the entire `drift_cc2` stream (15/15) — no evidence the model ever reached a stable adapted state within this evaluation window.
- Network-layer faults (`delay`, `loss`) are entirely out of scope — the model cannot, by design, detect them.
