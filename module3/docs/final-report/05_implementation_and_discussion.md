# Chapter 5 — Implementation, Discussion and Evaluation

**Module:** Module 3 — System Anomaly Detector (Drift-Aware Approach)  
**Canonical code:** `module3/module3_pipeline/`  
**Canonical metrics:** `module3/models/*.pkl` (prefer these over older narrative numbers in legacy research docs)  
**Design reference:** Chapter 4 (Figures 4.1–4.4)

---

## Part A — Implementation

## 5.1 Software Technologies and Environment

| Item | Evidence |
|------|----------|
| Language | Python 3.9 (Jupyter kernel `python39-pytorch`) |
| DL | PyTorch `2.8.0+cpu` |
| Classical ML | scikit-learn (`PCA`, `RobustScaler`, `IsolationForest`, metrics) |
| Stats | SciPy `ks_2samp` |
| Data | pandas, NumPy |
| Persistence | `torch.save` / `joblib` / `pickle` under `models/` |
| Hardware | CPU training/eval in recorded runs |

The VAE has **10,778** parameters; checkpoint size ≈ **46.7 KB**. Single-window inference latency ≈ **0.45 ms** (CPU) in `incremental_learning_eval.pkl`.

---

## 5.2 Repository Layout (Implementation Map)

```text
module3/
├── module3_pipeline/          # CANONICAL FINAL PIPELINE
│   ├── clean_and_split.ipynb
│   ├── windowing_pca.ipynb
│   ├── train_vae.ipynb
│   ├── vae_eval.ipynb
│   ├── adaptive_threshold_blended.ipynb
│   ├── incremental_learning.ipynb
│   ├── baseline_comparison.ipynb
│   ├── final_comparison.ipynb
│   └── results_baseline_discussion.md
├── models/                    # frozen metrics + weights for W=30 pipeline
├── final-model/               # VAE-alone integration package
├── module3_pipeline_v2/       # W=60 experiment (not default)
├── experiments/               # ablations / legacy / negative results
└── docs/final-report/         # this documentation set
```

---

## 5.3 Dataset Preparation

**Source:** AIOpsArena (Sock-Shop-style Kubernetes fault injection), 15 s sampling.

| Case | Role |
|------|------|
| `complex_case1` | Sole train/val/test source (~35 h) |
| `complex_case2` | Sole **reported** drift evaluation set |
| `single_case1/2` | Processed upstream; excluded from final drift tables |

**Cleaning (verified pipeline steps):**

1. Merge per-KPI CSVs → wide format (7 features).
2. Label from `groundtruth.json` (ID prefix normalisation).
3. Deduplicate `(case, cmdb_id, timestamp)` — removes periodic duplicate artifact (~25% of raw rows).
4. Mark gaps where \(\Delta t > 30\,\mathrm{s}\).
5. Relabel `delay`/`loss` → normal (**network bottlenecks not delivered**).
6. Time-split CC1 normals 70/10/20; route all CC1 anomalies to test.
7. `RobustScaler` fit on `cc1_train` only; clip ±20.

**Approx. cleaned row counts:**

| Split | Rows | Anomalies |
|-------|-----:|----------:|
| cc1_train | 156,479 | 0 |
| cc1_val | 22,356 | 0 |
| cc1_test | 44,968 | 256 |
| drift_cc2 | 77,760 | 372 |

---

## 5.4 Sliding Window and PCA Implementation

**Notebook:** `windowing_pca.ipynb`

| Parameter | Value |
|-----------|------:|
| `WINDOW_SIZE` | 30 |
| `STRIDE` | 1 |
| Gap policy | Skip windows spanning gaps |
| Flatten | \(30\times7 \rightarrow 210\) |
| PCA | `n_components=0.99`, `whiten=True`, `svd_solver='full'` |
| Output dim | **26** |
| Fit on | `cc1_train` windows only |

**Window counts (W=30):** train 154,198 (0 anom); test 44,185 (256 anom); drift_cc2 76,977 (720 anom windows after window labelling).

---

## 5.5 VAE Implementation

**Notebook:** `train_vae.ipynb`  
**Class (integration mirror):** `final-model/vae_alone_loader.py`

| Item | Implementation |
|------|----------------|
| Architecture | `26→64→32→z(32)→32→64→26` |
| `latent_dim` | 32 (ablation {8,16,32}) |
| `beta_max` | 0.01 (search avoided collapse) |
| Train optimiser | Adam lr \(10^{-3}\) |
| Batch / epochs | 512 / ≤300, patience 20 |
| Score | Mean MSE via decoder(\(\mu\)) |
| Train guard | Assert train labels all zero |

Saved artefacts: `models/vae_cc1.pt`, `models/vae_cc1_meta.pkl`.

---

## 5.6 Static Threshold and Evaluation Implementation

**Notebook:** `vae_eval.ipynb`

Thresholds computed **only** from anomaly-free `cc1_val` / train stats:

| Name | Definition | Primary? |
|------|------------|:--------:|
| `val_p99` | 99th percentile of val MSE | **Yes** (deployed) |
| `k=2` | `mu_train + 2·sigma_train` | No |
| `k=3` | `mu_train + 3·sigma_train` | No |

Metrics: PR-AUC (primary), ROC-AUC, Precision/Recall/F1/FPR, oracle-best F1, per-fault recall, detection delay, offline KS.

---

## 5.7 Blended Adaptive Threshold Implementation

**Notebook:** `adaptive_threshold_blended.ipynb`  
**Artefact:** `models/vae_cc1_adaptive_blended_eval.pkl`

```text
w = n_local / (n_local + PRIOR_STRENGTH)
threshold = blended_mean + k * blended_std
```

| Parameter | Frozen value |
|-----------|-------------:|
| `BUFFER_SIZE` | 500 |
| `PRIOR_STRENGTH` | 500 |
| `k_adaptive` | **6.2** |
| `std_cap` | ≈ 0.00955 |

Naive recent-only mean+α·std (`experiments/adaptive_threshold.ipynb`) is historical and not part of the final path.

---

## 5.8 KS Monitoring and Incremental Fine-Tune Implementation

**Notebook:** `incremental_learning.ipynb`  
**Artefact:** `models/incremental_learning_eval.pkl`

| Parameter | Value |
|-----------|------:|
| `REFIT_INTERVAL` | 5000 |
| `FT_BUFFER_SIZE` | 2000 |
| `FT_EPOCHS` | 5 |
| `FT_LR` | \(10^{-4}\) |
| Optimiser | **Adam** (`torch.optim.Adam`) |
| KS α | 0.001 |

Protocol: control stream (adaptive, no fine-tune) vs treatment (adaptive + KS-triggered fine-tune) on chronological `drift_cc2` replay.

---

## 5.9 Anomaly Decision Logic (Pseudocode)

```text
function SCORE(window_30x7):
    x ← RobustScaler.transform(window)
    x ← PCA.transform(flatten(x))        # → R^26
    μ ← Encoder_mu(x)
    x̂ ← Decoder(μ)
    return mean((x̂ − x)^2)

function DECIDE_STATIC(mse):
    return mse > val_p99

function DECIDE_BLENDED(mse, container, buffer, anchor, k):
    update buffer with recent scores
    compute blended_mean, blended_std with PRIOR_STRENGTH
    return mse > blended_mean + k * blended_std
```

---

## 5.10 Integration Package Implementation

**Files:** `final-model/vae_cc1_final_model.pt`, `vae_alone_loader.py`

Bundles scaler, PCA, VAE weights, and static threshold. API returns `{reconstruction_mse, is_anomaly, threshold_used}`. **Does not** include adaptive threshold or incremental learning.

---

## Part B — Discussion and Evaluation

## 5.11 Evaluation Methodology

| Axis | Protocol |
|------|----------|
| Learning | Unsupervised (benign-only train) |
| Primary metric | **PR-AUC** (severe imbalance) |
| Operating point | F1 / P / R / FPR at `val_p99` or blended \(k\) |
| ID set | `cc1_test` |
| Drift set | `drift_cc2` only (final reporting) |
| Threshold discipline | Fit on anomaly-free `cc1_val` / train stats — never on test labels |
| Baselines | Deterministic AE; Isolation Forest |
| Ablation | VAE alone → +adaptive → full (adaptive+IL) |

---

## 5.12 Experiments Conducted

### Experiment E1 — Deployed VAE-alone (seed 42, `val_p99`)

| Field | Detail |
|-------|--------|
| Source | `vae_cc1_eval.pkl` / `final_comparison_results.pkl` |
| Objective | Establish static detector performance |

| Set | PR-AUC | ROC-AUC | F1 | Precision | Recall | FPR |
|-----|-------:|--------:|---:|----------:|-------:|----:|
| cc1_test | **0.6014** | 0.8763 | **0.6175** | 0.630 | 0.605 | 0.0021 |
| drift_cc2 | **0.4089** | 0.8812 | **0.2757** | 0.168 | 0.772 | 0.0362 |

**Oracle-best F1:** cc1_test **0.6857**; drift_cc2 **0.5147**.

**Per-fault recall @ val_p99 (cc1_test):** cpu 0.580; memory 0.763; pod-failure 0.500.  
**Per-fault recall (drift_cc2):** cpu 0.908; memory 0.713; pod-failure 0.737.

**Detection delay (drift_cc2):** 12/12 events detected in the recorded table; example delays include 75 s (memory on frontend replicas).

**Offline KS (drift_cc2 vs train MSE):** statistic **0.4097**, \(p=0.0\).

**Demonstrates:** Core VAE detects ID faults with usable F1; drift set shows high recall but low precision under static threshold.

---

### Experiment E2 — Baseline comparison

| Field | Detail |
|-------|--------|
| Source | `final_comparison_results.pkl` → `baseline_comparison` |
| Models | AE, Isolation Forest, VAE-alone |

**cc1_test**

| Method | PR-AUC | F1 |
|--------|-------:|---:|
| Isolation Forest | 0.114 | 0.174 |
| Autoencoder | 0.413 | 0.354 |
| **VAE** | **0.601** | **0.618** |

**drift_cc2**

| Method | PR-AUC | F1 |
|--------|-------:|---:|
| Autoencoder | 0.079 | 0.159 |
| Isolation Forest | 0.252 | 0.220 |
| **VAE** | **0.409** | **0.276** |

**Demonstrates:** VAE dominates AE and IF on the primary protocol for both ID and drift PR-AUC/F1.

---

### Experiment E3 — Blended adaptive threshold

| Field | Detail |
|-------|--------|
| Source | `vae_cc1_adaptive_blended_eval.pkl` |
| Config | prior=500, buffer=500, **k=6.2** |

| Set | Static F1 | Blended F1 | Blended PR-AUC |
|-----|----------:|-----------:|---------------:|
| cc1_test | 0.618 | **0.682** | **0.6495** |
| drift_cc2 | 0.276 | **0.387** | **0.6434** |

**Demonstrates:** Adaptive thresholding substantially improves drift F1 and PR-AUC; also lifts ID F1. Supports Novelty 4.

---

### Experiment E4 — Incremental learning (control vs treatment)

| Field | Detail |
|-------|--------|
| Source | `incremental_learning_eval.pkl` |
| Set | drift_cc2 stream |
| Fine-tunes fired | **15 / 15** KS checks |

| Arm | Precision | Recall | F1 |
|-----|----------:|-------:|---:|
| Control (adaptive, no FT) | 0.223 | 0.786 | **0.347** |
| Treatment (+ KS fine-tune) | 0.261 | 0.771 | **0.390** |

**ΔF1 ≈ +0.043.**  
**Demonstrates:** Incremental fine-tuning helps, but triggers on every check—model never fully “settles.” Supports Novelty 2 partially.

---

### Experiment E5 — Final ablation (alone / adaptive / full)

| Field | Detail |
|-------|--------|
| Source | `final_comparison_results.pkl` → `ablation_study` |

| Config | cc1_test F1 | cc1_test PR-AUC | drift_cc2 F1 | drift_cc2 PR-AUC |
|--------|------------:|----------------:|-------------:|-----------------:|
| VAE alone | 0.618 | 0.601 | 0.276 | 0.409 |
| + Adaptive | **0.682** | 0.650 | **0.387** | 0.643 |
| Full (adaptive+IL) | **0.688** | 0.653 | **0.389** | 0.648 |

**Best practical research stack:** VAE + blended adaptive threshold. Full IL adds only a marginal CC2 F1 gain (**0.387 → 0.389**) beyond adaptive in this rollup.

---

### Experiment E6 — Window-size ablations (side experiments)

Canonical default remains **W=30**. Side studies (not changing the final pipeline) reported:

| Window | cc1_test PR-AUC / F1 (approx. from ablation notebooks) | Notes |
|-------:|--------------------------------------------------------|-------|
| 30 | 0.601 / 0.618 | **Deployed** |
| 45 | Higher ID PR-AUC in ablation notebook; weak drift | Side study |
| 50 | Strong ID; weak drift | Side study |
| 60 | Very strong ID (e.g. PR-AUC ~0.92 in ablation); drift mixed / often weaker after full pipeline | `module3_pipeline_v2` / experiments |

**Decision:** Keep W=30 as final for drift-aware balance and integration consistency. W=60 is an alternate research branch, not the shipped default.

---

### Experiment E7 — Multi-seed / oracle ceiling

| Study | Finding |
|-------|---------|
| Multi-seed variance (3 seeds) | cc1_test F1 ≈ **0.545±0.066**; drift_cc2 F1 ≈ **0.205±0.051** (recorded in research discussion) |
| Oracle ceiling (5 seeds) | ID oracle F1 roughly **0.63–0.70**; none broke ~0.70; seed 42 retained as leak-free pick |

**Demonstrates:** Seed sensitivity exists; seed-42 operating point is not an extreme cherry-pick beyond the oracle band.

---

### Experiment E8 — Negative results

| Experiment | Outcome |
|------------|---------|
| Extended 11 features | cc1_test F1 fell (≈0.618→0.421); drift collapsed — rejected |
| Max-pool scoring | PR-AUC collapse (e.g. ≈0.601→0.108) — rejected |
| Naive adaptive mean+3·std | ID F1 collapse in historical run — superseded by blended |
| Legacy v5 pipeline | ID F1≈0.528 / drift≈0.152 — superseded by CC1-only redesign |

---

## 5.13 Results Analysis

**Does the VAE detect anomalies?** Yes on ID data (F1≈0.618, PR-AUC≈0.601) and better than AE/IF. Memory faults show strong recall; pod-failure is harder on ID (recall 0.50 at val_p99).

**Impact of PCA / multi-metric modelling.** Joint CPU+memory windows with whitened PCA enable detection of cpu, memory, and pod-failure classes. Network bottlenecks remain undetectable by design.

**Impact of adaptive thresholding.** Largest measured gain on drift: F1 **0.276 → 0.387** and PR-AUC **0.409 → 0.643**.

**Impact of incremental learning.** Positive but smaller than adaptive thresholding in the controlled IL study (+0.043 F1); marginal in the final ablation rollup over adaptive alone.

**Changing workloads.** Offline KS confirms score-distribution shift on CC2. Adaptive thresholding is the primary mitigator; fine-tuning is secondary.

**False positives / negatives.** Static CC2 precision is low (0.168) despite high recall (0.772)—classic drift miscalibration. Blended threshold raises precision to 0.255 while keeping recall ≈0.80.

---

## 5.14 Novelty Evaluation

| # | Proposed novelty | Status | Evidence |
|---|------------------|--------|----------|
| 1 | Lightweight drift-aware AD integrating sliding window + PCA + single VAE + incremental learning | ✅ **Fully implemented** (research pipeline) | `windowing_pca`, `train_vae`, `incremental_learning`; 10.8k params; IL not in integration package |
| 2 | Efficient real-time adaptation without full retrain / heavy ensembles | ⚠️ **Partially implemented** | Sub-ms inference; Adam fine-tune improves F1; fires 15/15 times (no stable catch-up); not shipped in demo |
| 3 | Multi-metric correlation (CPU, memory, **network**) via PCA + recon error | ⚠️ **Partially implemented** | CPU+memory joint modelling verified; **network bottlenecks not delivered** |
| 4 | Adaptive thresholding reduces FP/FN under changing workloads | ✅ **Fully implemented** | Blended thr; F1/PR-AUC gains in `vae_cc1_adaptive_blended_eval.pkl` |

---

## 5.15 Comparison With Existing Work

| Aspect | Typical literature | Module 3 |
|--------|-------------------|----------|
| Model | AE / IF / ensembles | Small VAE + PCA |
| Threshold | Often fixed | Blended adaptive + static baseline |
| Drift | Often omitted | KS monitor + fine-tune ablation |
| Eval discipline | Mixed | Anomaly-free val calibration; PR-AUC primary |
| Scope honesty | Sometimes overstated | Network explicitly out of scope |

Module 3 differs by combining a **minimal** deep detector with a **measured** drift stack and by refusing to claim network-fault detection without network features.

---

## 5.16 Limitations

1. **Network bottlenecks not delivered** — delay/loss relabelled normal; no Istio/network KPIs.
2. **Integration ≠ full research stack** — demo uses static VAE-alone only.
3. **Incremental learning always triggers** on CC2 checks — limited evidence of reaching a new stable normal.
4. **SC1/SC2 excluded** from final drift reporting despite large KS vs CC1 historically.
5. **Seed variance** non-trivial for VAE-alone F1.
6. **Dependency pinning** incomplete (sklearn/numpy versions not locked).
7. **Raw `data/`** may be absent in some checkouts — reproducibility depends on regenerating processed artefacts.
8. **Window default** not re-selected after later W=60 ID gains; decision prioritised drift/integration consistency.
9. Proposal says **SGD**; implementation uses **Adam** for fine-tunes.
10. Cross-notebook re-run hazards exist if older adaptive notebooks expect pre-rescope `vae_eval` keys (documented in research history).

---

## 5.17 Future Work

1. Add network KPIs (or traces) if network-bottleneck detection is required.
2. Ship adaptive thresholding in the integration package with integrity gates.
3. Improve fine-tune policy (champion/challenger, adapter-only updates, fewer forced triggers).
4. Revisit session-disjoint or multi-case drift reporting (SC1/SC2) with clear claim labels.
5. Pin dependency versions; package a non-notebook inference library for the full drift loop.
6. Broader multi-seed reporting for adaptive/full configurations (not only VAE-alone).

---

## 5.18 Final Evaluation Plan (Remaining Gaps)

| Gap | Evaluation plan |
|-----|-----------------|
| Production streaming of adaptive+IL | Deploy buffer/KS/FT loop on a live or long replay stream; log FPR over time with fixed alert budget |
| Network fault claim | Only after adding network features; recompute per-fault recall for delay/loss |
| Fine-tune stability | Measure post-FT val MSE on held-out normals; require non-regression gate before accepting weights |
| W=30 vs W=60 under identical drift protocol | Re-run blended+IL on both with the same CC2 stream script; pick by joint ID/drift utility, not ID alone |

---

## 5.19 Closing Statement

Module 3’s canonical `module3_pipeline/` delivers a **PCA-whitened VAE** for container CPU/memory performance anomalies, with **strong evidence** that blended adaptive thresholding improves drift-set detection and **supporting evidence** that KS-triggered incremental fine-tuning yields additional gains. Network-bottleneck detection from the original proposal is **not delivered**. The integration handoff correctly exposes the validated **VAE-alone** scorer for deterministic demo use.
