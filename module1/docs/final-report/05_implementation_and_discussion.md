# Chapter 5 — Implementation, Discussion and Evaluation

**Module:** Module 1 — Long-Term Resource Forecasting (Hybrid Prophet + GRU)  
**Canonical production code:** `module1/app/` + `module1/artifacts/hybrid_v1/`  
**Canonical Day-1 metrics:** `module1/experiments/baseline_reference_2026-07-14/`  
**Design reference:** Chapter 4 (Figures 4.1–4.4)

---

## Part A — Implementation

## 5.1 Software Technologies and Environment

| Item | Evidence |
|------|----------|
| Language | Python |
| API | FastAPI (`app/main.py`, `app/api/`) |
| Deep learning | TensorFlow / Keras GRU (`model.keras`) |
| Seasonal model | Facebook Prophet |
| Persistence | Pickle scalers / residual stats; JSON config |
| Entry | `module1/run.py` (default port **8000**) |

Serving is **stateless**: no online fine-tune; artifacts are read-only.

---

## 5.2 Repository Layout (Implementation Map)

```text
module1/
├── app/                      # production FastAPI service
│   ├── api/                  # routes, schemas
│   ├── inference/            # hybrid pipeline, GRU predict
│   ├── prophet/              # per-request Prophet
│   ├── preprocessing/        # validate + scale
│   └── services/             # artifact loader, orchestration
├── artifacts/hybrid_v1/      # frozen production bundle
├── experiments/              # evaluation freezes & research runs
├── docs/implementation/      # service docs
├── docs-experiments/         # research write-ups
├── examples/                 # unseen-container demo scripts
└── docs/final-report/        # this documentation set
```

---

## 5.3 Dataset Preparation

**Source:** Alibaba Cluster Trace — container CPU utilisation.

| Item | Value |
|------|-------|
| Target | `cpu_util_percent` (0–100) |
| Sampling after prep | **15 minutes** |
| Containers after filters | **486** |
| Production scalers | **435** known IDs |
| Per-container split | 80% train / 20% val (chronological) |
| Approx. duration | ~**8 days**/container (≈613 train + ≈154 val points) |
| Gap handling (offline) | 15-min resample + **linear interpolate** |

Raw `data/` is gitignored in typical checkouts; pipeline stages are documented under `docs-experiments` data-pipeline references.

---

## 5.4 Preprocessing Implementation

Offline (research/production training lineage):

1. Filter containers (length / non-zero variance gates).
2. Resample to 15-minute means.
3. Linear interpolate missing bins.
4. Fit per-container `MinMaxScaler`.
5. Export train/val series for Prophet residual construction.

Online (API):

1. Validate ≥200 points, 15-minute spacing, CPU range.
2. Apply frozen scaler if `container_id` known; else fit on request history.
3. **Do not** interpolate gaps in the request — client must pre-fill.

---

## 5.5 Prophet Implementation

**Module:** `app/prophet/forecaster.py`

| Setting | Value |
|---------|------:|
| Daily seasonality | True |
| Weekly seasonality | False |
| Fit timing | **Every request** on scaled history |
| Horizon | 96 steps |

Prophet contributes the seasonal baseline used both to form residuals and as the future additive component.

---

## 5.6 Residual Calculation and GRU Implementation

**Residual:** `cpu_scaled − prophet_yhat`, then global z-score using `residual_stats.pkl`.

**GRU architecture (frozen):**

```text
GRU(256) → Dropout(0.2) → GRU(128) → Dropout(0.2) → GRU(64)
  → Dense(128, ReLU) → Dense(96)
```

| Tensor | Shape |
|--------|-------|
| Input | `(batch, 96, 1)` |
| Output | `(batch, 96)` |

Loaded once at startup from `artifacts/hybrid_v1/model.keras`.

---

## 5.7 Hybrid Forecast Generation (Pseudocode)

```text
function FORECAST(container_id, timestamps, cpu_history):
    validate_request(...)
    scaler ← lookup_or_fit(container_id, cpu_history)
    y ← scaler.transform(cpu_history)
    prophet ← fit_prophet(timestamps, y)
    yhat_hist, yhat_future ← prophet_predict(96)
    r ← zscore(y − yhat_hist)
    r_hat ← GRU(r[-96:])
    y_hat ← inverse_zscore(r_hat) + yhat_future
    return scaler.inverse_transform(y_hat)
```

Response also exposes Prophet and GRU components for inspection.

---

## 5.8 Peak-Aware Implementation (Research Only)

Peak-aware Hybrid experiments (e.g. `experiments/peak_aware_2026-07-14_164518/`):

- Peak definition: P90-based labelling in the study protocol
- Loss: weighted MSE with λ=5 on peaks
- Control baseline: frozen Hybrid Day-1 metrics (immutable freeze folder)

**Not copied into `artifacts/hybrid_v1/`.**

---

## 5.9 Artifact Bundle

| File | Role |
|------|------|
| `model.keras` | GRU weights |
| `scalers.pkl` | Per-container MinMax |
| `residual_stats.pkl` | Global residual mean/std + window/horizon |
| `production_config.json` | Frozen hyperparameters / model type |
| `prophet_metadata.pkl` | Training-time Prophet config snapshot |
| `train_container_ids.json` | Known scaler IDs |

---

## Part B — Discussion and Evaluation

## 5.10 Evaluation Methodology

| Axis | Protocol |
|------|----------|
| Primary horizon | Day-1 = **96 steps** |
| Units | Real CPU % |
| Cohort | **99** containers (1 skipped) |
| Primary metrics | MAE, RMSE, MAPE (mean ± std across containers) |
| Production baseline freeze | `baseline_reference_2026-07-14` |
| Comparisons | Global GRU; peak-aware variants; window ablation; secondary residual studies |

**MAPE caution:** near-zero actual CPU inflates MAPE (std ≈ 615 on Hybrid). Prefer MAE/RMSE for narrative claims.

---

## 5.11 Experiments Conducted

### Experiment E1 — Frozen Hybrid Prophet + GRU (production baseline)

| Field | Detail |
|-------|--------|
| Source | `experiments/baseline_reference_2026-07-14/evaluation/evaluation_summary.csv` |
| Objective | Freeze Day-1 control metrics for all later studies |

| Metric | Mean | Std |
|--------|-----:|----:|
| **MAE** | **1.745882** | 2.486804 |
| **RMSE** | **2.387813** | 3.219141 |
| MAPE | 111.940318 | 615.299100 |

**Demonstrates:** Production hybrid operating point on the 99-container protocol.

---

### Experiment E2 — Hybrid vs Global GRU

| Field | Detail |
|-------|--------|
| Source | `experiments/hybrid_vs_global_2026-07-17_095147/phase6_comparison.json` |
| Objective | Fair comparison of Hybrid vs Global GRU on same Day-1 task |

| Metric | Hybrid | Global GRU | Δ (Global − Hybrid) |
|--------|-------:|-----------:|--------------------:|
| MAE | **1.7459** | 1.9242 | +0.178 |
| RMSE | **2.3878** | 2.6105 | +0.223 |
| MAPE | 111.94 | 116.50 | +4.56 |

Hybrid lower MAE on **67/99** containers.

**Demonstrates:** Hybrid beats pure Global GRU on the primary cohort — supports Novelty 1.

---

### Experiment E3 — Prophet diagnostic vs Hybrid

| Field | Detail |
|-------|--------|
| Source | Residual-pattern / CSRLE diagnostics |
| Objective | Quantify how much GRU adds beyond Prophet |

| Model | Day-1 mean MAE (diagnostic) |
|-------|----------------------------:|
| Prophet alone | ≈ **1.733** |
| Hybrid | ≈ **1.746** |

Prophet explains a large share of variance (documented ~78% ceiling in limitations). GRU residual learning is **incremental and sometimes mixed** on Day-1 MAE; Hybrid’s advantage is clearest vs Global GRU, not always vs Prophet-only.

---

### Experiment E4 — Input window ablation (Hybrid)

| Window | Mean MAE | Mean RMSE |
|-------:|---------:|----------:|
| **96** | **1.7402** | **2.3792** |
| 288 | 1.7523 | 2.3920 |

**Decision:** keep 96-step input window for Hybrid.

---

### Experiment E5 — Peak-aware Hybrid

| Field | Detail |
|-------|--------|
| Source | Peak-aware Hybrid phase evaluation / discussion |
| Objective | Improve peak-period accuracy via weighted MSE |

| Tier | Baseline | Peak-aware | Δ |
|------|---------:|-----------:|--:|
| Overall MAE | 1.746 | 1.741 | −0.005 |
| Peak MAE | 2.501 | 2.526 | **+0.024** |
| Non-peak MAE | 1.485 | 1.470 | −0.015 |

**Demonstrates:** Peak-aware Hybrid **does not** improve peaks. Not promoted to production.

---

### Experiment E6 — Peak-aware Global GRU

| Tier | Baseline | Treatment | Δ |
|------|---------:|----------:|--:|
| Overall MAE | 1.924 | 2.174 | **+0.249** |
| Peak MAE | 3.426 | 2.649 | **−0.777** |
| Non-peak MAE | 1.406 | 2.010 | +0.604 |

**Demonstrates:** Peak gains possible for Global GRU, but overall accuracy degrades — still not a production Hybrid upgrade.

---

### Experiment E7 — Secondary residual / architecture studies (brief)

| Study | Headline |
|-------|----------|
| HCERL (context features) | Best MAE **V0 1.732**; V1–V4 worse — keep simple Hybrid |
| GGTCE (Global longer windows) | G96 MAE **1.924** best among G96/G192/G288 |
| RRE (robust residual scaling) | No meaningful MAE win vs R0 |
| CSRLE / LFHE | Residual learnability / dispersion studies; no production replacement of `hybrid_v1` |

---

### Experiment E8 — Unseen-container API demo

| Field | Detail |
|-------|--------|
| Source | `examples/` unseen validation (`c_10312`) |
| Metrics | MAE **0.35%**, RMSE **0.47%**, MAPE **0.52%** |

**Note:** Single-container demo under scaler-mode “new”; **not** a substitute for the 99-container Day-1 protocol. Primary claims should cite MAE **1.746**.

---

## 5.12 Results Analysis

**Best production approach:** Frozen **Hybrid Prophet + GRU** (`hybrid_v1`) with Day-1 MAE **1.746**, beating Global GRU (**1.924**).

**Prophet contribution:** Provides the seasonal baseline; large fraction of explainable variance.  

**GRU contribution:** Residual correction that improves over Global GRU; gains vs Prophet-only are small/mixed on Day-1 diagnostic MAE — residual signal is weak.

**Long-term trends / seasonality:** Handled primarily by Prophet (daily seasonality on).  

**Peaks:** Peak-aware Hybrid failed its goal (peak MAE up). Production model remains MSE-trained Hybrid; limitations note possible under-forecast of bursts.

**Missing data:** Training/eval series are regularised with resample + linear interpolation. Live API expects gap-free inputs.

---

## 5.13 Novelty Evaluation

| # | Proposed novelty | Status | Evidence |
|---|------------------|--------|----------|
| 1 | Robust hybrid (statistical + deep learning) | ✅ **Fully implemented** | Prophet + **GRU** additive hybrid in production; Hybrid MAE 1.746 vs Global 1.924 |
| 2 | Peak-aware learning | ⚠️ **Partially implemented** (evaluated; not successful on Hybrid peaks) | Full experiment trail; peak MAE +0.024; **not in `hybrid_v1`** |
| 3 | Missing-data robustness | ✅ **Implemented in data pipeline** | 15-min resample + linear interpolation for train/eval preparation; API requires client-side gap-free series |

**Wording correction:** Proposal “LSTM” → delivered **GRU**.

---

## 5.14 Comparison With Existing Work

| Aspect | Typical hybrid papers | Module 1 |
|--------|----------------------|----------|
| Deep component | Often LSTM | **GRU** |
| Serving | Notebook / offline | FastAPI frozen bundle |
| Peak claims | Sometimes unverified | Evaluated; Hybrid peak claim rejected |
| Scope | Often multi-metric | CPU-only, honest limits |
| Adaptation | Sometimes online | Stateless; no retrain API |

---

## 5.15 Limitations

1. **CPU only** — no memory/network forecasts.  
2. **Horizon capped at 96 steps** (24 h).  
3. **No online drift / retrain** in the service.  
4. **API does not fill gaps** — client must pre-interpolate.  
5. **Peak-aware not productionised** after negative Hybrid peak result.  
6. **Residual learner is weak** — Prophet carries most structure; GRU gains are incremental.  
7. **MAPE unstable** near idle CPU.  
8. **Latency** 5–30+ s/request due to Prophet refit.  
9. **Formal multi-container unseen holdout protocol** incomplete relative to temporal val protocol.  
10. Some research training scripts referenced historically may not all be present in the current tree (artifacts + experiments remain the evidence for metrics).

---

## 5.16 Future Work

1. Stronger residual models or features if Hybrid–Prophet gap should widen.  
2. Redesign peak-aware loss / peak definition before any production merge.  
3. Optional server-side gap interpolation aligned with the offline pipeline.  
4. Uncertainty intervals for capacity planning.  
5. Multi-metric extension (memory) if required by the wider system.  
6. Formal unseen-container cohort evaluation beyond single demos.

---

## 5.17 Final Evaluation Plan (Remaining Gaps)

| Gap | Plan |
|-----|------|
| Unseen cohort | Freeze N never-trained IDs; report Day-1 MAE/RMSE with scaler-mode new |
| Peak redesign | Try alternate peak labels / losses; require peak MAE↓ and overall MAE not↑ before merge |
| Gap-handling UX | Add optional interpolate flag in API; A/B latency and accuracy on gappy histories |
| Prophet vs Hybrid significance | Paired per-container tests on Day-1 MAE across the 99-cohort |

---

## 5.18 Closing Statement

Module 1 delivers a production **Prophet + GRU** hybrid for **24-hour container CPU** forecasting at **15-minute** resolution, with primary cohort MAE **1.746** outperforming a Global GRU baseline. Peak-aware training was evaluated and **not** adopted for Hybrid production. Missing ticks in offline data are handled by **resampling and linear interpolation**. The proposal’s **LSTM** residual component is **not** present in the as-built system—**GRU** is the correct core-model wording for all final claims.
