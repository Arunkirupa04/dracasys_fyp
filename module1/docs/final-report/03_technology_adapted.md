# Chapter 3 — Technologies and Techniques Adopted

**Module:** Module 1 — Long-Term Resource Forecasting (Hybrid Prophet + GRU)  
**Purpose:** Explain each adopted technique in terms of the Module 1 problem, and map it to the actual implementation under `module1/app/` and `module1/artifacts/hybrid_v1/`.

---

## 3.1 Overview

Module 1 combines interpretable seasonal forecasting with a neural residual corrector. Technologies are selected because container CPU series show (i) daily structure suitable for Prophet, (ii) leftover nonlinear residual structure, and (iii) a need for a stable production inference path.

Canonical implementation homes:

| Technique | Primary source |
|-----------|----------------|
| Prophet fit / forecast | `app/prophet/forecaster.py` |
| Hybrid combine | `app/inference/hybrid_pipeline.py` |
| GRU predict | `app/inference/` + `artifacts/hybrid_v1/model.keras` |
| Scaling / validation | `app/preprocessing/` |
| API orchestration | `app/api/`, `app/services/` |
| Research training / eval | `experiments/`, `docs-experiments/` |

---

## 3.2 Why Prophet and GRU Are Combined

Prophet alone captures trend and daily seasonality well but leaves a residual series. A pure GRU on raw CPU can learn seasonality implicitly but is harder to interpret and, in Module 1’s Global GRU baseline, underperforms the hybrid on Day-1 MAE.

**Design choice:**  
\(\hat{y} = \mathrm{Prophet}(y) + \mathrm{GRU}(y - \mathrm{Prophet}(y))\).

Prophet explains the seasonal baseline; the GRU focuses capacity on what remains. Production metadata records `model_type: hybrid_prophet_gru`. The proposal’s **LSTM** residual network was **not** implemented; **GRU** is the delivered architecture.

---

## 3.3 Prophet (Trend and Seasonality)

### What it is

An additive forecasting model with trend and seasonal components, designed for scalable business time series [1].

### How it works

Fits a model on timestamps + values; forecasts future seasonal baseline. Module 1 enables **daily** seasonality and disables **weekly** seasonality (`production_config.json`).

### Why it is suitable

Container CPU often shows strong within-day structure over multi-day histories. Prophet provides a fast, interpretable baseline and is refit on each request’s history so new containers do not require stored Prophet weights.

### How it is used

At inference: validate history → MinMax-scale CPU → fit Prophet in memory → produce 96-step `yhat` in scaled space → subtract from recent history to form residuals for the GRU.

---

## 3.4 Residual Forecasting

### What it is

The leftover signal after removing a baseline forecast: \(r_t = y_t^{\mathrm{scaled}} - \hat{y}_t^{\mathrm{Prophet}}\).

### How it works

Residuals are z-scored with frozen global `res_mean` / `res_std` from `residual_stats.pkl`, then windowed (last 96 steps) as GRU input.

### Why it is suitable

Separates “what seasonality explains” from “what still needs learning,” aligning with hybrid forecasting practice [2].

### How it is used

Training and inference both use residual windows of shape `(batch, 96, 1)` → GRU → `(batch, 96)` residual forecast → add back to Prophet future → inverse MinMax to CPU %.

---

## 3.5 Gated Recurrent Unit (GRU)

### What it is

A gated RNN that updates a hidden state with reset and update gates [3], [4].

### How it works

Processes the residual sequence and emits a multi-step residual vector (direct 96-output head).

### Why it is suitable

Lighter than LSTM for a single-channel residual series while retaining nonlinear temporal modelling. Matches the frozen production architecture.

### How it is used

**Architecture (frozen):**  
`GRU(256) → Dropout(0.2) → GRU(128) → Dropout(0.2) → GRU(64) → Dense(128, ReLU) → Dense(96)`.

Weights live in `artifacts/hybrid_v1/model.keras` and are loaded once at service startup (not updated online).

---

## 3.6 Hybrid Forecast Generation

### What it is

Additive recombination of Prophet future and GRU residual forecast, then inverse scaling.

### How it works

```text
cpu_scaled ← MinMax(container)
prophet_hist, prophet_future ← Prophet.fit_predict(...)
residual_window ← zscore(cpu_scaled − prophet_hist)[-96:]
residual_hat ← GRU(residual_window)
y_hat_scaled ← prophet_future + inverse_zscore(residual_hat)
y_hat_cpu ← inverse_MinMax(y_hat_scaled)
```

### Why it is suitable

Produces both a final number for planners and optional component fields for debugging (`prophet_component_percent`, `gru_residual_component_percent`).

---

## 3.7 Scaling and Normalisation

### What it is

MinMax scaling of CPU % per container; global z-score of residuals.

### How it works

- Known IDs: frozen scalers in `scalers.pkl` (435 containers).
- Unknown IDs: fit MinMax on the provided request history.
- Residuals: frozen `residual_stats.pkl` constants.

### Why it is suitable

Keeps Prophet/GRU in stable numeric ranges and supports new containers without a full retrain.

---

## 3.8 Peak-Aware Learning (Research Technique)

### What it is

Up-weighting loss on timesteps labelled as peaks (e.g. above P90) during GRU training [5].

### How it works

Module 1 peak-aware Hybrid used weighted MSE (λ=5, P90 peaks) and stratified peak/non-peak evaluation.

### Why it was considered

Capacity risk is concentrated in high-utilisation periods.

### How it is used in the as-built system

Fully implemented as research experiments. **Not** used in production `hybrid_v1` because Hybrid **peak MAE worsened** (+0.024) while overall MAE was essentially flat. Documented as a completed evaluation with a negative/mixed outcome, not as a deployed novelty win.

---

## 3.9 Missing-Data Handling (Resample + Interpolation)

### What it is

Converting irregular samples to a regular grid and filling gaps [6].

### How it works

Research/production data pipeline: aggregate to **15-minute** means, then **linear interpolate** per container before train/val export.

### Why it is suitable

Alibaba raw traces are not guaranteed to be perfectly regular after filtering; models assume fixed 15-minute spacing.

### How it is used

Applied in offline preprocessing for all training and evaluation series. The live API requires clients to send an already gap-free 15-minute series (service validation does not interpolate request gaps).

---

## 3.10 Long-Term Evaluation Protocol

### What it is

Multi-step forecast scoring in real CPU % units.

### How it works

Primary protocol: **Day-1 = 96 steps**, mean MAE/RMSE/MAPE over **99** evaluable containers.

### Why it is suitable

Matches the production horizon lock (max 96 steps) and capacity-planning use case.

---

## 3.11 Technology-to-Pipeline Mapping

| Stage | Technologies |
|-------|--------------|
| Offline prep | Filter, 15-min resample, linear interpolate, MinMax, split |
| Train | Prophet residuals → GRU |
| Serve | FastAPI → validate → scale → Prophet refit → GRU → combine |
| Research extras | Peak-weighted loss; Global GRU baseline; residual diagnostics |

---

## References

[1] S. J. Taylor and B. Letham, “Forecasting at scale,” *The American Statistician*, 2018.  
[2] G. P. Zhang, “Time series forecasting using a hybrid ARIMA and neural network model,” *Neurocomputing*, 2003.  
[3] K. Cho et al., “Learning phrase representations…,” *EMNLP*, 2014.  
[4] J. Chung et al., “Empirical evaluation of gated recurrent neural networks…,” arXiv:1412.3555, 2014.  
[5] Cost-sensitive / peak-weighted forecasting practice.  
[6] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice* (missing data / regularity).
