# Chapter 2 — Literature Review: Long-Term Resource Forecasting, Prophet, and Hybrid Deep Models

**Module:** Module 1 — Long-Term Resource Forecasting (Hybrid Prophet + GRU)  
**Purpose:** Position Module 1 against existing work and identify the research gap addressed by the as-built system.

---

## 2.1 Resource Usage Forecasting in Computing Systems

Forecasting CPU and related utilisation supports capacity planning, autoscaling, and energy-aware scheduling [1], [2]. Cloud and cluster literature emphasises both short control horizons (seconds–minutes) and longer planning horizons (hours–days) [3], [4]. Module 1 targets the **planning horizon**: a **24-hour** forecast at **15-minute** steps for container CPU percentage.

---

## 2.2 Time-Series Forecasting Fundamentals

Classical methods include exponential smoothing, ARIMA-family models, and seasonal decompositions [5], [6]. Evaluation commonly uses MAE, RMSE, and MAPE, with care when series approach zero (MAPE can explode) [5]. Persistence and purely seasonal baselines remain important controls for near-regular industrial telemetry.

---

## 2.3 Trend and Seasonal Forecasting with Prophet

Prophet models time series as an additive combination of trend, seasonality, and holiday effects, with robust handling of missing data and outliers in many business settings [7]. Daily and weekly seasonalities are configurable. Strengths include interpretability and fast fitting on moderately long series. Limitations include weaker modelling of complex nonlinear residuals and bursty spikes when the additive structure is insufficient [7], [8].

Module 1 uses Prophet with **daily seasonality enabled** and **weekly seasonality disabled**, refit on each inference request’s history to form the seasonal baseline.

---

## 2.4 LSTM and GRU Forecasting Models

Recurrent networks capture sequential dependence. LSTMs use gating to mitigate vanishing gradients [9]; GRUs simplify the gating structure while remaining competitive on many sequence tasks [10], [11]. Deep forecasting surveys place RNNs among strong nonlinear baselines for multivariate and residual series [12].

**Proposal versus implementation:** early project wording referred to an LSTM residual learner. The Module 1 codebase and production artifacts implement a **GRU** residual network (`hybrid_prophet_gru`). This literature section therefore treats GRU as the delivered deep component; LSTM remains relevant background, not the deployed model.

---

## 2.5 Hybrid Statistical + Deep Learning Forecasts

Hybrid designs typically let a statistical model capture trend/seasonality while a neural network learns residuals or corrections [8], [13]. Additive hybrids are common: \(\hat{y} = \hat{y}_{\mathrm{stat}} + \hat{r}_{\mathrm{NN}}\). Benefits include interpretability of the seasonal component and flexible residual modelling. Risks include weak residual signal (if the statistical model already explains most variance) and error accumulation over long horizons.

Module 1 implements exactly this additive residual pattern with Prophet + GRU.

---

## 2.6 Resource Forecasting in Containerized Environments

Container telemetry (cAdvisor/Prometheus style) is fine-grained and multi-tenant [1], [14]. Published systems often forecast CPU and memory for autoscaling [15], [16]. Cross-container generalisation is difficult because absolute scales differ; per-container normalisation is a practical response. Module 1 forecasts **CPU only**, using per-container MinMax scalers for known IDs and history-based scaler fit for new IDs.

---

## 2.7 Peak-Aware Forecasting

Peak periods dominate capacity risk: under-forecasting peaks causes shortages, while over-forecasting wastes resources [3]. Weighted losses, peak-focused metrics, and stratified evaluation are common responses [17]. Module 1 ran controlled **peak-aware** experiments (P90 peak definition, weighted MSE). Results were **mixed**: Hybrid peak MAE did not improve; a Global GRU peak-aware variant improved peaks but degraded overall MAE. Peak-aware weights were therefore **not** merged into production `hybrid_v1`.

---

## 2.8 Missing Data and Interpolation in Time Series

Real traces often have irregular timestamps and gaps. Standard practice includes regular resampling and interpolation (linear, spline, forward-fill) before model fitting [5], [18]. Prophet itself tolerates missing timestamps in many settings [7]. Module 1’s research/production **data pipeline** resamples to 15-minute bins and applies **linear interpolation**. The inference API expects a gap-free series from the client (gaps must be pre-filled before the request).

---

## 2.9 Comparative Summary

| Approach | Horizon focus | Seasonality | Nonlinear residuals | Peak handling | Typical limitation |
|----------|---------------|-------------|---------------------|---------------|--------------------|
| Persistence / SES | Any | Weak | No | Poor under jumps | Strong only on calm series |
| Prophet alone | Medium–long | Strong | Limited | Indirect | Residual structure unused |
| Pure GRU/LSTM | Short–medium | Learned | Yes | Via loss design | May miss clear seasonality |
| Heavy ensembles | Varies | Mixed | Yes | Possible | Costly to serve |
| **Module 1 Hybrid** | **24 h @ 15 min** | Prophet daily | **GRU residual** | Research only | CPU-only; residual gains modest |

---

## 2.10 Research Gap and Module 1 Positioning

From the survey, three gaps are relevant:

1. **Container long-horizon CPU forecasting** that exposes a clear seasonal baseline *and* a residual neural corrector in one production API.
2. **Honest evaluation** of whether residual deep learning and peak-weighted training actually improve capacity-relevant errors.
3. **Operational packaging** of a frozen hybrid model for integration (stateless service), separate from research notebooks.

Module 1 addresses gap (1) with Prophet + GRU residual hybrid forecasting on Alibaba CPU traces. It addresses gap (2) with multi-container Day-1 metrics, Hybrid vs Global GRU comparison, and peak-aware ablation results (including negative peak findings). It addresses gap (3) with the `hybrid_v1` FastAPI sidecar.

**What Module 1 does not claim to close:** multi-metric (memory/network) forecasting; online adaptive retraining; LSTM-based residuals (GRU is used instead); peak-aware production deployment after mixed results.

---

## References

[1] D. Bernstein, “Containers and cloud…,” *IEEE Cloud Computing*, 2014.  
[2] B. Burns et al., “Borg, Omega, and Kubernetes,” *CACM*, 2016.  
[3] C. Delimitrou and C. Kozyrakis, “Quasar…,” *ASPLOS*, 2014.  
[4] J. Duggan et al., “Predicting the performance of virtual machine migration,” *MASCOTS*, 2015.  
[5] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*.  
[6] G. E. P. Box et al., *Time Series Analysis: Forecasting and Control*.  
[7] S. J. Taylor and B. Letham, “Forecasting at scale,” *The American Statistician*, 2018.  
[8] Zhang, “Time series forecasting using a hybrid ARIMA and neural network model,” *Neurocomputing*, 2003.  
[9] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, 1997.  
[10] K. Cho et al., “Learning phrase representations…,” *EMNLP*, 2014.  
[11] J. Chung et al., “Empirical evaluation of gated recurrent neural networks on sequence modeling,” arXiv:1412.3555, 2014.  
[12] B. Lim and S. Zohren, “Time-series forecasting with deep learning: A survey,” *Phil. Trans. R. Soc. A*, 2021.  
[13] Khashei and Bijari, hybrid forecasting literature on ARIMA–ANN combinations.  
[14] Prometheus / cAdvisor monitoring practice in Kubernetes.  
[15] Container autoscaling / resource prediction studies in cloud-native literature.  
[16] Guruge et al. / Priyadarshana et al.–style Prophet–LSTM hybrid cloud papers (proposal context; Module 1 delivers GRU).  
[17] Cost-sensitive and peak-focused forecasting literature.  
[18] Little and Rubin, *Statistical Analysis with Missing Data* (interpolation/imputation context).
