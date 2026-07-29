# Chapter 2 — Literature Review: Container Resource Forecasting, Drift Adaptation, and Multivariate Time-Series Models

**Module:** Module 2 — Drift-Aware Short-Term Resource Prediction  
**Purpose:** Position Module 2 against existing work and identify the research gap addressed by the as-built system.

---

## 2.1 Resource Usage Prediction in Computing Systems

Forecasting CPU, memory, and related utilisation enables capacity planning, autoscaling, and energy-aware scheduling [1], [2]. Early approaches used linear models, ARIMA-family methods, and exponential smoothing [3]. Cloud and virtual-machine literature later emphasised short prediction horizons for elastic control loops [4], [5].

In container environments, telemetry is typically fine-grained (often 15 s or finer) and multi-tenant [6]. Prediction must tolerate noisy counters, missing samples, and regime changes. Module 2 targets **short horizons** (15–45 s) suitable for near-term reactive-to-proactive control rather than long-range capacity planning.

---

## 2.2 Time-Series Forecasting Fundamentals

Univariate methods predict one series from its own history. Multivariate methods use several series jointly [3], [7]. Short-term forecasting emphasises recent local dynamics; long-term forecasting emphasises seasonality and trends. Evaluation commonly uses MAE, RMSE, and MAPE, with care when targets approach zero [8].

**Persistence** (predict “no change”) is a critical baseline for slowly changing industrial telemetry: many learned models fail to beat it when series are near-random-walk or when cumulative counters dominate [9]. Module 2’s history explicitly encountered this phenomenon and redesigned the output head accordingly (residual anchoring).

---

## 2.3 Container and Cloud Resource Prediction

Studies on Docker/Kubernetes resource prediction apply machine learning to cAdvisor/Prometheus metrics [10], [11]. Common targets are CPU and memory; network and disk appear when the monitoring stack exposes them. Cross-deployment generalisation is difficult because absolute counter scales differ across clusters and lifetimes—an issue Module 2 hit empirically when training and testing on different AIOpsArena cases (~44× CPU scale mismatch in legacy experiments).

---

## 2.4 GRU and LSTM Forecasting Models

Recurrent networks model sequential dependence. LSTMs use gating to mitigate vanishing gradients [12]; GRUs simplify the gating structure while remaining competitive on many sequence tasks [13], [14]. Deep forecasting surveys place RNNs among strong nonlinear baselines for multivariate series [7], [15].

**Residual / delta formulations.** Predicting corrections relative to the last observation (or relative to a simple baseline) often stabilises learning on near-persistent series [16]. Module 2’s `AdaptiveGRUModel` adopts `prediction = last_value + correction` with a zero-initialised correction head so training starts exactly at persistence.

---

## 2.5 Short-Term Forecasting for Control Loops

Autoscalers and admission controllers benefit from predictions seconds to minutes ahead [4], [17]. Longer horizons accumulate uncertainty. Module 2 trains separate checkpoints for horizons 1, 2, and 3 steps (15 s spacing), reporting that longer horizons need more epochs and yield higher error—consistent with forecasting theory.

---

## 2.6 Concept Drift and Adaptive Forecasting

Concept drift denotes changes in the joint distribution of inputs and targets over time [18], [19]. Responses include: (i) windowing / forgetting; (ii) explicit drift detectors; (iii) incremental or online updates; (iv) ensemble switching [18], [20].

Error-based detectors monitor residual statistics (moving averages, control limits, Page–Hinkley, ADWIN) [20], [21]. Module 2’s `DriftMonitor` uses an EWMA of chunk error and a z-score against a frozen warmup reference, requiring sustained hits before declaring drift.

---

## 2.7 Adaptive Sliding Windows

Adaptive windows change how much history informs a prediction. Some systems enlarge windows under high volatility for stability; others shorten windows to emphasise recent behaviour [18], [22]. Module 2 implements a **variability-driven lookback in [500, 1000]** computed from each container’s training-period rolling std of CPU. In the as-built formula, **higher recent variability shortens** lookback toward 500 (not lengthens it). Lengths are **precomputed in preprocessing**, not recomputed continuously at streaming inference.

---

## 2.8 Online and Incremental Learning

Module 2’s `OnlineAdapter` performs Adam fine-tuning (lr \(10^{-4}\), up to 4096 recent windows, 1 epoch) when `DriftMonitor` triggers. Phase 4 stream validation shows Adaptive MAPE improved versus Static after these updates; for the project’s showcase scope this online-learning path is treated as fully implemented.

---

## 2.9 Workload Bursts and Spike Handling

Bursty workloads produce abrupt level or rate changes [25], [26]. Evaluation strategies include: mining natural spikes; injecting synthetic bursts; and reporting separate metrics on spike-affected versus normal segments. Natural spikes in Module 2’s chosen case were dominated by collection artifacts, motivating **synthetic ramp–hold–decay injection** with spike masks for regime evaluation.

---

## 2.10 Multivariate Forecasting (CPU and Memory)

Module 2 forecasts four related CPU and memory series jointly from a shared GRU latent state (one network → four outputs). Network and disk KPIs are not present in the final dataset and are outside the implemented scope.

---

## 2.11 Comparative Summary

| Approach | Horizon | Drift handling | Burst handling | Multivariate | Typical limitation |
|----------|---------|----------------|----------------|--------------|--------------------|
| Persistence | Any | None | Poor under jumps | Per series | Strong on calm counters |
| SES / ETS [3] | Short–medium | Limited | Limited | Usually univariate | Fixed smoothing |
| ARIMA / VAR [3], [7] | Medium | Offline refit | Limited | VAR multivariate | Stationarity assumptions |
| LSTM/GRU forecast [12]–[15] | Configurable | Often none | Data-dependent | Possible | May lose to persistence |
| Online drift systems [18]–[21] | — | Detector + update | Indirect | Optional | Contamination risk |
| **Module 2 (as-built)** | 15–45 s | EWMA/z + fine-tune | Synthetic bursts + regimes | CPU+mem joint heads | Clean-data persistence dominance; modest adaptive MAPE gains |

---

## 2.12 Research Gap and Module 2 Positioning

**Gaps identified**

1. Short-horizon container forecasting that remains competitive with strong persistence baselines on cumulative CPU counters.
2. Drift-aware evaluation that proves updates actually change weights and can stabilise stream error.
3. Burst-stress protocols when natural spikes are artifactual.

**How Module 2 addresses them**

- Residual GRU gives a structural floor at persistence (addresses gap 1 partially; clean-test persistence still wins overall).
- Phase-4 `DriftMonitor` + `OnlineAdapter` with checksum-verified updates (gap 2).
- Synthetic burst injection + spike/normal MAPE tables (gap 3).

**Remaining gap:** multi-seed uncertainty reporting and optional live lookback refresh as robustness extensions (not required to claim the validated drift-aware / online-learning novelty).

---

## References

[1] C. Delimitrou and C. Kozyrakis, “Quasar,” *ASPLOS*, 2014.  
[2] B. Burns et al., “Borg, Omega, and Kubernetes,” *CACM*, 2016.  
[3] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, OTexts.  
[4] N. Roy, A. Dubey, and A. Gokhale, “Efficient autoscaling in the cloud using predictive models,” *IEEE Cloud*, 2011.  
[5] A. Gandhi et al., “AutoScale: Dynamic, robust capacity management,” *TOCS*, 2012.  
[6] D. Bernstein, “Containers and cloud,” *IEEE Cloud Computing*, 2014.  
[7] B. Lim and S. Zohren, “Time-series forecasting with deep learning: A survey,” *Phil. Trans. R. Soc. A*, 2021.  
[8] R. J. Hyndman and A. B. Koehler, “Another look at measures of forecast accuracy,” *IJF*, 2006.  
[9] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, “Statistical and machine learning forecasting methods,” *PLOS ONE*, 2018.  
[10] Applied studies on Docker/cAdvisor resource prediction (CPU/memory time series).  
[11] Prometheus / cAdvisor monitoring conventions for container KPIs.  
[12] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, 1997.  
[13] K. Cho et al., “Learning phrase representations…,” *EMNLP*, 2014.  
[14] J. Chung et al., “Empirical evaluation of gated recurrent neural networks,” arXiv:1412.3555, 2014.  
[15] B. N. Oreshkin et al., “N-BEATS,” *ICLR*, 2020 (deep forecasting context).  
[16] Residual / baseline-anchored forecasting practice in industrial ML.  
[17] T.-W. Yang et al., short-horizon cloud workload prediction for autoscaling.  
[18] J. Gama et al., “A survey on concept drift adaptation,” *ACM CSUR*, 2014.  
[19] J. Lu et al., “Learning under concept drift,” *IEEE TKDE*, 2019.  
[20] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” *SDM*, 2007.  
[21] E. S. Page, “Continuous inspection schemes,” *Biometrika*, 1954 (change detection lineage).  
[22] Adaptive windowing in streaming analytics systems.  
[23] S. C. H. Hoi et al., “Online learning: A comprehensive survey,” *Neurocomputing*, 2021.  
[24] M. McCloskey and N. J. Cohen, “Catastrophic interference,” 1989.  
[25] A. Ali-Eldin et al., workload burstiness analyses.  
[26] M. F. Arlitt and C. L. Williamson, “Web server workload characterization,” *SIGMETRICS*, 1996.
