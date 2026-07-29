# Summary (Thesis-Ready)

## Experiment

**Temporal Memory Analysis (TMA)** — read-only diagnostic, protocol **tma_v1.0**, run `2026-07-26_170052`.

## Research question

Did the fixed **96-step input window** capture meaningful temporal dependencies in the selected Alibaba workloads, or does useful memory exist beyond one day?

**Scope clarification:** This question must be answered **separately** for Global GRU (which sees raw CPU) and Hybrid GRU (which sees Prophet residuals). TMA analyzes three distinct signals and does not mix them.

## Methods (one paragraph)

Long-lag ACF/PACF (up to 672 steps), daily periodicity analysis, FFT/PSD, memory-length estimation, and window-sufficiency metrics were computed on 99 evaluable containers using frozen Alibaba train/validation data. Three series were analyzed independently: **original CPU**, **Prophet-equivalent residuals** (read-only Fourier OLS, no Prophet.fit()), and **Hybrid post-forecast residuals** (frozen day-1 inference cache). No forecasting models were retrained.

## Key findings (by signal)

### Signal 1 — Original CPU (relevant to Global GRU)

1. Strong daily (lag-96 mean ACF = **0.33**) and multi-day structure (lag-192 = **0.25**; 51.5% of containers |ACF| ≥ 0.1 at 2 days).
2. Only **40%** of cumulative squared-ACF energy captured within 96 lags; **60%** lies beyond one day.
3. Multi-day periodic echoes persist through lags 288–480.

### Signal 2 — Prophet-equivalent residual (relevant to Hybrid GRU input)

1. Lag-96 mean ACF drops to **0.042** — daily cycles largely removed from the residual signal.
2. Daily-cycle PSD fraction ≈ **0.7%**.
3. The signal the Hybrid GRU receives has **much shorter memory** than raw CPU.

### Signal 3 — Hybrid post-forecast residual (relevant to Hybrid forecast quality)

1. Day-1 post-forecast errors decorrelate within **~3 steps**.
2. Little long-range error autocorrelation in the cached 96-step validation block.

## What TMA proved

| Claim | Supported? |
|-------|-----------|
| Raw Alibaba CPU has substantial long-range temporal memory | **Yes** |
| Prophet removes most long-range dependence before Hybrid residual learning | **Yes** |
| 96-step window may be short for Global GRU on raw CPU | **Yes** (scientific justification) |
| 96-step window is methodologically supported for Hybrid residual learning | **Yes** |
| Longer windows would improve MAE/RMSE | **No** — not tested |
| Hybrid should adopt longer windows | **No** — not supported by residual memory evidence |
| GRU failed due to insufficient context | **No** — inconsistent with CSRLE and residual signal analysis |

## Conclusion

### Hybrid GRU

The **96-step residual-learning window is supported** by TMA. Prophet removes most long-range temporal structure (lag-96 residual ACF ≈ 0.04). The GRU never sees raw CPU. Evidence does **not** support attributing Hybrid limitations to an insufficient input window.

### Global GRU

The **96-step window appears short** relative to the temporal memory in raw CPU (~40% squared-ACF capture). Investigating longer windows (**192, 288, 384**) is a **scientifically justified future direction** — but TMA does **not** prove forecast improvement.

## Implications for thesis

> The Temporal Memory Analysis demonstrates that the selected Alibaba workloads contain substantial long-range temporal dependence in the original CPU series. However, the Prophet decomposition employed within the Hybrid architecture removes most of this long-range structure before residual learning. Consequently, the evidence supports the use of a 96-step residual-learning window within the Hybrid pipeline while simultaneously identifying longer-context raw CPU forecasting as a promising direction for future Global forecasting research.

## Research narrative position

TMA completes the diagnostic chain:

| Experiment | Layer addressed |
|------------|----------------|
| Residual Pattern Analysis | Residual structure exists after Prophet |
| CSRLE | GRU can learn synthetic temporal structure |
| LFHE | Variance collapse is partly objective-driven |
| Ridge Residual Audit | Remaining structure is predominantly linear |
| **TMA** | **Long memory is in CPU; Prophet removes it; Hybrid sees short-memory residuals; Global GRU may need longer context** |

These findings are **consistent**, not contradictory. TMA narrows future work: longer windows for Global GRU; loss function, linear modelling, and residual learnability for Hybrid.

## Verdict strings

**Global GRU:** Evidence indicates non-trivial temporal dependence beyond 96 lags on original CPU; future work may investigate longer input windows (without claiming forecast improvement).

**Hybrid GRU:** 96-step residual-learning window is methodologically supported; Hybrid limitations should not be attributed to insufficient raw-CPU context length.

## Artifacts

- Report: `experiments/temporal_memory_analysis_2026-07-26_170052/reports/final_report.json`
- Interpretation: `experiments/temporal_memory_analysis_2026-07-26_170052/reports/final_report.md`
- Notebook: `notebooks/temporal_memory_analysis.ipynb`
- Docs: `docs/temporal_memory_analysis/`
