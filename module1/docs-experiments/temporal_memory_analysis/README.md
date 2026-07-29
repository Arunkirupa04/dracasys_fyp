# Temporal Memory Analysis (TMA)

## Status

**COMPLETE** — `experiments/temporal_memory_analysis_2026-07-26_170052/`

## Purpose

Read-only diagnostic answering whether the fixed **96-step input window** (1 day at 15-minute resolution) captures temporal dependencies in the selected Alibaba workloads, or whether useful memory exists beyond one day.

TMA closes a **methodological gap** in the FYP: before attributing forecasting limitations to the GRU architecture, the project needed evidence on whether the fixed input window itself was adequate for the data.

## Three distinct signals (do not mix)

TMA analyzes **three separate time series**. Each answers a different question. Results from one series must **never** be transferred to another without explicit justification.

| Signal | What it represents | Primary question |
|--------|-------------------|------------------|
| **1. Original CPU** | Raw workload utilisation | How much temporal memory exists in the Alibaba workloads? |
| **2. Prophet-equivalent residual** | CPU minus daily seasonality and trend — the signal the Hybrid GRU actually receives | Does long-range memory survive Prophet decomposition? |
| **3. Hybrid post-forecast residual** | Actual minus Hybrid forecast on validation day-1 | Does the Hybrid model leave structured forecast errors? |

See [methodology.md](methodology.md) for series definitions and [discussion.md](discussion.md) for full interpretation.

## What TMA proved

1. **Original CPU** retains substantial daily and multi-day temporal memory (lag-96 mean ACF ≈ 0.33; lag-192 ≈ 0.25; only ~40% of cumulative squared-ACF energy within 96 lags).
2. **Prophet-equivalent residuals** contain much weaker long-range dependence (lag-96 mean ACF ≈ 0.04) — Prophet removes most daily temporal structure before residual learning.
3. **Hybrid post-forecast residuals** decorrelate within ~3 steps — little structured error autocorrelation on the day-1 forecast block.
4. For **Global GRU** (raw CPU forecasting), TMA provides **scientific justification** to investigate longer input windows (192, 288, 384) in future work.
5. For **Hybrid GRU**, TMA **supports the original pipeline design**: Prophet handles long-range behaviour; the GRU handles short-term residual correction on a substantially shorter-memory signal.

## What TMA did NOT prove

TMA quantifies **available temporal autocorrelation** only. It does **not** claim:

- Longer windows improve MAE or RMSE
- Hybrid should use longer windows
- Global GRU accuracy would necessarily improve with longer context
- The GRU failed because of insufficient input context

See [limitations.md](limitations.md) — subsection *What TMA Does Not Claim*.

## Verdict (interpreted)

| Architecture | TMA conclusion |
|--------------|----------------|
| **Global GRU** | 96-step window appears short relative to raw CPU memory; longer-context investigation is scientifically justified (not performance-proven). |
| **Hybrid GRU** | 96-step residual-learning window is methodologically supported; evidence does **not** support the hypothesis that Hybrid underperformed because the window was too short. |

## Relationship to prior experiments

TMA complements — and is consistent with — the prior diagnostic chain:

```
Residual Pattern Analysis  →  Prophet residuals still contain structure
CSRLE                      →  GRU can learn synthetic temporal structure
LFHE                       →  Variance collapse is partly objective-driven
Ridge Residual Audit       →  Remaining structure is predominantly linear
TMA                        →  Long memory lives in CPU; Prophet removes it;
                              Hybrid GRU sees short-memory residuals
```

Full narrative: [discussion.md](discussion.md#relationship-to-previous-experiments).

## Authoritative artifacts

| Path | Content |
|------|---------|
| `config/temporal_memory_analysis_config.json` | Frozen protocol v1.0 |
| `acf/`, `pacf/` | Per-series long-lag matrices and cohort summaries |
| `statistics/` | Daily periodicity, memory length, window sufficiency, paired tests |
| `plots/` | Publication-quality PNG/PDF figures |
| `reports/final_report.json` | Machine-readable answers to 7 research questions |
| `reports/final_report.md` | Human-readable report with interpretation |
| `notebooks/temporal_memory_analysis.ipynb` | Interactive walkthrough (manual execution) |

## Documentation index

| Document | Purpose |
|----------|---------|
| [research_question.md](research_question.md) | Research framing |
| [methodology.md](methodology.md) | Protocol, three-series design, scope |
| [implementation_log.md](implementation_log.md) | Build chronology |
| [analysis.md](analysis.md) | Analytical pipeline |
| [results.md](results.md) | Quantitative outcomes with per-series interpretation |
| [discussion.md](discussion.md) | Full scientific interpretation and research narrative |
| [limitations.md](limitations.md) | Constraints, caveats, and explicit non-claims |
| [verification.md](verification.md) | Integrity checks |
| [summary.md](summary.md) | Thesis-ready summary and wording |

## Read-only guarantee

No modification of Hybrid, Global GRU, Peak-Aware, CSRLE, LFHE, RRA, or Audit experiments. No model training in this experiment. This documentation update is interpretation-only; numerical results are unchanged.
