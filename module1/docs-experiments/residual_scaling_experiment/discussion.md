# Discussion

## Framing

RRE v1.0 isolates **one** change: how the Prophet residual is scaled before GRU training. Stage 0 already established that R3 does not introduce new temporal information relative to R0. Any observed difference must therefore be attributed to:

- Gradient scale and MSE curvature under Adam
- Effective learning rate relative to residual magnitude
- Sparsity of near-zero inputs (R3: 3.3% vs R0: 10.5% at |x| < 0.05)
- Robustness to heavy-tail outliers during training

## Empirical outcome (run `rre_2026-07-27_123441`)

RRE v1.0 produced a **null result**:

- Day-1 MAE: R0 **1.740%** vs R3 1.745% (Δ = +0.0052%, Wilcoxon p = 0.975)
- R3 converged one epoch earlier (best epoch 4 vs 5) but did not improve CPU forecasts
- Raw MSE loss is not comparable across arms (R3 scaled variance ≈ 7× R0)
- **Recommendation:** retain frozen global z-score (R0) as Hybrid standard

This aligns with HCERL, peak-aware, and window experiments: residual preprocessing is not the bottleneck.

### If R3 improves accuracy significantly

Robust scaling may improve optimisation enough to translate into better Day-1 forecasts despite identical temporal structure. This would suggest the Hybrid bottleneck is partly **numerical**, not informational — consistent with flat loss plateaus observed in production training.

### If R3 improves optimisation but not accuracy

The GRU may converge faster or with smaller generalisation gap without changing forecast quality. This supports the optimisation hypothesis but would **not** justify replacing the Hybrid baseline scaling in deployment.

### If R3 shows no improvement (null result)

The frozen global z-score baseline is validated. Combined with HCERL, peak-aware, and window experiments, this strengthens the conclusion that Hybrid performance is limited by factors other than residual scaling — e.g. Prophet ceiling, sequence Markovian sufficiency, or architecture capacity.

## Relation to Stage 0

Stage 0 rejected velocity because it **destroyed** temporal structure. RRE v1.0 never tests velocity. The research narrative is:

1. Stage 0: choose scaling that preserves temporal information → R3 tied with R0
2. RRE v1.0: test whether R3's numerical properties help GRU optimisation

## Relation to production training

Production Hybrid training showed early plateau at ~epoch 1 with flat MSE. If R3 reaches a lower val_loss or later best epoch, it may indicate better conditioning — even if CPU MAE gains are small.
