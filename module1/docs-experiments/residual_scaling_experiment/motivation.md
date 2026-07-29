# Motivation

Hybrid Prophet+GRU consistently improves over Prophet alone, yet a series of training-side interventions — peak-aware loss, context enrichment (HCERL), longer input windows (GGTCE), and LFHE — failed to produce statistically significant gains.

Stage 0 (Residual Representation Diagnostics) established that:

1. The level residual (R0) preserves the strongest temporal structure.
2. Velocity differencing (R1) **removes** temporal dependence and was rejected.
3. Per-container normalization (R2) offers no clear advantage.
4. Robust median/MAD scaling (R3) preserves **identical** ACF/PACF to R0 while improving numerical properties (lower sparsity in scaled units, same skew/kurtosis shape).

The remaining hypothesis is therefore **not** whether another temporal representation contains more information.

It is whether **numerical conditioning** of the GRU input — holding temporal content fixed — makes optimisation easier and thereby improves forecasts.

This experiment tests that hypothesis objectively. Negative results are equally valuable.
