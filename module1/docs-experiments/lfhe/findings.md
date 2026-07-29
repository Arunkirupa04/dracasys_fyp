# LFHE Findings

## Primary findings

1. **DA-MSE dramatically increases predicted residual amplitude** — cohort std ratio 0.10 → 0.83 (100% of containers improved vs MSE retrain).

2. **Energy recovery improves in parallel** — ERR 0.011 → 0.823.

3. **Correlation does not improve** — Pearson r 0.113 → 0.087 (MSE retrain actually had higher r than DA-MSE in this session).

4. **MAE guardrail violated** — residual MAE increased by 0.0114 (> 0.010 threshold).

5. **CPU forecasting unchanged** — no material Hybrid − Prophet MAE gain.

6. **Reproduction gate passed** — LFHE evaluation pipeline exactly reproduces Stage 2 B0 metrics on frozen weights.

## Interpretation of inconclusive_I1

The dispersion penalty **works mechanistically** — it forces higher output variance. However, at λ=1.0 it produces **over-dispersion** without shape alignment:

- Amplitude restored (even beyond Ridge reference 0.22)
- Temporal correlation not improved
- Point-error cost increased

This supports a **partial** role for the objective (MSE alone permits under-dispersion) but **rejects** the simple hypothesis that DA-MSE at frozen λ simultaneously fixes collapse **and** preserves/improves shape fidelity.

## What this does not show

- Does not justify production deployment of DA-MSE
- Does not imply λ tuning would fix correlation (deferred to future work)
- Does not reopen CSRLE generator or architecture questions
