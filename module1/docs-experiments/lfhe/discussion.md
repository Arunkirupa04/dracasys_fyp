# LFHE Discussion

## Research question

> Is variance collapse primarily caused by MSE rather than Hybrid architecture?

## Answer (from LFHE v1.1)

**Partially, but insufficiently at λ=1.0.** DA-MSE proves that the **training objective can control dispersion** without architectural change — variance collapse is **not purely architectural**. However, fixing dispersion alone **does not** recover temporal correlation and **costs** MAE at the frozen λ.

## Mechanism

MSE permits low-amplitude corrections with acceptable squared error. The one-sided log-variance penalty removes that local optimum by penalizing σ_ŷ ≪ σ_y. At λ=1.0, the optimizer over-compensates — σ_ŷ often exceeds σ_y (std ratio > 1), producing noisy corrections that increase energy without matching lag structure.

## Relation to Ridge analysis

Ridge achieved std ratio ~0.22 **with** higher r (~0.158). DA-MSE exceeded Ridge amplitude (~0.83) **without** r gain — confirming that **amplitude alone is not sufficient** for shape learning; linear inductive bias (Ridge) or balanced objective design is needed.

## Horizon diagnostics

MSE arm shows variable collapse across bands (lowest at 25–48). Not a clean monotonic long-horizon degradation — collapse is **partly horizon-specific** but also **global**.

## Thesis language (suggested)

> LFHE tested whether a frozen dispersion-augmented MSE objective could reduce Hybrid GRU variance collapse on CSRLE B0 without architectural change. DA-MSE substantially increased residual amplitude and energy recovery (std ratio 0.10→0.83) but did not improve Pearson correlation and violated the MAE guardrail at λ=1.0. The result is **inconclusive for full H₁**: objective design contributes to dispersion, but **MSE-only vs DA-MSE-only** does not resolve shape learning; variance collapse has **objective and capacity components**.

## Limitations

- Single λ (frozen by design)
- Non-deterministic MSE retrain vs Stage 2 (reproduction via frozen weights)
- B0 only (no Condition A confirmatory run)
