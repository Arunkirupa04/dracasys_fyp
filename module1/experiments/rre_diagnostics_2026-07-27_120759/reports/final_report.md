# Residual Representation Diagnostic Study (Stage 0)

**Protocol:** rre_diagnostics_v1.0
**Timestamp:** 2026-07-27_120759
**Containers:** 99

## Why this stage exists

Prior work established that Prophet explains ~78% of CPU variance and that Hybrid improves over Prophet, but changes to loss, context, window size, and peak weighting did not yield significant gains. The remaining limitation may lie in **how** the residual is represented for GRU learning — not **how** the GRU is trained. Representation selection must therefore precede RRE v1.0.

Velocity (first-order difference) was **not** assumed optimal. This study compares R0–R3 using read-only statistical diagnostics only.

## Comparison summary

| ID | Name | mean |ACF| | ADF reject frac | Skew | Sparsity (<0.05) |
|----|------|----------|-----------------|------|------------------|
| R0 | Level residual (global z-score) | 0.0348 | 100.00% | 1.43 | 10.53% |
| R1 | First-order difference (velocity) | 0.0116 | 100.00% | -0.07 | 27.30% |
| R2 | Per-container normalization | 0.0348 | 100.00% | 3.88 | 6.65% |
| R3 | Robust scaling (median/MAD) | 0.0348 | 100.00% | 1.43 | 3.35% |

## Evidence-based answers

1. **Most useful temporal structure:** R0
2. **Most learnable (diagnostic proxy):** R3
3. **Best temporal/stability balance:** R3
4. **Primary RRE candidate:** R3 — Robust scaling (median/MAD)
5. **Level baseline remains primary?** No — an alternative representation is recommended

## Recommendation rationale

Representation R3 (Robust scaling (median/MAD)) achieved the lowest mean rank across temporal structure, learnability proxy, statistical stability, and their balance. This recommendation is derived solely from diagnostic statistics — velocity (R1) was not assumed a priori.
