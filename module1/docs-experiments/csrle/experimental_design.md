# CSRLE Experimental Design

## Research question

Under controlled synthetic injection of known temporal structure into otherwise real Alibaba CPU traces, does the **frozen Hybrid Prophet + GRU pipeline** improve Day-1 forecasts beyond Prophet alone—and beyond i.i.d. noise controls—when residual structure is intentionally forecastable?

## Hypotheses (cautious framing)

1. **H1 (positive control):** If B0 injects deterministic periodic structure that Prophet retains in residuals (I ≈ N), the Hybrid GRU should outperform Prophet and C0 on B0 under the controlled condition.
2. **H2 (stochastic temporal):** If B1 injects nonlinear AR structure with partial Prophet retention, Hybrid should outperform C1 when VR is sufficient.
3. **H3 (negative controls):** C0/C1 i.i.d. injections should not yield systematic Hybrid gains over Prophet beyond sampling noise.
4. **H4 (real data):** Condition A should reproduce frozen Hybrid performance within pre-specified tolerance before synthetic results are interpreted.

These are **controlled synthetic** hypotheses—not claims about real-world Alibaba forecastability.

## Independent variable

**Injection mechanism:** none (A), limit-cycle (B0), SNAR (B1), i.i.d. matched to B0 (C0), i.i.d. matched to B1 (C1).

## Controlled variables

- Same 99 evaluable containers
- Same train/val temporal split
- Same Prophet and (when run) GRU architecture and training protocol
- κ = 0.10 fixed before any GRU results
- seed = 42, pre-warm = 200
- Hard clip [0, 100] CPU %
- Train-only normalization for z and condition scalers

## Why the same 99 containers (not all 484)

CSRLE requires pairwise cross-condition comparison on identical timelines, evaluable Day-1 sequences, and frozen scaler availability—the same cohort used in the locked Hybrid baseline and residual-pattern analysis. Using a different subset would break comparability with published baseline metrics.

## Why κ is fixed before GRU results

κ sets injection amplitude relative to each container's train CPU volatility. Tuning κ after observing GRU performance would confound "learnability" with amplitude optimization and invalidate the controlled design.

## Why synthetic mechanisms cannot be tuned post-hoc

CSRLE is a **validation** of an existing pipeline under known ground truth. Redesigning B0/B1 or κ after seeing GRU outcomes would convert the experiment into generator optimization, not pipeline testing. Pre-specified failure logic applies instead.

## Positive control (B0)

B0-LC provides deterministic, strongly autocorrelated structure (~4 h period). If Prophet **absorbs** this into ŷ_B^P, retention metrics (VS high, ρ(N,I) low) flag unsuitability. If **clip boundary** violations are excessive, the generator gate fails without redesign.

## Stochastic condition (B1)

B1-SNAR adds nonlinear temporal dependence with innovation—closer to realistic residual complexity while remaining reproducible.

## Negative controls (C0, C1)

Same marginal injection scale as B0/B1 but **no temporal dependence** in z. Any Hybrid advantage on C0/C1 would suggest overfitting or pipeline artifacts, not structure learning.

## Pre-specified failure logic

| Failure | Action |
|---------|--------|
| B0 generator gate | Document; no B0 GRU; do not redesign B0 |
| B0 retention gate | Document unsuitable positive control; no B0 GRU |
| B1 generator/retention gate | Same protocol; no B1 GRU if failed |
| Condition A reproduction | Do not interpret synthetic conditions |
| Clip gate cohort failure | Synthetic config not frozen; synthetic GRU blocked |

## Expected outcome cases

| Outcome | Interpretation (under controlled condition) |
|---------|---------------------------------------------|
| B0 retention pass, GRU beats Prophet & C0 | Consistent with pipeline learning deterministic residual structure |
| B0 retention pass, GRU weak | Consistent with architectural/training limitation under known structure |
| B0 retention fail (Prophet absorbs) | B0 unsuitable as positive control under this Prophet config |
| Clip gate fail | κ·σ interaction with low/high CPU containers violates boundary assumptions |
| C0 ≈ C1 ≈ B0/B1 for Hybrid | Suggests injection not driving GRU signal |

## Current pre-GRU status (2026-07-25)

- **Generator clip gate:** FAILED all synthetic conditions (~9–11% containers exceed 1% clip-rate threshold; limit 5%).
- **B0/B1 retention:** PASSED — Prophet did **not** absorb B0 periodic structure (VS ≈ 0.002).
- **B0/B1 GRU:** Blocked pending generator gate (per protocol), not retention failure.
- **Frozen config:** Not written.

Authoritative artifact: `experiments/synthetic_residual_learnability_2026-07-25_163013/synthetic_validation/pre_gru_summary.json`
