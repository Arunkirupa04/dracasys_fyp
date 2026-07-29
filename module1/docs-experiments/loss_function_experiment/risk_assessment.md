# Risk Assessment

## Threats to internal validity

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Control arm fails to reproduce Stage 2 B0** | High | Mandatory reproduction gate before treatment comparison; abort if failed |
| **Implementation bug in DA-MSE gradient** | High | Unit tests on σ̂ gradient sign (under-dispersed batch → penalty decreases as amplitude increases); compare to manual finite differences |
| **Different early-stopping epochs confound comparison** | Medium | Report best epoch and val loss curves; pre-specify that epoch difference is expected and not a flaw if metrics compared at best checkpoint |
| **Per-sequence std unstable for flat y** | Medium | Exclude or flag containers with σ_y < 1e−4 (should be rare on B0); document count; primary analysis on full 99 with sensitivity excluding flagged |
| **Leakage via scaler misfit** | High | Reuse exact Stage 2 scaler protocol — train only |

## Threats to external validity

| Risk | Severity | Mitigation |
|------|----------|------------|
| **B0-specific gains do not transfer to real data** | Medium | Optional Condition A confirmatory arm |
| **Synthetic linear residual structure dominates** | Medium | Acknowledge Ridge analysis — even if H₁ holds, claim is **objective enables linearly structured residuals**, not universal nonlinear learning |
| **Single λ = 1.0 may be suboptimal** | Low (by design) | Accept suboptimality — hypothesis test, not tuning; document in future_scope |
| **99 containers may be insufficient for small effects** | Low | Stage 2 detected Δr ≈ 0.035 B0−C0 with CI excluding 0; same n adequate for similar effect sizes |

## Threats to construct validity

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Std ratio conflates amplitude with shape** | Medium | Pair with Pearson r — require both for success |
| **Pearson r unstable on low-variance y** | Medium | Same σ_y floor flag; report active-injection subset |
| **MSE and MAE decoupling misread as success** | Medium | MAE guardrail S3 mandatory |
| **Ridge ceiling misinterpreted as DA-MSE target** | Low | Document Ridge as reference only, not required threshold |

## Threats to conclusion validity

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Multiple comparisons inflate false positives** | Low | Single primary loss; two primary metrics (r, std ratio) pre-specified; bootstrap CIs |
| **Post-hoc use of diagnostics for hypothesis fishing** | Medium | Freeze rule: horizon ERR / band std ratio **excluded** from success_failure_criteria.md |
| **λ tuning pressure after null result** | Medium | λ = 1.0 frozen; document in future_scope; reject post-hoc sweeps |

## Operational risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Overwrite frozen CSRLE artifacts** | Critical | New experiment directory only; read-only mounts in scripts |
| **Accidental Prophet retrain** | High | Load frozen Prophet outputs or cache from Stage 2 pipeline |
| **Scope creep into loss sweep** | Medium | Protocol lock — DA-MSE only; future_scope for alternatives |
| **Thesis timeline overrun** | Medium | Minimum viable: B0 two-arm only (~2 GRU trains) |

## Risk matrix summary

| Category | Top risk | Residual risk after mitigation |
|----------|----------|-------------------------------|
| Internal | Control reproduction | Low if gate enforced |
| External | Real-data generalization | Medium — optional A arm |
| Construct | Amplitude vs shape | Low if S1+S2 jointly required |
| Conclusion | Post-hoc moving thresholds | Low if config frozen pre-run |

## Abort criteria (stop experiment)

1. LFHE-MSE-B0 fails reproduction gate after **one** pipeline fix attempt → revise implementation, not hypothesis.
2. DA-MSE training diverges (NaN loss) on > 10% epochs → **design revision** (check λ, ε, gradient).
3. Discovered bug in Stage 2 data loader → fix and restart both arms; do not interpret partial runs.

## Residual acceptance

**Proceed to implementation** if team accepts:

- B0-only primary is sufficient for thesis hypothesis chapter.
- λ = 1.0 fixed without sensitivity sweep.
- Ridge remains read-only reference, not retrained competitor inside LFHE.
