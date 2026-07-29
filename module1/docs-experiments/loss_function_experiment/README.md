# Loss-Function Hypothesis Experiment (LFHE)

## Status

**Phase:** **APPROVED IN PRINCIPLE** — protocol v1.1 frozen (diagnostic extensions)  
**Implementation:** Not started  
**Training:** Authorized after `lfhe_config_frozen.json` is written  

## Purpose

Test whether **variance collapse** in the fixed Hybrid Prophet + GRU residual learner is **primarily caused by the MSE training objective** rather than by the Hybrid architecture itself.

This is a **hypothesis-testing experiment**, not an accuracy-optimization or hyperparameter-tuning study.

## Research question

> Is the observed variance collapse primarily caused by the optimization objective (MSE loss) rather than by the Hybrid architecture itself?

## Recommended treatment

**Dispersion-Augmented MSE (DA-MSE)** — MSE plus a **one-sided log-variance matching penalty** that penalizes under-dispersed predictions.

See [loss_function_selection.md](loss_function_selection.md) for full justification.

## Experiment arms

| Arm | Loss | Role |
|-----|------|------|
| **Control** | Standard MSE | Reproduces frozen CSRLE Stage 2 B0 protocol |
| **Treatment** | DA-MSE (λ = 1.0, frozen) | Single approved objective change |

**λ = 1.0** is frozen **before** implementation — a deliberate hypothesis-test choice, not a tuned hyperparameter. Searching for optimal λ would add a second experimental factor and weaken causal interpretation; sensitivity analysis is deferred to [future_scope.md](future_scope.md).

## Diagnostic extensions (v1.1 — descriptive only)

| Analysis | Purpose |
|----------|---------|
| [Horizon-wise variance recovery](diagnostic_analysis.md#1-horizon-wise-variance-recovery) | Std ratio by bands 1–12, 13–24, 25–48, 49–72, 73–96 — detect progressive horizon collapse |
| [Residual energy recovery](diagnostic_analysis.md#2-residual-energy-recovery) | ERR = Σ(ŷ²)/Σ(y²) — quantify reproduced residual signal mass |

These **do not** change hypotheses, success criteria, or the DA-MSE formulation. See [diagnostic_analysis.md](diagnostic_analysis.md).

## Scope lock

| Component | Change allowed? |
|-----------|----------------|
| GRU architecture | **No** |
| Prophet pipeline | **No** |
| CSRLE synthetic generators / data | **No** |
| Evaluation metrics / holdout protocol | **No** |
| Training loss (residual GRU only) | **Yes — DA-MSE vs MSE** |

## Primary evaluation setting

**CSRLE B0-LC** (positive control, 99 containers) — same frozen experiment as Stage 2.

**Confirmatory (optional, same protocol):** Condition A (real-data Hybrid reproduction).

## Documentation index

| Document | Purpose |
|----------|---------|
| [research_question.md](research_question.md) | Formal RQ and scope boundaries |
| [background.md](background.md) | Prior findings motivating the experiment |
| [hypothesis.md](hypothesis.md) | Null/alternative hypotheses |
| [methodology.md](methodology.md) | Locked training and inference methodology |
| [loss_function_selection.md](loss_function_selection.md) | Loss choice, rejections, formulation |
| [experimental_design.md](experimental_design.md) | Arms, conditions, sample size, controls |
| [evaluation_protocol.md](evaluation_protocol.md) | Primary metrics, statistics, reporting |
| [diagnostic_analysis.md](diagnostic_analysis.md) | Horizon variance + energy recovery (descriptive) |
| [success_failure_criteria.md](success_failure_criteria.md) | Decision rules |
| [risk_assessment.md](risk_assessment.md) | Threats to validity and mitigations |
| [implementation_plan.md](implementation_plan.md) | Post-approval build steps (no code here) |
| [reproducibility.md](reproducibility.md) | Seeds, configs, artifact layout |
| [future_scope.md](future_scope.md) | Explicitly out-of-scope follow-ups |

## Relation to prior work

| Prior artifact | Relationship |
|----------------|--------------|
| CSRLE Stage 2 (`experiments/synthetic_residual_learnability_2026-07-25_164307/`) | Control arm reference; data and eval frozen |
| Ridge vs GRU analysis | Motivation; Ridge std ratio ≈ 0.22 sets empirical ceiling for dispersion recovery |
| Peak-aware / Global GRU experiments | Out of scope |

## Approval gate

Implementation may begin when:

1. Protocol v1.1 approved (**done in principle**).
2. Frozen config file `config/lfhe_config_frozen.json` is written.
3. No existing CSRLE / Hybrid artifacts are modified.

**Ready for implementation:** experimental design unchanged; diagnostic analyses and λ documentation strengthened only.
