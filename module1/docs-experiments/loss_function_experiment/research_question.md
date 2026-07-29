# Research Question

## Primary research question

> **Is the observed variance collapse in Hybrid GRU residual predictions primarily caused by the optimization objective (MSE loss) rather than by the Hybrid architecture itself?**

## Operationalization

“Variance collapse” means the GRU predicts residual sequences whose **within-trajectory standard deviation** is far smaller than the actual residual standard deviation — quantified by the **std ratio** (predicted std / actual std) already used in CSRLE Stage 2.

“Primarily caused by MSE” means that **replacing only the training objective** (keeping architecture, Prophet, data, and evaluation identical) **materially restores dispersion and correlation** toward levels demonstrated achievable by a linear baseline (Ridge, std ratio ≈ 0.22 on B0).

“Hybrid architecture itself” refers to the **fixed** two-layer stacked GRU residual corrector (256→128→64), 96-step input window, Prophet decomposition, and residual scaling pipeline — unchanged from CSRLE Stage 2 and Hybrid Phase 3.

## Sub-questions (answered by the same experiment)

1. Does an objective that explicitly penalizes under-dispersion increase std ratio without architectural change?
2. Does any correlation gain accompany dispersion recovery, or is amplitude restored without shape improvement?
3. Does CPU-level Hybrid performance change materially, or do residual-shape gains remain decoupled from MAE (as in Ridge analysis)?

## What this experiment is

- A **controlled causal probe** of the training objective.
- A **single-loss comparison** (MSE vs DA-MSE).
- A **CSRLE B0-first** hypothesis test with optional Condition A confirmation.

## What this experiment is not

- Not a general loss-function benchmark (no Huber, quantile, focal, NLL sweep).
- Not a hyperparameter search (λ, learning rate, architecture fixed).
- Not a claim that Hybrid should replace Prophet or Ridge in production.
- Not a reopening of CSRLE synthetic generator design or Stage 1/2 artifact edits.

## Primary success criterion (conceptual)

If DA-MSE **significantly increases std ratio and residual Pearson r** on B0 relative to MSE control, while architecture and data are unchanged, the evidence **supports** the hypothesis that MSE (specifically its lack of dispersion incentive) is a **primary contributor** to variance collapse.

## Primary failure criterion (conceptual)

If std ratio and r remain at MSE-control levels (≈ 0.07 and ≈ 0.08 on B0), the evidence **rejects** the hypothesis that objective change alone is sufficient — implicating architecture, optimization dynamics, or target structure instead.
