# Experimental Design

## Overview

```
                    ┌─────────────────────────────────────┐
                    │     Frozen CSRLE B0-LC data         │
                    │  (99 containers, same Prophet preds) │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
    ┌──────────────────┐                         ┌──────────────────┐
    │  Arm A: MSE      │                         │ Arm B: DA-MSE    │
    │  (control)       │                         │ (treatment)      │
    │  λ_disp = 0      │                         │ λ = 1.0          │
    └────────┬─────────┘                         └────────┬─────────┘
             │                                            │
             └──────────────────┬─────────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Identical evaluation   │
                    │ (CSRLE Stage 2 metrics)│
                    └───────────────────────┘
```

## Independent variable

**Training loss function** (2 levels: MSE, DA-MSE).

## Dependent variables

See [evaluation_protocol.md](evaluation_protocol.md). Primary: **std ratio**, **residual Pearson r**. Secondary: R², MAE, CPU Hybrid − Prophet delta.

## Experimental units and pairing

- **Unit:** One container × one Day-1 validation trajectory (96 steps).
- **Pairing:** Each container evaluated under both arms (separate trained models, same val data).
- **n = 99** for full cohort tests; **n = 91** for active-injection subset.

## Conditions

| Condition | Role in LFHE | Train? |
|-----------|--------------|--------|
| **B0-LC** | **Primary** — positive control with known structure | Yes (both arms) |
| **A (real)** | **Optional confirmatory** — external validity | Yes (both arms) if approved |
| B1, C0, C1 | **Excluded** | No — not needed for objective hypothesis |

**Rationale for B0-only primary:** B0 is where (i) GRU vs Ridge dispersion gap is largest, (ii) CSRLE structure signal exists, (iii) objective hypothesis is most falsifiable. C0/C1 test generator discrimination (CSRLE Stage 2 already complete), not MSE mechanism.

## Control strategy

### Internal control

**MSE arm (LFHE-MSE-B0)** re-trains GRU with standard MSE under LFHE experiment directory. Must reproduce Stage 2 B0 metrics within tolerance — validates pipeline before treatment comparison.

### Historical reference (not re-trained)

| Reference | Use |
|-----------|-----|
| Stage 2 B0 GRU | Metric reproduction target |
| Stage 2 B0 Ridge baseline | Empirical dispersion ceiling (std ratio ≈ 0.22, r ≈ 0.158) |
| Stage 2 Condition A | Confirmatory baseline if optional arm run |

Ridge is **not re-trained** in LFHE — read-only ceiling from frozen CSV.

## Randomization and seeds

- Fixed global seed for weight initialization and sequence shuffling.
- **Same seed** across MSE and DA-MSE arms (fair comparison).
- Separate run directories prevent checkpoint collision.

## Blinding

Not applicable (automated metrics). Analysis script should compute paired comparisons before viewing plots to reduce confirmation bias during review.

## Sample size justification

n = 99 containers matches entire evaluable CSRLE cohort — same power as Stage 2 paired bootstrap (B0 vs C0: r difference CI width ≈ 0.06). No additional containers available without breaking frozen protocol.

## Exclusions

Same as CSRLE: containers failing preprocessing gates excluded identically (should remain 99/99 if using frozen splits).

## Multiplicity

| Comparison | Type | Adjustment |
|------------|------|------------|
| DA-MSE vs MSE on B0 (r) | Primary | Report 95% bootstrap CI; pre-specified threshold |
| DA-MSE vs MSE on B0 (std ratio) | Primary | Same |
| DA-MSE vs MSE on B0 (MAE) | Guardrail | No success claim if violated |
| Condition A (optional) | Confirmatory | **No multiplicity correction** — descriptive only |
| CPU metrics | Exploratory | Not used for hypothesis decision |
| Horizon-band std ratio | **Diagnostic** | Not used for hypothesis decision |
| Energy recovery ratio (ERR) | **Diagnostic** | Not used for hypothesis decision |

## Post-evaluation diagnostics (v1.1)

After primary metrics are computed, run **descriptive-only** analyses per [diagnostic_analysis.md](diagnostic_analysis.md):

1. Horizon-wise variance recovery (5 bands)
2. Residual energy recovery (ERR)
3. Notebook + PNG/PDF exports

**Not varied:** hypotheses, success thresholds, loss formulation, λ.

## Timeline (post-approval)

1. Write frozen config JSON.
2. Implement loss in isolated module (not modifying Stage 2 code paths in place).
3. Run LFHE-MSE-B0 → reproduction gate.
4. Run LFHE-DA-B0 → primary comparison.
5. Compute diagnostic analyses (horizon variance, energy recovery).
6. Generate `notebooks/lfhe_visualization.ipynb` + PNG/PDF plots.
7. Optional Condition A arms.
8. Generate report + update thesis discussion — **no artifact overwrites**.

## What is explicitly not varied

- Input features to GRU
- Window length (96)
- Prophet model
- Synthetic injection parameters (κ, α map)
- Evaluation holdout boundaries
- Optimizer type
- Learning rate (unless Stage 2 metadata ambiguous — then freeze literal value from B0 `training_metadata.json`)
