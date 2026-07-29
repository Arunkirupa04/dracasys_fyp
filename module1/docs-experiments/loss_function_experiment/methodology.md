# Methodology

## Design type

**Prospective, paired, single-factor controlled experiment.**

- **Factor:** Training loss (MSE vs DA-MSE).
- **Unit of analysis:** Container-level Day-1 residual trajectory (96 steps).
- **Blocking:** Same container appears in both arms (retrained separately; paired comparison on same val trajectories).

## Locked components (identical across arms)

### Prophet

- Same Prophet fitting procedure as CSRLE Stage 2 B0 and Hybrid Phase 3.
- Same train/validation temporal split per container.
- Same Day-1 Prophet forecasts used as residual baselines.
- **No re-tuning** of Prophet hyperparameters.

### GRU architecture

Identical to frozen Hybrid / CSRLE Stage 2:

```
Input: (batch, 96, n_features)  — n_features from frozen feature schema
GRU(256, return_sequences=True)
GRU(128, return_sequences=True)
GRU(64)
Dense(96)  — 96-step residual vector output
```

- Same activation, dropout (if any in Stage 2), and weight initialization seed.
- **No** variance head, **no** auxiliary outputs.

### Data

- **Primary:** CSRLE B0-LC frozen parquets (`data/b0_lc/train_syn.parquet`, `val_syn.parquet`) and injection manifest from `experiments/synthetic_residual_learnability_2026-07-25_164307/`.
- **Confirmatory (optional):** Condition A real Alibaba preprocessing outputs used in Stage 1 reproduction — same 99 containers.
- Same α map, same boundary-safe injection — **generators not re-run**.

### Sequence construction

- Input window: **96** past residual steps.
- Target: **96** future residual steps (aligned with CSRLE Stage 2 sliding windows).
- Train-only scalers and residual normalization (`residual_stats.pkl` protocol) — fit on train split only per condition.
- 80/20 chronological sequence split within train for early stopping (same as Hybrid Phase 3).

### Optimizer and schedule

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | Adam | Hybrid Phase 3 |
| Learning rate | **Same as Stage 2 B0** (frozen from `training_metadata.json`) | No retuning |
| Batch size | Same as Stage 2 B0 | No retuning |
| Max epochs | Same cap as Stage 2 B0 | No retuning |
| Early stopping | `monitor="val_loss"`, patience, restore best weights | **val_loss uses arm-specific loss** |
| Random seed | Fixed global seed (see reproducibility.md) | New runs only |

**Note:** Early stopping monitors the **training objective of each arm** (MSE val loss vs DA-MSE val loss). This is correct — stopping criterion must match optimized objective. Epoch counts may differ between arms; report best epoch and loss curves.

### Inference

- Identical to CSRLE Stage 2: load best weights, apply train-fitted scalers, Prophet + GRU sum for CPU forecast.
- **No** test-time calibration or post-hoc variance scaling.

## Variable component (sole deliberate change)

| Arm | Training loss |
|-----|---------------|
| Control | Standard MSE on 96-step residual vector |
| Treatment | DA-MSE (see loss_function_selection.md) |

### Frozen λ (dispersion weight)

**λ = 1.0** is locked in the protocol before implementation. LFHE is a **hypothesis test**, not hyperparameter optimization. Searching for an optimal λ would introduce a second experimental variable and weaken causal interpretation of the dispersion supervision effect. Full rationale: [loss_function_selection.md](loss_function_selection.md#explicit-λ-justification-λ--10).

## Diagnostic analyses (descriptive — v1.1)

Post-evaluation diagnostics extend interpretability **without** changing hypotheses or success criteria:

| Analysis | Document |
|----------|----------|
| Horizon-wise variance recovery | [diagnostic_analysis.md](diagnostic_analysis.md) |
| Residual energy recovery (ERR) | [diagnostic_analysis.md](diagnostic_analysis.md) |

**Not used for:** model selection, early stopping, hypothesis accept/reject.

## Training runs required

| Run ID | Condition | Loss | Purpose |
|--------|-----------|------|---------|
| LFHE-MSE-B0 | B0-LC | MSE | Control (should match Stage 2 B0 within reproduction tolerance) |
| LFHE-DA-B0 | B0-LC | DA-MSE | Primary treatment |
| LFHE-MSE-A | A (optional) | MSE | Confirmatory control |
| LFHE-DA-A | A (optional) | DA-MSE | Confirmatory treatment |

**Minimum viable experiment:** LFHE-MSE-B0 + LFHE-DA-B0 only.

## Reproduction check (control arm)

Before interpreting treatment effects, LFHE-MSE-B0 must **reproduce** frozen Stage 2 B0 metrics within pre-specified tolerance:

| Metric | Tolerance vs Stage 2 B0 |
|--------|-------------------------|
| Cohort mean Pearson r | \|Δ\| ≤ 0.015 |
| Cohort mean std ratio | \|Δ\| ≤ 0.015 |
| Cohort mean residual MAE | \|Δ\| ≤ 0.005 |

Failure to reproduce invalidates the control arm — **do not compare DA-MSE** until MSE control passes.

## Analysis populations

| Population | n | Use |
|------------|---|-----|
| Full cohort | 99 | Primary inference |
| Active injection (α > 0) | 91 | Secondary (CSRLE convention) |
| α = 0 containers | 8 | Descriptive only — no injection signal |

## Ethical / artifact constraints

- Do **not** modify `experiments/synthetic_residual_learnability_2026-07-25_164307/` in place.
- Write new artifacts under `experiments/loss_function_hypothesis_<timestamp>/`.
- Treat Stage 2 B0 model as **read-only reference**, not overwrite target.
