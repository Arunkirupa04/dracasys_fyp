# Background

## Established findings (frozen — not re-tested here)

### 1. Hybrid outperforms Global GRU

The Hybrid Prophet + GRU module achieves lower Day-1 CPU forecasting error than Global GRU under the locked Alibaba evaluation protocol (99 evaluable containers, 96-step holdout).

### 2. Peak-aware learning provides limited benefit

Timestep-weighted MSE (peak-aware training) did not yield meaningful Hybrid or Global GRU improvements over uniform MSE. Objective modifications focused on **peak weighting** are therefore deprioritized for this experiment.

### 3. Prophet residuals retain temporal structure

Post-Prophet validation residuals on real Alibaba data exhibit short-lag autocorrelation (prior diagnostic mean |ACF|₁₋₁₀ ≈ 0.20). Prophet removes slow structure but leaves **forecastable short-horizon residual dynamics**.

### 4. CSRLE: weak but non-zero GRU residual learning

The Controlled Synthetic Residual Learnability Experiment (CSRLE, protocol v2) demonstrated:

| Condition | Cohort mean Pearson r | Cohort mean std ratio |
|-----------|----------------------|------------------------|
| A (real) | 0.016 | 0.050 |
| B0 (positive control) | 0.084 | 0.073 |
| C0 (IID control) | 0.049 | 0.043 |

B0 > C0 on correlation and std ratio (paired bootstrap CIs exclude zero for r and std ratio). R² remained negative; CPU Hybrid − Prophet MAE deltas were negligible.

**Interpretation:** The fixed Hybrid GRU **can** learn some controlled structure, but **weakly** — stronger than matched IID noise, insufficient for strong shape fidelity or CPU gains.

### 5. Ridge analysis: MSE-compatible MAE, incompatible correlation

Read-only analysis of frozen CSRLE Stage 2 B0 (`docs/csrle_ridge_vs_gru_analysis/`) established:

| Metric | GRU (MSE) | Ridge (linear 96→96) |
|--------|-----------|----------------------|
| Residual Pearson r | 0.084 | **0.158** |
| Std ratio | **0.073** | **0.220** |
| Residual MAE | ~0.094 | ~0.094 |

Additional findings:

- B0 Prophet residuals behave as **short-memory linear processes** (AR(1) R² ≈ 0.20; polynomial trend adds ~0 beyond AR(2)).
- GRU and Ridge have **nearly identical spectral correlation** (~0.318 cohort mean) — Ridge wins on **amplitude and lag-linear tracking**, not frequency recovery.
- GRU predictions are **conservative, low-amplitude corrections** — adequate for MSE, inadequate for temporal shape.
- Condition A shows the **same variance-collapse pattern** (std ratio ≈ 0.05).

## Gap motivating this experiment

Ridge demonstrates that **higher dispersion and correlation are achievable on identical B0 residual inputs** without changing Prophet or data. The GRU under MSE does not reach this ceiling.

Two explanations compete:

| Explanation | Claim |
|-------------|-------|
| **Architecture-limited** | GRU capacity, depth, or inductive bias cannot represent the lag-linear map Ridge captures. |
| **Objective-limited** | MSE rewards accurate mean error but **does not penalize under-dispersion**; the optimizer settles on shrunk, near-flat corrections with acceptable MAE. |

CSRLE + Ridge analysis **partially supports objective-limited**: identical MAE with divergent correlation implies MSE does not enforce shape/dispersion matching. It does **not** prove MSE is the sole cause — that requires a **prospective objective swap**.

## Literature context

- **MSE vs correlation objectives:** Joint MSE+PCC training often plateaus because MSE gradients dominate correlation gradients as prediction variance increases (recent analysis of attention regressors; dispersion-normalized PCC proposed to restore gradient balance). This supports treating **dispersion** as a first-class training target, not an emergent property of MSE.
- **Heteroscedastic NLL pitfalls:** Gaussian NLL can degrade mean fits by down-weighting hard examples via variance — but fixing this typically requires **learned variance heads** (architecture change). Not suitable for isolated loss hypothesis testing.
- **Faithful / β-NLL regression:** Variance-aware losses exist, but dual-output heads confound “loss vs capacity.” A **single-output dispersion penalty on MSE** is the minimal intervention aligned with observed under-dispersion.

## Empirical ceiling for dispersion recovery (reference only)

On B0, Ridge cohort mean std ratio ≈ **0.22** (not 1.0). DA-MSE is **not expected to reach 1.0**; partial movement from ≈ 0.07 toward ≈ 0.15–0.22 would support the hypothesis. Full recovery to Ridge levels would suggest objective was the dominant bottleneck; zero movement would reject it.
