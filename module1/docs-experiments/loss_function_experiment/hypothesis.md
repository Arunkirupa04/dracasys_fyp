# Hypotheses

## Primary hypothesis (alternative)

**H₁ (Objective-limited variance collapse):**  
Under identical Hybrid architecture, Prophet decomposition, CSRLE B0 data, and training protocol, replacing MSE with **Dispersion-Augmented MSE (DA-MSE)** will produce residual predictions with **significantly higher within-trajectory std ratio** and **significantly higher residual Pearson r** compared to MSE-trained controls, without material MAE degradation.

*Mechanism:* MSE alone permits low-variance predictions that minimize squared error on zero-mean, small-amplitude residuals. An explicit under-dispersion penalty removes a local optimum that Ridge avoids (linear map preserves more amplitude).

## Null hypothesis

**H₀ (Not objective-primary):**  
DA-MSE and MSE produce **equivalent** std ratio and Pearson r distributions on B0 (paired differences not statistically distinguishable), indicating variance collapse is **not primarily attributable** to the MSE objective alone.

*Implication if H₀ holds:* Architecture, sequence construction, residual scaling, early stopping on MSE validation loss, or fundamental linearity of B0 targets dominate — objective swap is insufficient.

## Secondary confirmatory hypothesis

**H₁′ (Generalization to real data):**  
If H₁ holds on B0, the same DA-MSE vs MSE comparison on **Condition A** will show **directionally consistent** std ratio and r improvements (magnitude may be smaller given weaker absolute structure).

**H₀′:** Condition A shows no improvement — B0 gains are synthetic-specific or an artifact of injected structure.

## Non-hypotheses (explicitly not tested)

| Statement | Status |
|-----------|--------|
| DA-MSE will make Hybrid beat Global GRU | Not claimed |
| DA-MSE will yield positive R² on B0 | Not required for H₁ |
| DA-MSE will improve Day-1 CPU MAE | Not expected; decoupling is informative |
| DA-MSE replaces MSE in production Hybrid | Out of scope |
| Nonlinear B0 injection is fully learnable | Already bounded by CSRLE + Ridge |

## Falsifiable predictions under H₁

1. Cohort mean std ratio (B0) increases from ≈ 0.07 to **≥ 0.12** (pre-specified minimum effect).
2. Paired bootstrap mean Δr (DA-MSE − MSE) **> 0** with 95% CI excluding 0.
3. ≥ 60% of containers show DA-MSE std ratio > MSE std ratio (mirroring Ridge vs GRU r comparison style).
4. Residual MAE increase **≤ 0.010** (absolute, scaled space) vs MSE control — dispersion gain is not bought by large error inflation.

## Predictions under H₀

1. Std ratio remains in **0.06–0.08** band (MSE-control equivalence).
2. Pearson r remains in **0.07–0.09** band.
3. Training curves differ in loss scale but **validation shape metrics** do not separate.

## Interpretation matrix

| Std ratio | Pearson r | MAE vs MSE | Interpretation |
|-----------|-------------|------------|----------------|
| ↑ significant | ↑ significant | stable | **Support H₁** — objective-primary |
| ↑ significant | ↔ | stable | **Partial support** — amplitude restored, shape not |
| ↔ | ↑ significant | stable | **Inconclusive / reject H₁** — r gain without dispersion contradicts mechanism |
| ↔ | ↔ | ↑ large | **Reject H₁** — objective not primary |
| ↑ | ↑ | ↑ large | **Inconclusive** — possible over-dispersion tradeoff |
