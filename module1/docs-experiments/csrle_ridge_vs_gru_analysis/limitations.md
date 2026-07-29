# Limitations

1. **Ridge reproduction, not saved Stage 2 Ridge artifact** — Stage 2 did not persist Ridge coefficients; analysis refits Ridge with frozen protocol (α=1.0). Predictions should match Stage 2 baseline CSV within floating-point tolerance.

2. **Prophet refit at inference** — Per-container Prophet is refit when reconstructing residuals; stochastic MCMC may cause tiny numerical differences vs Stage 2 run. GRU weights are loaded from frozen files.

3. **Day-1 only** — Same scope as CSRLE Stage 2 evaluation.

4. **Linear diagnostics on nonlinear injection** — B0 generator is nonlinear in state space; Prophet + residual projection may yield **linearly predictable lag structure** even when full dynamics are nonlinear.

5. **Entropy metrics** — Implemented without external `antropy`; parameters fixed (permutation order 3, sample entropy m=2).

6. **Explanatory, not causal** — Observational comparison of two fixed predictors on frozen data; cannot isolate single root cause.

7. **No new experiments recommended** — Per protocol, this analysis bounds interpretation of completed CSRLE only.
