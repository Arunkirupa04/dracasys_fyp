# Limitations

1. **Single challenger** — Only R3 is tested against R0. R2 and other scalings (e.g. quantile, log-transform) are out of scope.

2. **No TF seed** — Matches frozen baseline for fairness, but exact weight replication is not guaranteed. Cross-arm comparison remains valid under identical protocol.

3. **Global statistics** — Both R0 and R3 use pooled train statistics. Per-container robust scaling (R2) was rejected in Stage 0 but could interact differently with optimisation.

4. **Diagnostic proxies ≠ learnability proof** — Stage 0 ACF equality does not guarantee identical GRU loss landscapes; RRE v1.0 empirically tests this.

5. **Day-1 evaluation only** — Consistent with frozen Hybrid protocol; multi-horizon effects not studied.

6. **One train/val split** — Fixed research cohort; no cross-validation.

7. **MSE loss sensitivity** — Robust scaling may interact differently with alternative losses (e.g. Huber); not tested here.

8. **Prophet refit per container at inference** — Computational cost limits experiment size; no ensembling.
