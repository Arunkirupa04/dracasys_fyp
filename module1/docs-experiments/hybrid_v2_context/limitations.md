# Limitations

1. **Train window short** (~6.5 days) — peak/regime stats noisy
2. **No seed locking** — cross-variant comparison fair; V0 may not bitwise-match frozen baseline weights
3. **Day-1 only** — Day-2 recursive multi-feature path simplified
4. **Single ablation ladder** — no independent feature-only models (by design)
5. **CPU-only context** — multivariate Alibaba telemetry unavailable
6. **MSE loss frozen** — LFHE showed objective may dominate dispersion
