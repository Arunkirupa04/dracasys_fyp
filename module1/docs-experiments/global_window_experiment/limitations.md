# GGTCE Limitations

1. **Single seed (42)** — no multi-seed variance estimate for window effect.
2. **Sequence reduction** — longer windows reduce training samples non-uniformly across containers.
3. **Frozen architecture** — no capacity adjustment for longer inputs; GRU may not fully exploit extended context.
4. **TMA memory score proxy** — uses squared-ACF capture fraction, not direct causal memory measure.
5. **Day-1 evaluation only** — matches Global baseline protocol; multi-day horizon not tested.
6. **No G384** — excluded per protocol; upper bound of useful context unknown.
7. **Cohort fixed at 99 containers** — Alibaba trace subset; generalization unverified.
