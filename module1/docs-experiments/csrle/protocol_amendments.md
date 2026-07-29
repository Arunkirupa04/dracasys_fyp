# CSRLE Protocol Amendments

Chronological record of approved deviations from the originally locked CSRLE protocol.

---

## Amendment 1 — Boundary-safe injection (2026-07-25)

### Status

**Approved by supervisor.** Implemented as Protocol **v2**. Pre-GRU validation rerun required before any synthetic GRU training.

### Prior run preserved (do not modify)

`experiments/synthetic_residual_learnability_2026-07-25_163013/`

**Version 1 outcome:** κ=0.10 target + hard clip only → temporal generators valid, Prophet retention passed, **clip/boundary gate FAILED** (~9–11% containers >1% clip rate; worst ~50% on near-idle containers).

### Motivation

Near-idle containers (y_base ≈ 0) with zero-mean oscillating N_target produce frequent negative raw values clipped at 0%, distorting waveform shape and effective variance. This is a **data-generation feasibility** issue discovered by the pre-specified pre-GRU clip gate — **not** a GRU-performance-driven change (no synthetic GRU had been trained).

### What changed

| Aspect | v1 | v2 (amendment) |
|--------|----|----------------|
| Injection | N = κσz′, clip(y_base+N) | N_target = κσz′; N_eff = α_c·N_target; clip safeguard |
| α_c | n/a | Train-only, frozen per container |
| κ | 0.10 effective for all | **κ=0.10 target**; κ_eff = α_c·0.10 |
| C0/C1 matching | std(N_target B) | std(N_eff B); inherited α policy |
| Clip gate | Failed | Re-tested under v2 |

### What did NOT change

- B0-LC / B1-SNAR dynamics, coefficients, innovation variance
- seed=42, prewarm=200, κ target=0.10
- 99-container cohort (no exclusions)
- Prophet configuration, GRU methodology (not yet run)
- Temporal split, evaluation protocol definition
- Retention gate thresholds

### α_c mathematical rule (final)

For each container c, using **train timesteps only**:

For each train timestep t with N_target(t) ≠ 0:

```
α_t^max = (100 − y_train(t)) / N_target(t)   if N_target > 0
α_t^max = y_train(t) / (−N_target(t))        if N_target < 0
```

(Timesteps with N_target = 0 are unconstrained.)

```
α_c = min(1, min_t α_t^max)
```

Including α_t^max = 0 when y_train=0 and N_target<0 (near-idle + negative injection).

Application (train and val, α_c frozen):

```
N_eff,c(t) = α_c × N_target,c(t)
y_syn,c(t) = clip(y_base,c(t) + N_eff,c(t), 0, 100)
```

### Why strict min (not quantile)

Analysis on v1 N_target showed:

- Strict min including zero margins → **0% cohort clip rate** on full timeline
- 90.9% containers retain α=1; 8 near-idle containers require α=0 (no injection feasible without distortion)
- Quantile relaxations did not materially improve amplitude while introducing train-boundary violations on val

Strict min maximizes amplitude subject to **zero train clipping**, preserves waveform shape (uniform scaling), and is deterministic with no validation leakage.

### Effective κ

```
κ_eff,c = α_c × 0.10
```

Wording: *"κ=0.10 defined the target synthetic injection amplitude, with a train-derived per-container boundary-safety factor applied where required."*

### C0/C1 matching (final)

After B0/B1 boundary-safe generation:

```
σ_eff,B = std(N_eff,B on train)
z_std_train = std(z_iid on train)
N_target,C = (σ_eff,B / (α_B · z_std_train)) · z_iid
α_C = min(α_B, α_strict(y_train, N_target,C))
N_eff,C = α_C · N_target,C
```

**Rationale:** Inherits paired B boundary policy; train z normalization ensures exact std match when α_C=α_B; additional α_C reduction only when IID shape requires tighter train feasibility.

### Amplitude-preservation gate (pre-specified)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| fraction α=1 | ≥ 85% | Most cohort at full target amplitude |
| fraction α=0 | ≤ 10% | Near-idle exceptions bounded |
| fraction α<0.25 | ≤ 10% | Avoid cohort-wide collapse |
| median α | ≥ 0.95 | Typical container near full scale |
| mean κ_eff | ≥ 0.08 | Cohort mean injection remains meaningful |

**Fail action:** STOP — report boundary-safe scaling insufficient; do not auto-tune.

### Clip gate (unchanged threshold)

≤ 5% of containers may exceed 1% per-timestep clip rate (residual safeguard after α scaling).

### Prophet MAE reconciliation note

CSRLE v1 reported Day-1 Prophet MAE ≈ 1.733; residual-pattern analysis reported ≈ 1.847 for **full validation window**. v2 records both explicitly.

---

## Amendment history index

| ID | Date | Summary | Reference run |
|----|------|---------|---------------|
| — | 2026-07-25 | Original v1 pre-GRU | `..._163013` (failed clip gate) |
| 1 | 2026-07-25 | Boundary-safe α_c | v2 run `..._164307` |
