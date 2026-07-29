# CSRLE Synthetic Generation

Complete specification for independent reproduction of B0, B1, C0, C1 synthetic series.

## Shared preprocessing (v2 boundary-safe)

1. **Base series:** Read-only `y_base(t)` in CPU % from frozen train/val parquets.
2. **Target injection (B0/B1):** `N_target(t) = κ · σ_cpu^train · z'(t)` with κ=0.10 target.
3. **Boundary safety (train-only):** `α_c = min(1, min_t α_t^max)` — see [protocol_amendments.md](protocol_amendments.md).
4. **Effective injection:** `N_eff(t) = α_c · N_target(t)`.
5. **Synthetic CPU:** `y_syn = clip(y_base + N_eff, 0, 100)`.
6. **Manifest N:** post-clip `y_syn − y_base`.
7. **Effective κ:** `κ_eff,c = α_c · 0.10`.

## RNG and seeds

- **Global seed:** 42 (`CSRLEConfig.global_seed`)
- **Per-container streams:** `container_rng(container_id, global_seed, stream_label)` in `utils/csrle/seeds.py`
- **B0 init:** Two uniform draws per container (`B0_init`, `B0_init2`) in [−0.4, 0.4]
- **B1/C0/C1:** Seeded normal draws; B1 uses one continuous stream across full timeline (train+val)

## Pre-warm (B0, B1)

200 virtual steps generated before the first observable step aligned to frozen timestamps. Pre-warm state is discarded; only post-warm z values map to data indices.

## Dimensionless generation → CPU injection

For B0 and B1:

```
z'_t = (z_t − mean(z_train)) / std(z_train)
N_target_t = κ · σ_cpu^train · z'_t
N_eff_t = α_c · N_target_t
y_syn_t = clip(y_base_t + N_eff_t, 0, 100)
```

For C0/C1 (after B0/B1):

```
σ_eff_B = std(N_eff_B on train)
N_target_C = (σ_eff_B / (α_B · std(z_train))) · z_iid
α_C = min(α_B, α_strict(y_train, N_target_C))
N_eff_C = α_C · N_target_C
```

## B0-LC (deterministic limit cycle)

**State update (t ≥ 2):**

```
xr = 0.92·x_{t−1} − 0.38·y_{t−1}
yr = 0.38·x_{t−1} + 0.92·y_{t−1}
r  = hypot(xr, yr)
x_t = tanh((0.70/r) · xr)
y_t = tanh((0.70/r) · yr)
z_t = x_t / √2
```

**Expected temporal properties:**

- Strong lag-1 autocorrelation in z (observed cohort mean ACF1 ≈ 0.924)
- Stable variance early vs late (ratio ≈ 1.0)
- Deterministic rollout RMSE ≈ 0 (exact recurrence)
- Pooled val FFT dominant period ≈ 15.7 steps ≈ **3.93 h** (authoritative: `prophet_retention_report.json`)

## B1-SNAR (stochastic nonlinear AR)

```
z_t = 0.55·z_{t−1} − 0.15·z_{t−2} + 0.08·z_{t−1}·z_{t−2} + η_t
η_t ~ N(0, 0.15²)
```

**Expected temporal properties:**

- Moderate ACF1 (cohort mean ≈ 0.477)
- Early/late train std ratio ≈ 1.0 (stationarity check)
- Ljung–Box rejection on R_B post-Prophet (retention phase)

## C0 / C1 (IID controls)

- **C0:** `matched_std = std(N_train)` from B0 per container
- **C1:** `matched_std = std(N_train)` from B1 per container
- Cohort mean |ACF1(z)| ≈ 0.03 (within IID gate threshold ≤ 0.05)

## Boundary handling

Hard clip to [0, 100] CPU %. Clip mask stored in manifest. **Pre-GRU finding:** 9–11% of containers exceed 1% per-timestep clip rate (see `validation_protocol.md`).

Worst offenders (clip rate > 0.48): `c_13308`, `c_14106`, `c_15035`, `c_15794` — typically low mean CPU with negative injection swings.

## Train/validation continuity

- Single continuous z stream per container (no RNG reset at train/val boundary)
- Normalization statistics computed from train z only, applied to full series
- Val injection uses same μ_z, σ_z, σ_cpu^train as train

## Reproducibility artifacts

| Artifact | Location |
|----------|----------|
| Full injection manifest | `experiments/.../data/ground_truth/injection_manifest.parquet` |
| Condition train/val parquets | `experiments/.../data/{condition}/train_syn.parquet`, `val_syn.parquet` |
| Generator config | `utils/csrle/config.py` |
| Implementation | `utils/csrle/generators.py`, `utils/csrle/dataset.py` |

## κ verification (pre-GRU run)

Train `std(N)` vs target `κ · σ_cpu^train`: cohort mean relative error ≈ 0 (machine precision). **PASS** for B0 and B1.

Authoritative: `experiments/synthetic_residual_learnability_2026-07-25_163013/synthetic_validation/generator_gate_report.json`
