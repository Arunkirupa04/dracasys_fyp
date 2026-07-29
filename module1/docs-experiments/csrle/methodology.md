# CSRLE Methodology

This document separates **locked existing Hybrid methodology** (unchanged) from **new CSRLE-specific methodology** (synthetic injection and validation).

---

## LOCKED EXISTING METHODOLOGY (read-only parity)

The following are frozen and reused without modification:

- **Temporal split:** frozen train/val parquets per container
- **Prophet:** `daily_seasonality=True`, `weekly_seasonality=False`; fit on train `cpu_scaled`; Day-1 horizon = 96 steps (15 min → 24 h)
- **Residual:** `R(t) = y(t) − ŷ_P(t)` in real CPU %
- **GRU (deferred until post-review):** 96 → 96 sequence forecasting on normalized residuals; architecture/training from `utils/hybrid_training.py`
- **Evaluation:** Day-1 MAE/RMSE/MAPE; container-level aggregation; same 99 evaluable containers as baseline
- **Reference metrics:** frozen baseline Hybrid MAE ≈ 1.746 (`experiments/baseline_reference_2026-07-14/`)

CSRLE tests this pipeline; it does **not** optimize it.

---

## NEW CSRLE-SPECIFIC METHODOLOGY

### Condition A reproduction (Stage 1 — completed)

Train a **new** experiment-local GRU on frozen real data using the same
methodology as `utils/hybrid_training.train_hybrid_gru`. Do not reuse
`models/hybrid_gru.keras` as the Condition A model.

Stage artifacts: `stage1_control_reproduction/` (separate from pre-GRU validation outputs).

### Conditions B0, B1, C0, C1 — Synthetic injection (Protocol v2)

**v1 (superseded for generation):** N = κσz′ with hard clip only — failed clip gate (see v1 run 163013).

**v2 (current):** Target amplitude κ=0.10 with train-derived boundary-safety α_c:

```
N_target = κ · σ_cpu^train · z′
α_c = min(1, min_t α_t^max)   [train only]
N_eff = α_c · N_target
y_syn = clip(y_base + N_eff, 0, 100)
κ_eff = α_c · 0.10
```

See [protocol_amendments.md](protocol_amendments.md) for full α_c definition and C0/C1 matching.

### B0-LC — Deterministic limit-cycle positive control

```
[xr, yr]ᵀ = R · [x_{t−1}, y_{t−1}]ᵀ,   R = [[0.92, −0.38], [0.38, 0.92]]
r* = 0.70
[x_t, y_t]ᵀ = tanh((r*/r) · [xr, yr]ᵀ)
z_t = x_t / √2
```

Initialization: container-seeded uniform in [−0.4, 0.4] for (x₀,y₀) and (x₁,y₁).

Expected ~16-step period (~4 h at 15 min/step).

### B1-SNAR — Stochastic nonlinear AR(2)

```
z_t = 0.55·z_{t−1} − 0.15·z_{t−2} + 0.08·z_{t−1}·z_{t−2} + η_t
η_t ~ N(0, 0.15²)
```

Continuous seeded RNG stream across train+val (no reset at boundary).

### C0 / C1 — IID negative controls

```
z_t ~ N(0, 1)   (container-seeded, stream C0 or C1)
N_t = std(N_train^B) · z_t
```

C0 matched to B0 train std(N); C1 matched to B1 train std(N).

### Prophet retention (pre-GRU diagnostic)

Separate Prophet fits:

- **P_A:** Condition A train → val Day-1 forecast `ŷ_A^P`
- **P_B:** Condition B train → val Day-1 forecast `ŷ_B^P`

On shared val timestamps (real calendar; same length):

```
ΔP(t) = ŷ_B^P(t) − ŷ_A^P(t)
N(t)  = y_B(t) − y_A(t)        (post-clip effective injection)
I(t)  = N(t) − ΔP(t)
R_A(t)= y_A(t) − ŷ_A^P(t)
R_B(t)= y_B(t) − ŷ_B^P(t)
```

**Identity (verified):** `I(t) = R_B(t) − R_A(t)`

Retention metrics: ρ(N,I), VS = Var(ΔP)/Var(N), VR = Var(I)/Var(N), ACF/Ljung–Box on R_B.

### Leakage prevention

- z-normalization uses **train segment only**
- Condition MinMaxScaler fit on **train y_syn only**
- Prophet fit on **train only**; val is forecast target
- GRU training uses train residuals only (Stage 2: per-condition independent models)
- Frozen base data checksums recorded at experiment start

### Evaluation protocol (Stage 2 — executed)

Implemented in `utils/csrle/stage2_*.py` and `scripts/run_csrle_stage2_synthetic_gru.py`:

- Prophet vs Hybrid CPU metrics per condition (Day-1)
- Residual prediction metrics (MAE, RMSE, Pearson r, R², std ratio)
- ZERO / PERSISTENCE / RIDGE / GRU baselines (same 96→96 protocol)
- Within-trajectory and horizon-wise (h=1,4,12,24,48,96) / horizon-band metrics
- Synthetic component recovery (Ĝ vs R_B primary; vs N, I supporting) for B0/B1
- Paired container-level bootstrap (B0 vs C0, B1 vs C1; full + active-injection subsets)
- Condition A from Stage 1 (not retrained)

**Results:** `stage2_synthetic_gru/stage2_final_report.json`

---

## Authoritative configuration

Pre-specified gates and constants: `utils/csrle/config.py`  
Run configuration snapshot: `experiments/synthetic_residual_learnability_2026-07-25_163013/synthetic_validation/pre_gru_summary.json`
