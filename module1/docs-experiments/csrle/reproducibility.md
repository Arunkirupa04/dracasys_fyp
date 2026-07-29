# CSRLE Reproducibility

## Environment

- Python: `./tf_metal_env/bin/python3` (project virtualenv)
- Key packages: pandas, numpy, prophet, scikit-learn, pyarrow
- OS tested: macOS (darwin 25.5.0)

## Seeds and constants

| Parameter | Value | Source |
|-----------|-------|--------|
| global_seed | 42 | `utils/csrle/config.py` |
| κ | 0.10 | `utils/csrle/config.py` |
| prewarm_steps | 200 | `utils/csrle/config.py` |
| CPU clip | [0, 100] | `CSRLEConfig` |

## Container cohort

- File: `data/selected_containers.npy` (read-only)
- Evaluable subset: 99 containers (see `pre_gru_summary.json`)

## Read-only inputs (checksums recorded per run)

```
data/train_df.parquet
data/val_df.parquet
data/scalers.pkl
data/selected_containers.npy
```

Checksums: `experiments/synthetic_residual_learnability_2026-07-25_163013/verification/frozen_integrity_checksums.json`

## Stage 2 — Synthetic GRU (B0/B1/C0/C1)

```bash
cd /Users/lehaan/Desktop/FYP_Long_Term
./tf_metal_env/bin/python3 scripts/run_csrle_stage2_synthetic_gru.py
```

**Artifacts:** `experiments/..._164307/stage2_synthetic_gru/`

Runtime: ~26 min (4 conditions × GRU train + 99-container eval with Prophet refit)

Does **not** retrain Condition A. Does **not** modify pre-GRU or Stage 1 artifacts.

---

## Stage 1 — Condition A reproduction

```bash
cd /Users/lehaan/Desktop/FYP_Long_Term
./tf_metal_env/bin/python3 scripts/run_csrle_stage1_control_reproduction.py
```

**Artifacts:** `experiments/..._164307/stage1_control_reproduction/`

Runtime: ~7 min (Prophet training + GRU + 99-container eval × 2)

---

## Pre-GRU validation (v2)

```bash
cd /Users/lehaan/Desktop/FYP_Long_Term
./tf_metal_env/bin/python3 scripts/run_csrle_pre_gru_validation.py
```

Runtime: ~68 s for 99 containers (Prophet retention dominates).

## Expected outputs (pre-GRU)

```
experiments/synthetic_residual_learnability_<timestamp>/
├── verification/frozen_integrity_checksums.json
├── config/synthetic_config_frozen.json          # only if B0+B1 generator pass
├── data/
│   ├── control/{train_syn,val_syn}.parquet
│   ├── b0_lc/{train_syn,val_syn}.parquet
│   ├── b1_snar/{train_syn,val_syn}.parquet
│   ├── c0_iid/{train_syn,val_syn}.parquet
│   ├── c1_iid/{train_syn,val_syn}.parquet
│   └── ground_truth/injection_manifest.parquet
└── synthetic_validation/
    ├── generator_gate_report.json
    ├── prophet_retention_report.json
    ├── prophet_retention_b0.csv
    ├── prophet_retention_b1.csv
    ├── control_prophet_only_mae.csv
    └── pre_gru_summary.json
```

## Authoritative artifacts by result class

| Result class | Authoritative file |
|--------------|-------------------|
| Gate pass/fail summary | `synthetic_validation/pre_gru_summary.json` |
| Generator metrics | `synthetic_validation/generator_gate_report.json` |
| Retention metrics | `synthetic_validation/prophet_retention_report.json` |
| Per-container retention | `synthetic_validation/prophet_retention_b{0,1}.csv` |
| Control Prophet MAE | `synthetic_validation/control_prophet_only_mae.csv` |
| Ground-truth injection | `data/ground_truth/injection_manifest.parquet` |
| Generator code | `utils/csrle/generators.py` |
| Gate thresholds | `utils/csrle/config.py` |

## Verification checks

1. **Identity:** max `identity_max_abs_err` ≈ 0 in retention CSVs
2. **κ:** `generator_gate_report.json` → kappa.pass == true for B0/B1
3. **C std match:** relative error ≤ 0.02 for C0 vs B0, C1 vs B1
4. **Frozen integrity:** compare checksums to recorded values if data unchanged
5. **Stage 2 outputs** under `stage2_synthetic_gru/` only; no writes to `models/` or frozen experiments

## Current run reference

**v2 experiment directory:** `experiments/synthetic_residual_learnability_2026-07-25_164307/`

**Outcome:** Pre-GRU v2 PASS; Stage 1 PASS; **Stage 2 COMPLETE**.

**Stage 2 summary:** `stage2_synthetic_gru/stage2_final_report.json`

**v1 audit run (preserved):** `experiments/synthetic_residual_learnability_2026-07-25_163013/`
