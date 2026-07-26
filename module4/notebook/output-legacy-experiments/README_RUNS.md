# Frozen / Legacy Run Notebooks

**Last updated:** 2026-07-11

| File | Status | Use in thesis? |
|------|--------|----------------|
| `mdc_model_vNext_lat_output.ipynb` | **REFERENCE** (core detector) | Yes — headline ROC/F1/multi-seed/HPO |
| `mdc_model_vNext_output.ipynb` | Older/alternate model run | Prefer `*_lat_output*` for 0.7402 lineage |
| `mdc_drift_aware_output.ipynb` | **LEGACY v1** | Core AUC/IF/fine-tune OK; **DO NOT cite adaptive FPR↓0.92** |
| `mdc_drift_aware_output_v2.ipynb` | **TARGET freeze (Task 4)** | Yes — after Colab re-run with Tasks 1–3 gates PASS |
| `mdc_preprocess_vNext_output.ipynb` | Preprocess reference | Re-run after timestamp export if needed |

## Why v1 drift output is legacy

The v1 drift-aware run recomputed the fixed threshold with a double-invert score convention:

- Loaded/recomputed thr ≈ **−0.0923** (wrong)
- Offline thr should be ≈ **−0.1434**
- Fixed stream FPR ≈ **97%** → adaptive “FPR reduction ≈ 0.92” is an **artifact**

## How to produce v2 freeze

1. Re-run `mdc_preprocess_vNext.ipynb` (exports `ts_*` into `windows_vnext.npz`)
2. Re-run `mdc_preprocess_vNext_mc.ipynb` (exports `ts_*` + multiclass)
3. Run updated `mdc_drift_aware.ipynb` (+ `mdc_eval_utils.py`)
4. Confirm `THRESHOLD INTEGRITY: PASS` and `freeze_ready=true`
5. Download executed notebook → save here as `mdc_drift_aware_output_v2.ipynb`
6. Keep this README; do not delete the legacy v1 file (audit trail)

## Locked headlines (see `docs/claim_evidence_matrix.md`)

- Abstract: **0.7529 ± 0.023**
- Methods default: **0.7402**
- Best-case HPO: **0.8138**
