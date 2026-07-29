# LFHE Verification

## Integrity checks

| Check | Status |
|-------|--------|
| CSRLE / Stage 2 artifacts modified | **No** |
| Hybrid / Global GRU artifacts modified | **No** |
| Only loss function changed (MSE vs DA-MSE) | **Yes** |
| Architecture identical | **Yes** |
| Prophet pipeline identical | **Yes** |
| CSRLE B0 data identical (hash in config) | **Yes** |
| λ frozen at 1.0 | **Yes** |
| Evaluation protocol = CSRLE Stage 2 | **Yes** |

## Reproduction gate

**Method:** Evaluate read-only Stage 2 B0 `hybrid_gru.keras` through LFHE pipeline.

| Metric | Max abs diff vs Stage 2 CSV |
|--------|----------------------------|
| Pearson r | ~1e−16 |
| Std ratio | ~1e−16 |
| MAE | ~1e−16 |

**Result:** PASS — see `verification/mse_reproduction_gate.json`.

## Loss unit tests

DA-MSE verified:
- Equals MSE when σ_ŷ ≥ σ_y
- Dispersion penalty when under-dispersed
- No penalty when over-dispersed
- Finite loss and gradients

## Notebook

`notebooks/lfhe_visualization.ipynb` — section-by-section, all artifacts visible.

Re-run: `python scripts/build_lfhe_visualization_notebook.py` (update timestamp in cell 1 if needed).
