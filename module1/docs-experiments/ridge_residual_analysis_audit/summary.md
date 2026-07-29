# RRAA Summary

## Audit

**Read-only audit of RRA v1.0** — scaling bug found; corrected metrics computed; AR linearity test performed.

## One-sentence takeaway

RRA v1.0's ACF increase was a **scaling artifact** (raw − z-scored subtraction); after correction, Ridge barely changes temporal diagnostics, remaining structure is **AR(1)-linear**, and **Ridge→GRU is not justified**.

## Explicit decisions

| # | Question | Answer |
|---|----------|--------|
| 1 | Ridge failed to remove dependence vs failed as predictor? | **Predictor failure** (r≈0.11); no meaningful structure removal after fix |
| 2 | Remaining dependence linear or nonlinear? | **Predominantly linear** (AR(1) r≈0.38) |
| 3 | Ridge→GRU vs more linear modelling? | **More linear modelling** — Ridge→GRU **not justified** |

## Corrected vs RRA metrics

| Metric | RRA (buggy) | Corrected |
|--------|-------------|-----------|
| Remaining \|ACF\| 1–10 | 0.266 | 0.201 |
| Ljung reject | 96.0% | 73.7% |

## Artifacts

- Audit: `experiments/ridge_residual_analysis_audit_2026-07-26_162500/`
- Notebook: `notebooks/ridge_residual_analysis_audit.ipynb`
- RRA (unchanged): `experiments/ridge_residual_analysis_2026-07-26_161500/`

## Thesis use

> A read-only audit identified a scaling inconsistency in RRA v1.0 (subtracting z-scored Ridge predictions from raw Prophet residuals). Corrected diagnostics showed Ridge does not materially alter autocorrelation or Ljung–Box outcomes; walk-forward AR(1) models explain remaining structure similarly to Prophet residuals. These findings do not support proceeding to a Ridge→GRU architecture experiment without first improving the linear residual stage.
