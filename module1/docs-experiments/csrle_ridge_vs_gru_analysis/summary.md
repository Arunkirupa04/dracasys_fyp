# Summary — Thesis-Ready Answers

Source: `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/results/evidence_synthesis.json`

---

## 1. Why did Ridge outperform GRU?

Ridge achieved higher residual correlation (r ≈ **0.158** vs **0.084**) because B0 Prophet residuals are **strongly linearly predictable** (AR(1) R² ≈ 0.20, |ACF|₁₋₁₀ ≈ 0.26), and Ridge’s **96→96 linear map** exploits lag structure directly. The GRU, trained with MSE, produced **variance-collapsed** predictions (std ratio ≈ **0.07** vs Ridge **0.22**), yielding similar MAE (~0.094) but poor shape tracking.

---

## 2. Is B0 actually nonlinear after Prophet?

The **generator** is nonlinear (B0-LC), but **Prophet residuals on validation** behave **predominantly like a short-memory linear process**: AR(2) R² ≈ 0.21 vs poly(2) trend R² ≈ 0.004. Nonlinearity in injection does not fully survive as nonlinear residual structure at the GRU target.

---

## 3. Does the GRU collapse toward low-variance predictions?

**Yes.** Cohort mean predicted/actual std ratio ≈ **0.073** for GRU vs **0.220** for Ridge. At horizon 4, GRU std ratio ≈ 0.39 vs Ridge ≈ 0.99.

---

## 4. Does MSE appear responsible?

**Partially.** Identical MAE (~0.094) with divergent correlation shows MSE does not reward variance matching. MSE-optimal GRU outputs resemble **small constant-like corrections** — adequate for mean error, inadequate for temporal correlation.

---

## 5. Architectural limitation or data limitation?

**Combined:** (a) **Target structure** after Prophet is largely linear-lag predictable (data/projection limitation of “nonlinear B0” at residual level); (b) **GRU + MSE** systematically under-predicts residual variance (training/architecture limitation). Ridge is matched to (a); GRU fails on (b) even when (a) holds.

---

## 6. How does this explain weak Alibaba residual learning?

Condition A shows the same **variance collapse** pattern (std ratio ≈ 0.05 in Stage 1). Real residuals also have short-lag structure (prior diagnostic ACF ≈ 0.20). The GRU **does not exploit linearly predictable structure** effectively; Ridge on B0 demonstrates that a **simple linear learner captures more correlation** when structure is present — implying the Hybrid bottleneck is **learning/objective behaviour**, not absence of all structure in real residuals.

---

## 7. What to state in the thesis discussion?

> On the CSRLE B0 positive control, Ridge outperformed the fixed Hybrid GRU in residual correlation because Prophet validation residuals retained **strong linear lag dependence** despite nonlinear injection, while the GRU under MSE training produced **severely variance-deflated** corrections. This separates (i) **existence of predictable structure** in Prophet residuals from (ii) **capacity of the fixed GRU training pipeline to exploit it**. The result bounds Hybrid claims: linearly structured synthetic residuals favour Ridge; real Alibaba residuals show analogous GRU variance collapse without comparable correlation gains.

**Do not** claim Ridge should replace GRU in production. **Do not** reopen CSRLE for retuning.

---

## Key figures for thesis

| Figure | Path |
|--------|------|
| ACF/PACF B0 pooled | `plots/analysis1_b0_acf_pacf.pdf` |
| Variance recovery | `plots/analysis4_variance_recovery_scatter.pdf` |
| Trajectories | `plots/analysis5_trajectory_median_delta_r_c_11461.pdf` |
| Horizon comparison | `plots/analysis7_horizon_comparison.pdf` |
| Container scatter | `plots/analysis8_container_scatter.pdf` |

Interactive reproduction: `notebooks/csrle_ridge_vs_gru_analysis.ipynb`
