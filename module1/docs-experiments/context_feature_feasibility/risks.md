# Task 9 — Risks

Assessment of risks for a context-enriched Hybrid v2 experiment.

---

## 1. Feature redundancy

**Risk:** Many proposed features are deterministic functions of the 96-step residual vector (rolling mean, lags, diffs).

**Evidence:** Read-only correlation shows \|r\| ≈ 0.5–0.6 between `res_roll_mean`, `res_diff1`, and residual — but these are **algebraic** relationships, not incremental information.

**Mitigation:** Compact 5-channel design; exclude residual transforms except one volatility proxy.

**Severity:** Medium if ignored; Low with compact set.

---

## 2. Prophet already models the same information

**Risk:** Calendar features and long-lag structure duplicate Prophet's trend + daily seasonality.

**Evidence:**
- TMA: Prophet-equivalent residual lag-96 ACF **0.042** vs CPU **0.33**
- Residual–hour correlation ≈ **0.017**

**Mitigation:** Do not prioritize calendar features; use Prophet level (`yhat`) only as **regime context**, not as second seasonality model.

**Severity:** High for calendar features; Low for level/volatility features.

---

## 3. Information leakage

**Risk:** Using validation CPU, future residuals, or val-computed thresholds in features.

**Mitigation:**
- Train-only scalers and P90 thresholds (existing peak-aware protocol)
- Causal rolling windows (no centered windows)
- Input window strictly **past** relative to forecast origin

**Severity:** High if protocol violated; Low under existing pipeline rules.

---

## 4. Increased model complexity

**Risk:** 96×N inputs increase first-layer parameters and overfitting risk with weak underlying signal.

**Evidence:** RLLA — Ridge best Pearson r ≈ **0.059** on real Prophet residuals; structure is weak.

**Mitigation:** Incremental ablation (96×2 → 96×5); early stopping unchanged; same cohort size.

**Severity:** Medium.

---

## 5. Overfitting

**Risk:** Context channels fit container-specific quirks in 6-day train window.

**Evidence:** 100 containers, ~600 train steps each; heavy-tailed residuals (kurtosis ≈ 58).

**Mitigation:** Prefer low-dimensional context; container static features (cpu_std) are low-risk; monitor val MAE **and** residual Pearson r / std ratio.

**Severity:** Medium.

---

## 6. Poor generalization / unseen containers

**Risk:** Peak thresholds and cpu_std are container-specific; new containers lack history.

**Mitigation:** Document deployment requirement: cold-start needs minimum train window for Prophet + stats; fallback to cohort defaults.

**Severity:** Medium for production; Low for thesis cohort evaluation.

---

## 7. Level–residual collinearity

**Risk:** `cpu_scaled = prophet_yhat + residual` — adding yhat alongside residual is partially collinear.

**Mitigation:** Accept explicit level channel as **inductive bias** (regime-dependent correction); ablation v2a tests incremental value; do not also add raw cpu_scaled.

**Severity:** Medium — empirical ablation required.

---

## 8. False hope from correlation analysis

**Risk:** High residual–cpu correlation (r ≈ 0.77) suggests easy gains, but Hybrid **already** adds residual to Prophet — improving residual prediction is the hard part (RPA, LFHE, RLLA).

**Mitigation:** Frame v2 as **hypothesis test**, not expected breakthrough; success = measurable r/std ratio improvement with MAE guardrail.

**Severity:** High (expectation management).

---

## 9. Objective function dominates (LFHE)

**Risk:** Even with perfect context, MSE may still collapse variance.

**Evidence:** LFHE — DA-MSE fixed std ratio but **worsened** Pearson r; CPU MAE unchanged.

**Mitigation:** Future v2 experiments should pair feature ablation with loss ablation (out of scope here).

**Severity:** High for correlation gains; documented.

---

## 10. Multivariate false path

**Risk:** Time spent on mem/net/disk features despite 98%+ missingness.

**Mitigation:** This study rules them out categorically.

**Severity:** N/A if recommendation followed.

---

## Risk summary table

| Risk | Likelihood | Impact | Priority |
|------|------------|--------|----------|
| Weak signal (RLLA) | High | High | Monitor metrics |
| MSE variance collapse (LFHE) | High | High | Pair with loss study |
| Prophet redundancy (calendar) | High | Low if excluded | Design OK |
| Overfitting | Medium | Medium | Ablation ladder |
| Leakage | Low (if protocol) | High | Verify splits |
| Feature redundancy | Medium | Medium | Compact set |
