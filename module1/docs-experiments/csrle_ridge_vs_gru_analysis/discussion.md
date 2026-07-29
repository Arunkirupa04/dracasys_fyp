# Discussion

## Central explanation (evidence-backed)

Ridge outperformed the frozen Hybrid GRU on B0 **not because the GRU failed to model a uniquely nonlinear residual**, but because:

1. **Prophet residuals on B0 are strongly linearly predictable** (AR(1) R² ≈ 0.20, mean |ACF|₁₋₁₀ ≈ 0.26), while polynomial nonlinear trend adds ~0% beyond AR(2).

2. The **96→96 Ridge map** directly exploits lag-linear structure in the input window — the same structure a linear autoregressive model captures.

3. The **GRU under MSE training collapses predicted residual variance** (cohort mean std ratio ≈ 0.07 vs actual), producing near-flat corrections that can achieve **similar MAE to zero/Ridge** but **cannot match Ridge correlation**.

4. **Spectral preservation is not the differentiator** at cohort level (mean spectral correlation ≈ 0.318 for both). Ridge wins on **amplitude and lag-linear tracking**, not dominant-frequency recovery.

5. B0 injection is nonlinear in **state space**, but the **observable Prophet residual** on Day-1 behaves like a **short-memory linear process** — sufficient for Ridge, insufficiently exploited by variance-shrunk GRU outputs.

---

## Is B0 “nonlinear after Prophet”?

**Partially.** The injected B0-LC dynamics are nonlinear, but post-Prophet pooled residuals show:

- High linear AR predictability  
- Negligible incremental poly(2) trend R²  
- Entropy comparable to A and C0  

So the **residual target seen by the GRU is predominantly linear in lag structure**, even though the generative process is nonlinear.

---

## Does GRU collapse toward low-variance predictions?

**Yes.** Cohort mean predicted/actual std ratio ≈ **0.073** (GRU) vs **0.220** (Ridge). At horizon 4, GRU std ratio ≈ 0.39 vs Ridge ≈ **0.99**. This matches residual-pattern analysis findings (near-flat GRU corrections on real data).

---

## Is MSE responsible?

**Contributing factor.** GRU and Ridge achieve **nearly identical MAE** (~0.094) while correlation differs sharply. MSE-optimal predictions under variance shrinkage can minimize error norm without matching temporal shape — explaining negative R² for both, but worse for Ridge at cohort level despite higher r.

---

## Architectural vs data limitation?

**Both, at different layers:**

- **Data/target:** B0 Prophet residuals are linearly structured → Ridge aligned with target.  
- **Training/objective:** MSE + GRU architecture yields variance collapse → fails to exploit even linearly predictable structure as well as Ridge.  
- **Not** evidence that nonlinear capacity is required for B0 **after Prophet** — linear model suffices on observed residuals.

---

## Link to weak Alibaba (Condition A) behaviour

Same mechanism plausibly applies: real Prophet residuals show short-lag dependence (residual-pattern analysis ACF ≈ 0.20), but frozen GRU outputs **~5% of actual residual std** on A. Ridge on B0 shows **partial** variance recovery (22%) and higher r — demonstrating the **pipeline can do better on linearly structured targets**, but GRU training/objective suppresses variance on both real and synthetic data.

---

## Interpretation boundary

This analysis **explains the Ridge vs GRU gap on B0**; it does **not** recommend retraining, architecture changes, or new forecasting experiments. CSRLE Stage 2 conclusions remain frozen.
