# RRA Discussion

## Interpretation

### Ridge does not whiten Prophet residuals

Under the frozen 96→96 Ridge protocol (α=1.0), the **remaining** series `e(t) = r_prophet(t) − r_ridge(t)` exhibits **stronger** short-lag autocorrelation and **more** Ljung–Box rejections than the original Prophet validation residuals.

This is consistent with Ridge achieving only modest linear alignment (mean r ≈ 0.13): subtracting a weakly correlated linear forecast **does not** remove temporal structure — it recombines signal and error into a higher-variance remainder (mean std ratio ≈ 1.82).

### Negative "ACF reduction" is not a contradiction

A positive scientific outcome: the experiment shows that **linear residual modeling alone is not equivalent to structure removal**. The diagnostic answers the research question directly:

- Ridge **does not** remove most temporal dependence (Q1: **No**).
- Remaining residuals are **not** closer to white noise (Q2: **No**).
- **96%** of containers still reject white noise (Q3: **Yes**).
- Remaining |ACF| is statistically meaningful (Q4: **Yes**).

### Implications for GRU / Ridge→GRU

| Finding | Implication |
|---------|-------------|
| Structure persists after Ridge | A nonlinear stage **may** still add value — but not because Ridge succeeded |
| Ridge r ≈ 0.13 on B0-LC | Matches CSRLE Stage 2 observation (Ridge > GRU on correlation) |
| Remaining variance inflated | Future Ridge→GRU must use **residual correction** design, not naive subtraction of a poor linear fit |
| LFHE GRU failed on shape | RRA confirms linear stage also fails to simplify the learning target |

### What this experiment does **not** claim

- It does **not** prove a Ridge→GRU stack will outperform Hybrid Prophet+GRU.
- It does **not** train or evaluate any GRU.
- It does **not** recommend implementing Ridge→GRU without a new frozen architecture protocol.

### Recommended next step

A **prospective Ridge→GRU architecture experiment** is **scientifically justified** because structure persists. That experiment must define:

1. How Ridge predictions are combined (fixed linear + GRU correction vs stacked inputs).
2. Success criteria on **CPU metrics** and **remaining residual whiteness**, not only Ridge r.
3. Comparison against Ridge-only and Hybrid Prophet+GRU baselines on identical B0-LC splits.

## Relation to prior work

| Experiment | Connection |
|------------|------------|
| Residual Pattern Analysis | Prophet residuals show structure; RRA asks if Ridge removes it — **No** |
| CSRLE Stage 2 | Ridge baseline r > GRU on B0; RRA explains linear stage leaves structured remainder |
| LFHE | GRU loss affects dispersion not shape; RRA shows linear stage also leaves structured remainder |
