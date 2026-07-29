# Hypotheses

## Primary hypotheses

### H0 (null)

Increasing the Global GRU input window **does not** improve Day-1 CPU forecasting performance beyond G96.

Operationalised: cohort mean MAE improvement ≤ practical threshold **and** bootstrap 95% CI for Δ MAE includes zero.

### H1 (alternative)

Longer input windows **improve** forecasting because they expose additional temporal dependence present in raw CPU but invisible to G96.

Operationalised: cohort mean Δ MAE < 0 with bootstrap CI excluding zero **and** effect size |d| ≥ 0.2.

---

## Secondary hypotheses

### H2 — Differential benefit (subgroup)

Improvement is **concentrated** in high long-memory containers (TMA top tertile).

- H2a: Δ MAE negatively correlated with TMA `pct_capture_within_96` (more uncaptured memory → more to gain)
- H2b: High-memory subgroup shows significant improvement; low-memory subgroup does not

### H3 — Diminishing returns

G192 captures most achievable gain; G288 provides **marginal incremental** benefit.

- Supported if: |Δ MAE(G96→G192)| > |Δ MAE(G192→G288)| and G192→G288 CI includes zero

### H4 — Overfitting

Longer windows **hurt** generalisation despite lower training loss.

- Supported if: G288 val_loss improves but Day-1 MAE worsens vs G96, or G288 MAE > G192

---

## Alternative outcomes (pre-registered interpretations)

| Outcome pattern | Interpretation |
|-----------------|----------------|
| G192 best, G288 ≈ G192 | **Optimal window ≈ 2 days** — recommend G192 |
| Monotonic improvement G96 < G192 < G288 | Long memory fully exploitable — recommend G288 if significant |
| No variant beats G96 | TMA memory is **not linearly learnable** by GRU under MSE |
| High-memory only gain | Supports TMA; use adaptive window or container-specific context in future work |
| All variants worse | Window increase adds noise / overfitting — keep G96 |
| G96≈G192, G288 worse | Sweet spot at 192; 288 exceeds useful span |

---

## Connection to TMA

| TMA finding | Hypothesis link |
|-------------|-----------------|
| 40% capture @ 96 lags | Room for H1 if GRU exploits lags 97–192 |
| Lag-192 ACF ≈ 0.25 | G192 directly aligned with 2-day structure |
| 73% capture @ 288 | G288 tests upper bound before data-length limit |
| Hybrid residual short memory | Does **not** predict Global outcome — separate test |

TMA justifies **attempting** GGTCE; GGTCE **tests** whether statistical memory translates to forecast gain.
