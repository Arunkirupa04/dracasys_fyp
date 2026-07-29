# Window Analysis

Estimates computed from frozen `data/train_df.parquet`, 99 evaluable containers, `forecast_horizon=96`, using `create_longterm_sequences` logic.

## Sequence counts

| Variant | Total sequences | Train (80%) | Val (20%) | Mean / container | Min / container | Zero-seq containers |
|---------|-----------------|-------------|-----------|------------------|-----------------|---------------------|
| **G96** | 41,749 | 33,399 | 8,350 | 421.7 | 409 | **0** |
| **G192** | 32,245 | 25,796 | 6,449 | 325.7 | 313 | **0** |
| **G288** | 22,741 | 18,192 | 4,549 | 229.7 | 217 | **0** |
| G384 | 13,237 | 10,589 | 2,648 | 133.7 | 121 | **0** |

**Formula:** `sequences_c = n_train_c - input_window - 96 + 1`  
**Train steps/container:** min 600, max 614, mean 612.7

### Reduction vs G96

| Variant | Sequence reduction | Remaining fraction |
|---------|-------------------|-------------------|
| G192 | −23% | 77% |
| G288 | −46% | 54% |
| G384 | −68% | 32% |

**Verdict:** All variants retain **>100 sequences per container** and **>4,500 total sequences** (G288). G384 is viable but **materially smaller** early-stopping validation set (2,648 vs 8,350).

---

## TMA squared-ACF capture alignment

Cohort mean cumulative squared-ACF capture on **original CPU** (TMA run `2026-07-26_170052`):

| Window (lags) | Mean capture | Increment vs previous |
|---------------|--------------|------------------------|
| 96 | **40.0%** | — |
| 192 | **59.9%** | +19.9 pp |
| 288 | **73.4%** | +13.5 pp |
| 672 | 100.0% | (not usable as input — exceeds train length) |

Each proposed window exposes a **meaningful additional fraction** of statistically measured memory — scientific motivation is strongest for **G192** and still present for **G288**.

---

## Memory / compute estimates

Assumptions: float32, 3 features, batch=256, 50 epochs max, Apple M1-class GPU.

| Variant | Input KB / sequence | Relative first-GRU input params | Est. train time vs G96 | Peak RAM (materialized X) |
|---------|---------------------|-----------------------------------|------------------------|---------------------------|
| G96 | ~1.1 | 1.0× | 1.0× | ~0.05 GB |
| G192 | ~2.2 | 2.0× | ~1.3–1.6× | ~0.07 GB |
| G288 | ~3.4 | 3.0× | ~1.6–2.0× | ~0.08 GB |
| G384 | ~4.5 | 4.0× | ~2.0–2.5× | ~0.06 GB |

Training builds sequences in memory like baseline — **practical on 16 GB RAM**. Bottleneck is **GRU sequence length** (longer unroll per forward pass), not dataset storage.

---

## Per-variant assessment

### G96 (control)

| Aspect | Assessment |
|--------|------------|
| Sequences | ✅ Maximum (41,749) |
| TMA capture | 40% — baseline |
| Benefit | Control reference |
| Risk | None — frozen baseline |
| Justification | Required control |

### G192

| Aspect | Assessment |
|--------|------------|
| Sequences | ✅ 32,245 (−23%) — adequate |
| TMA capture | +20 pp memory exposure |
| Expected benefit | **Moderate** — captures 2-day periodicity (lag-192 ACF ≈ 0.25) |
| Risk | Low–medium overfitting |
| Justification | **Strong — primary experimental arm** |

### G288

| Aspect | Assessment |
|--------|------------|
| Sequences | ✅ 22,741 (−46%) — adequate |
| TMA capture | +13.5 pp over G192 (73% total) |
| Expected benefit | **Incremental** — diminishing returns region |
| Risk | Medium overfitting; smaller val split |
| Justification | **Supported — tests plateau hypothesis** |

### G384 (optional)

| Aspect | Assessment |
|--------|------------|
| Sequences | ⚠️ 13,237 (−68%); min 121/container |
| TMA capture | Theoretical gain <15 pp over G288 |
| Expected benefit | **Low marginal** — train span ≈ 4 days; window eats 63% of history |
| Risk | **High** — sequence poverty, overfitting, unstable early stopping |
| Justification | **Exclude from primary protocol** — sensitivity only if G288 shows clear gains |

---

## Task 4 — Does 288 exceed useful temporal memory?

**No — but it approaches the practical ceiling for this dataset.**

Reasons:

1. **Statistical memory:** 288 lags still leave ~26% of squared-ACF energy beyond the window (on the TMA timeline). Useful dependence **still exists** beyond 288 in the diagnostic sense.

2. **Data length constraint:** Train series ≈ 614 steps. G288 uses 288 + 96 = 384 steps per sample → only **~230 steps of "new" history** relative to G96 beyond the shared tail. The model cannot access memory longer than ~4 days regardless of window.

3. **Diminishing returns:** Increment 192→288 (+13.5 pp capture) < increment 96→192 (+19.9 pp). **G288 is the last primary arm with reasonable power** before sequence count collapses.

4. **TMA lag structure:** Multi-day echoes persist to lag 480, but **forecast utility** of lags 289–480 is untested and likely smaller than 1–288.

**Conclusion:** G288 is **justified** as the upper primary window; it does not exceed *available* memory in the dataset, but likely enters **diminishing statistical returns** relative to G192.
