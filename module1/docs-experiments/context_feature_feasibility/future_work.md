# Future Work

Items for **after** this feasibility study. None are implemented here.

---

## 1. Hybrid v2 incremental ablation experiment

- Implement 96×2, 96×3, 96×5 variants per [hybrid_v2_design.md](hybrid_v2_design.md)
- Reuse frozen Prophet protocol from baseline
- Compare against `experiments/baseline_reference_2026-07-14` Hybrid GRU
- Log hyperparameters per experiments skill

---

## 2. Hybrid v2 + loss function

LFHE showed dispersion fixes alone insufficient. If v2 improves r modestly but std ratio still low:

- Combine best v2 feature set with DA-MSE (λ grid)
- Test whether context + objective jointly address CSRLE–real-data gap

---

## 3. Prophet component ablation

- Feed `prophet_daily` vs `prophet_yhat` vs none
- Quantify whether GRU needs full level or only deviation from seasonal curve

---

## 4. Unseen container generalisation

- Train on subset; evaluate context features with **cohort-default** cpu_std and P90 fallback
- Tests deployment realism

---

## 5. Global GRU long-window study (separate track)

TMA supports 192/288/384 windows for **raw CPU** Global GRU — orthogonal to Hybrid v2.

---

## 6. Alternative residual learners

If v2 fails:

- Ridge/stacked linear on same 96×5 features (RLLA extension)
- Lightweight TCN/1D-CNN with context
- Compare whether failure is GRU-specific or signal-limited

---

## 7. Extended Alibaba multivariate (only if data re-ingested)

If raw trace re-processed with aligned mem/net at 15-min resolution:

- Re-run this feasibility matrix
- Until then, multivariate context is **out of scope**

---

## 8. Documentation updates

After v2 experiment:

- Add `docs/hybrid_v2/` with results
- Update thesis narrative chain: RPA → CSRLE → LFHE → TMA → RLLA → **Context Feasibility → v2 Results**
