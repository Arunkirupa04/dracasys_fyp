# Hybrid Peak Error Attribution — Discussion Notes

**Label:** Post-hoc diagnostic / interpretation support — NOT a new experiment.

---

## Analysis D — Global architecture comparison (interpretation only)

No Prophet comparison was computed for Global GRU (Prophet is not part of that pipeline).

### Pipeline contrast (locked experimental designs)

**Hybrid Prophet + GRU**

```
CPU (train/val)
    ↓
Prophet  →  day1 prophet forecast (real CPU %)
    ↓
Residual = actual − prophet (scaled)
    ↓
GRU predicts residual correction only
    ↓
Final forecast = prophet + GRU residual
```

**Global GRU**

```
CPU (train/val)
    ↓
GRU predicts cpu_scaled directly
    ↓
Final forecast = GRU output (inverse scaled)
```

### Why identical Peak-Aware methodology produced different outcomes

Under the locked configuration (P90, λ = 5, timestep-weighted MSE):

| Aspect | Hybrid track | Global track |
|--------|--------------|--------------|
| Peak-aware weight applied to | GRU **residual** targets only | **Full CPU** trajectory targets |
| Prophet peak fit | Unchanged; defines much of peak-level forecast | N/A |
| Locked peak-aware Hybrid Δpeak MAE | +0.024 (no benefit) | — |
| Locked peak-aware Global Δpeak MAE | — | −0.777 (substantial benefit vs Global baseline) |

**Interpretation supported by evidence (not proven causation):**

1. On Hybrid, Peak-Aware learning reweights errors on the **residual pathway only**. The post-hoc attribution diagnostic indicates that **Prophet Peak MAE is close to Hybrid Peak MAE** and that **residual corrections at peak timesteps are relatively small**. Therefore, reweighting GRU residual training has **limited leverage** over final peak CPU forecasts.

2. On Global, the GRU models the **entire signal**. Peak-aware weighting directly reshapes the primary forecasting objective. The locked Global experiment shows a **large peak-subset improvement** (−0.777 peak MAE) accompanied by **non-peak and overall degradation**. This pattern is **consistent with** the GRU having direct control over peak timesteps, unlike the Hybrid decomposition.

3. **Architecture dependence** follows from **where** peak-aware weighting acts in the pipeline, not from different λ or peak definitions (those were held constant across tracks).

### Wording guidance for thesis

- Use: *"The evidence suggests…"*, *"The results are consistent with…"*, *"The findings support the interpretation…"*
- Avoid: *"Prophet causes Peak-Aware failure"* or *"Proves GRU cannot learn peaks"*

### Relationship to locked Peak-Aware results

This diagnostic **does not replace** the locked paired comparisons. It provides **mechanistic context** for why Hybrid Peak-Aware showed negligible peak benefit while Global Peak-Aware showed large within-architecture peak gains.
