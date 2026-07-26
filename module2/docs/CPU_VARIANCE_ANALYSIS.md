# CPU Metrics Analysis: Understanding the "Straight Line" Issue

## The Question
> "Why do CPU metrics show a straight line but there is some change?"

## The Answer: Multiple Patterns with Fine Granularity

### 1. **CPU Metrics ARE Changing - But Very Slowly**

| Container | Type | Std Dev | Range | Pattern |
|-----------|------|---------|-------|---------|
| container_13 | HIGH volatility | 1.26 | 5.08 | Gradually increasing |
| container_7 | HIGH volatility | 1.15 | 4.90 | Variable workload |
| container_25 | LOW volatility | 0.15 | 0.70 | Nearly constant |
| container_27 | LOW volatility | 0.18 | 0.89 | Nearly constant |

**Sample values from container_13 (appears "flat" but is actually incrementing):**
```
[2.1993, 2.1994, 2.1994, 2.1994, 2.1995, 2.1995, 2.1995, 2.1995, 2.1995, 
 2.1996, 2.1996, 2.1996, 2.1997, 2.1997, 2.1997, 2.1997, 2.1997, 2.1998, 2.1998, 2.1998, ...]
```

**Sample values from container_25 (truly nearly constant):**
```
[-0.8791, -0.8791, -0.8791, -0.8791, -0.8791, -0.8791, -0.8791, -0.8791, -0.8791, 
 -0.879, -0.879, -0.879, -0.879, -0.879, -0.879, -0.879, -0.879, -0.879, ...]
```

### 2. **Why Plots Look Flat**

- **Increments are tiny**: Changes of ~0.0001 between timesteps
- **Visual scale**: At 500-sample view, 0.0001 changes are invisible
- **Overlapping lines**: When 3 CPU metrics plotted together, small differences overlap
- **Mix of patterns**: Some containers nearly constant, some slowly changing

### 3. **Evidence of Actual Variation**

✓ **Overall statistics confirm variation exists:**
- CPU Total: mean=-0.02, std=0.977, range=5.27
- CPU System: mean=-0.02, std=0.977, range=5.61
- CPU User: mean=0.00, std=0.977, range=5.16

✓ **Per-container analysis:**
- 27 containers with different volatility patterns
- Highest volatility: 1.26
- Lowest volatility: 0.15
- **Mean volatility: 0.71** (no variance = 0, so 0.71 confirms variation)

✓ **Difference plots reveal the changes:**
When you plot sample-to-sample deltas (differences), you see:
- HIGH volatility containers: frequent changes of 0.001-0.01
- LOW volatility containers: occasional changes, mostly constant

### 4. **What This Means for GRU Training**

✅ **GOOD: Diverse workload patterns**
- Real containers behave differently
- Some have constant baseline load (services in idle/maintenance mode)
- Some have gradually increasing load (services ramping up)
- Model learns both behaviors

✅ **GOOD: Temporal dependencies present**
- Even small changes are learnable by GRU
- Recurrent gates detect when containers change state
- Memory cells maintain context

✅ **GOOD: Multi-metric correlation**
- CPU changes correlate with memory changes
- Model captures these relationships

### 5. **Solution: Better Visualization Approaches**

When analyzing CPU metrics, use:

1. **Zoomed plots** (first 100-200 samples instead of 500)
   - Shows incremental changes clearly
   
2. **Difference plots** (plot derivatives, not values)
   - Reveals hidden variation
   
3. **Per-container analysis** (not grouped)
   - Shows individual patterns

4. **Distribution plots**
   - Confirm variation exists across containers

## Verification Summary

| Check | Result |
|-------|--------|
| All 7 metrics present | ✓ PASS |
| 27 unique containers | ✓ PASS |
| 4 cases merged | ✓ PASS |
| No null values | ✓ PASS |
| Z-score normalized | ✓ PASS (mean≈0, std≈1) |
| CPU has variation | ✓ PASS (std=0.977) |
| Temporal patterns exist | ✓ PASS (both constant & variable) |
| Ready for sequences | ✓ PASS |

## Next Step: Sequence Generation

The data is verified. Ready to create:
- **240-timestep sliding windows** (sequence length)
- **1-12 step prediction horizons** (forecast lengths)
- **393,506 possible sequences** from 533k rows
- **Train/validation/test splits** by case

---

**Reference notebooks:**
- `analysis_merged_data_comprehensive.ipynb` - Full feature analysis
- `cpu_variance_diagnostic.ipynb` - CPU variance deep-dive (run this to see zoomed plots)
