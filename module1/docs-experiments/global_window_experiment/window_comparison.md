# GGTCE Window Comparison

Three independent models: G96, G192, G288 (output horizon fixed at 96).

## Transitions

| Transition | Interpretation |
|------------|----------------|
| G96 → G192 | First doubling of temporal context |
| G192 → G288 | Second extension (+50%) |
| G96 → G288 | Full protocol span |

## Statistics

For each transition and metric:

- Per-container Δ (right − left)
- Mean/median Δ, Cohen's d
- Paired bootstrap 95% CI
- Wilcoxon signed-rank p-value
- Fraction of containers improved

## Sequence reduction trade-off

Documented in `sequence_analysis/sequence_reduction.csv`:

- Training sequence count decreases as input window grows
- Compare training time and mean Day-1 MAE across variants
