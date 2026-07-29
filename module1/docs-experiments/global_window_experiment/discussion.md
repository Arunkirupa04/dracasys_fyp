# GGTCE Discussion

**Run:** `experiments/ggtce_2026-07-26_190616`

## Does longer temporal context help Global GRU?

No. Under frozen architecture and training protocol, the 96-step input window is optimal for this Alibaba trace cohort. Both extensions tested (192, 288) produced worse Day-1 CPU forecasts.

The non-monotonic pattern (G192 worst, G288 intermediate) suggests a **sequence-count / overfitting trade-off**: G192 loses 23% of training sequences vs G96 while the GRU capacity is unchanged, and early stopping at epoch 7 indicates unstable validation. G288 trains longer (46 epochs) and recovers some Pearson correlation, but never beats G96 on MAE.

## Relation to Temporal Memory Analysis

TMA established that 40–73% of squared ACF energy lies beyond 96 lags for many containers — i.e., long-range structure exists in the data. GGTCE shows that **existence of memory ≠ exploitable memory** for a fixed-capacity Global GRU trained with MSE on direct CPU forecasts.

The co-primary memory subgroup analysis found no significant correlation between memory score and window benefit. High-memory containers did not systematically improve with G288; if anything, they degraded slightly more on average.

## Sequence reduction vs accuracy

| Variant | Sequences | Δ vs G96 | Mean MAE |
|---------|-----------|----------|----------|
| G96 | 41,650 | — | 1.924 |
| G192 | 32,146 | −23% | 2.255 |
| G288 | 22,642 | −46% | 2.099 |

Additional history does not compensate for fewer sliding-window training samples under the current protocol.

## Recommendation

**Retain G96** as the frozen Global GRU baseline. Do not replace with G192 or G288 for production forecasting.

## Thesis chapter implications

GGTCE closes the loop opened by TMA: temporal memory is measurable but not automatically learnable by window extension alone. Future work may require architecture changes (attention, dilated convolutions) or multi-scale inputs rather than naive history lengthening.
