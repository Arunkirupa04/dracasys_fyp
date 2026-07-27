# GRU Pipeline

## Role

The GRU predicts the **scaled residual** time series — what Prophet leaves unexplained — for the next 96 steps.

## Model (frozen)

Loaded once from `artifacts/hybrid_v1/model.keras`.

```
Input:  (1, 96, 1)  — last 96 global z-scored residuals
Output: (96,)        — predicted future scaled residuals
```

## Residual normalisation

Always uses global training statistics from `residual_stats.pkl`:

```
residual_scaled = (residual - res_mean) / res_std
```

Never refit at inference.

## Prediction (`app/inference/gru_predictor.py`)

1. Take last 96 `residual_scaled` values from history
2. `model.predict(x)` → 96-step output
3. Truncate to `prediction_horizon_steps` if < 96
4. Inverse z-score: `residual = pred * res_std + res_mean`

## Combination

```
final_scaled = prophet_yhat_future + residual_level
final_cpu% = MinMax.inverse_transform(final_scaled)
```

## Training summary (informational)

| Property | Value |
|----------|-------|
| Containers | 435 known |
| Sequences | 183,012 |
| Optimizer | Adam |
| Loss | MSE |
| Params | ~405,000 |

Training code is **not** included in this service.

## Limitations

- Fixed 96-step input window
- Maximum 96-step output horizon
- No uncertainty quantification
- MSE-trained — may under-predict spikes

See [artifact_reference.md](artifact_reference.md).
