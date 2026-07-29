# Methodology

## Frozen components (unchanged from baseline Hybrid)

| Component | Setting |
|-----------|---------|
| Prophet | `daily_seasonality=True`, `weekly_seasonality=False`, per-container train fit |
| Residual | `cpu_scaled - prophet_pred`, global z-score on train |
| GRU | `GRU(256)→Dropout(0.2)→GRU(128)→Dropout(0.2)→GRU(64)→Dense(128,relu)→Dense(96)` |
| Optimizer | Adam (lr=0.001 default) |
| Loss | MSE |
| Epochs | 100 max, early stopping patience=10 |
| Batch size | 64 |
| Shuffle | False |
| Sequence split | Chronological 80/20 on sliding windows |
| Input window | 96 |
| Day-1 horizon | 96 |
| Cohort | 100 selected containers (~99 evaluable) |

## Variable component

GRU input channels only — cumulative ablation ladder V0→V4.

## Training

Each variant trained **independently** (fresh weights). No transfer learning between variants.

## Evaluation

Identical to Hybrid Day-1 protocol plus CSRLE extended metrics:

- CPU: MAE, RMSE, MAPE
- Peak subset: peak MAE/RMSE (train P90 thresholds)
- Residual: MAE, RMSE, Pearson r, R², std ratio, energy recovery

## Statistics

Paired per-container analysis for transitions V0→V1→V2→V3→V4:

- Mean/median delta
- Bootstrap 95% CI (seed 12345)
- Wilcoxon signed-rank
- Cohen's d effect size
