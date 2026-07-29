# GGTCE Evaluation

Evaluation reuses the frozen Global GRU temporal-holdout protocol:

- Per-container MinMax scaling (frozen `data/scalers.pkl`)
- Last `input_window` train steps as model input
- Day-1 horizon: first 96 validation steps
- Metrics: Day-1 MAE/RMSE/MAPE (real CPU %), scaled forecast MAE/RMSE, Pearson r, R², std ratio

Extended metrics are computed in `utils/ggtce/evaluation.py`.

Window comparison uses paired bootstrap (10k resamples) and Wilcoxon signed-rank tests per container.

Results are written to `experiments/ggtce_*/window_comparison/` after each run.
