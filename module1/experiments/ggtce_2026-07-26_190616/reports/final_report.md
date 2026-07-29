# GGTCE — Final Report

## Global GRU Temporal Context Experiment

### Final Answers

**1_does_longer_window_improve:** No — G96 vs G96 MAE 1.924→1.924 (+0.00%)

**2_best_input_window:** G96 (G96 — Input 96 (baseline reproduction)) — mean Day-1 MAE=1.924

**3_statistically_significant:** Yes (G96→G288 bootstrap/Wilcoxon on MAE; mean Δ=0.1751)

**4_practically_meaningful:** No (|MAE change|=0.00%, threshold 1%)

**5_high_memory_benefit_more:** No / inconclusive — memory–ΔMAE Pearson r=-0.140; low-memory gain negligible: False

**6_diminishing_returns:** Evidence of diminishing returns after 192 (|Δ| G96→G192=0.3307, G192→G288=0.1556)

**7_context_vs_sequence_tradeoff:** G96 sequences=41650, g96=41650 (0.0% reduction). Accuracy did not improve despite fewer samples.

**8_replace_global_baseline:** Keep G96 baseline

**9_production_recommendation:** Keep frozen G96 Global GRU — longer windows did not justify complexity/cost

**10_relation_to_tma:** TMA showed 40–73% squared-ACF capture at 96–288 lags; GGTCE tests whether exploiting that memory improves forecasts. High-memory subgroup did not consistently gain more than low-memory workloads.

**g96_replication_passed:** True

## Cohort Comparison

| variant_id | n | mean_day1_mae | mean_day1_rmse | mean_day1_mape | mean_forecast_mae_scaled | mean_forecast_rmse_scaled | mean_forecast_pearson_r | mean_forecast_r2 | mean_forecast_std_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g96 | 99 | 1.9242406355693225 | 2.610536144884579 | 116.5025742728598 | 0.10324519119088012 | 0.1508849760921221 | 0.415500244816224 | 0.06264650359822685 | 0.517161189876937 |
| g192 | 99 | 2.2549448980192537 | 2.946503231653163 | 121.62159687857879 | 0.11555823671169575 | 0.1635510016802713 | 0.27776181250853965 | -0.12164389693385527 | 0.49839210684633684 |
| g288 | 99 | 2.0993443744119133 | 2.8426729629768475 | 114.81717159915074 | 0.10984430621646665 | 0.15945417338653814 | 0.3773027670159756 | -0.0036570079500886136 | 0.5367331016805814 |

## Sequence Reduction

| variant_id | input_window | n_sequences | n_train_sequences | training_seconds | best_epoch | epochs_run | mean_per_container_sequences | mean_day1_mae | mean_forecast_pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g96 | 96 | 41650 | 33320 | 177.45836287501152 | 25 | 35 | 421.7070707070707 | 1.9242406355693225 | 0.415500244816224 |
| g192 | 192 | 32146 | 25716 | 107.39743274997454 | 7 | 17 | 325.7070707070707 | 2.2549448980192537 | 0.27776181250853965 |
| g288 | 288 | 22642 | 18113 | 284.7665681250219 | 36 | 46 | 229.7070707070707 | 2.0993443744119133 | 0.3773027670159756 |
