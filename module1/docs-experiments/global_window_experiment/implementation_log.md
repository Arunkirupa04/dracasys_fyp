# GGTCE Implementation Log

| Date | Event |
|------|-------|
| 2026-07-26 | Protocol approved; `utils/ggtce/` module implemented |
| 2026-07-26 | `scripts/run_ggtce_experiment.py` — full pipeline |
| 2026-07-26 | `scripts/finish_ggtce_experiment.py` — resume/post-process |
| 2026-07-26 | `scripts/build_global_window_experiment_notebook.py` |
| 2026-07-26 | **Authoritative run:** `experiments/ggtce_2026-07-26_190616` |

## Bug fixes during run

- `summarize_variant_cohort`: duplicate column names in residual block
- `plot_sequence_count_tradeoff`: duplicate merge columns
- `write_final_report`: removed `tabulate` dependency

## Outcome

G96 best (MAE 1.924); G192/G288 degrade. G96 replication passed.
