# Stage D — Window-size decision

## Inputs
- Current static ID PR-AUC (`vae_cc1_eval.pkl`): **0.6014**
- Stage D trigger rule: ID PR-AUC < 0.65 → triggered = **True**
- Joint acceptance bar (must all pass to adopt window=60):
  - ID PR-AUC ≥ 0.75
  - Drift PR-AUC ≥ 0.38
  - Drift F1 ≥ 0.30
  - Drift Precision ≥ 0.25

## Prior window=60 evidence (experiments/, read-only)
| Set | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| cc1_test | 0.918 | 0.912 | 0.986 | 0.848 |
| drift_cc2 | 0.331 | 0.155 | 0.086 | 0.800 |
| Full Model drift F1 | 0.165 | | | |

## Decision
- **joint_bar_ok**: False
- **adopt_window_60**: False
- **final_window_size**: **30**

Keep window=30 — Stage D would be triggered by ID PR-AUC<0.65, but joint drift acceptance bar fails on prior window=60 evidence (Drift PR-AUC 0.33 < 0.38, F1 0.155 < 0.30, Precision 0.086 < 0.25). Do not edit windowing_pca.ipynb / train_vae.ipynb.

`windowing_pca.ipynb` and `train_vae.ipynb` were **not** modified.
