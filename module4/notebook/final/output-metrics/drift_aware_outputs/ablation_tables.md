# Ablation tables (auto-exported)

## Table A — Offline detector configs

| config | roc_auc | f1 | mcc | delta_roc_vs_default | source |
| --- | --- | --- | --- | --- | --- |
| Full vNext HPO | 0.8446 | 0.8030 | 0.7080 | 0.1679 | metrics_vnext.json |
| vNext default (AUC-best) | 0.6767 | 0.6733 | 0.4461 | 0.0000 | metrics_vnext.json |
| Multi-seed mean | 0.7261 |  | 0.5592 | 0.0494 | metrics_vnext.json |
| Isolation Forest (mean+max) | 0.6741 | 0.7104 | 0.5034 | -0.0025 | §8 |

## Table B — Streaming monitoring policies

| config | f1_last_half | recall_last_half | fpr_last_half | f1_gain_last_half | interpretation | source |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed threshold (stream) | 0.6824 | 0.7842 | 0.2767 |  |  | §5 stream (offline thr) |
| Adaptive threshold (stream) | 0.7091 | 0.7571 | 0.2035 | 0.0268 | adaptive_supported | §5 adaptive |
| Fine-tune on drift slice |  |  |  |  |  | §6 fine-tune |

**Rule:** Do not compare offline ROC-AUC to stream FPR in the same sentence without stating they are different evaluation protocols.
