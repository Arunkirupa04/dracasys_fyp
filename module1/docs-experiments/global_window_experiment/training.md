# GGTCE Training

**Run:** `experiments/ggtce_2026-07-26_190616`

Independent training per variant (seed 42, no weight sharing).

| Variant | Input | Sequences | Epochs (best) | Training time |
|---------|-------|-----------|---------------|---------------|
| G96 | 96 | 41,650 | 35 (best 25) | 177 s |
| G192 | 192 | 32,146 | 17 (best 7) | 107 s |
| G288 | 288 | 22,642 | 46 (best 36) | 285 s |

Training curves: `plots/01_training_curves.png`

Per-variant metadata: `variants/{g96,g192,g288}/training/training_metadata.json`
