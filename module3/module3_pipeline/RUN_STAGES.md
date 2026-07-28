# How to execute the Stage A–E pipeline updates

Data (`data/processed/windows_cc1/*.npy` and split CSVs) is gitignored and
must be present locally. Model artifacts under `module3/models/` are already present.

## Order (python39-pytorch kernel)

```text
1. vae_eval.ipynb                    # Stage A diagnostics → updates vae_cc1_eval.pkl
2. adaptive_threshold_blended.ipynb  # Stage B relative scoring
3. incremental_learning.ipynb        # Stage C stabilized FT
4. baseline_comparison.ipynb         # CC2-only baselines + VAE↔Gaussian score fusion (base PR-AUC)
5. final_comparison.ipynb            # Ablation + fusion row + Stage E/D
```

### Base PR-AUC without window=60

Run **`baseline_comparison.ipynb`** (needs `windows_cc1` + `vae_cc1.pt`).  
Deployed fusion: `α=0.5` → `α·z(VAE) + (1-α)·z(Gaussian)`.  
Success: `fusion_deployed` cc1_test PR-AUC **>** VAE (~0.60); stretch **≥ 0.65**.

Headless pattern:

```powershell
jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=python39-pytorch `
  module3/module3_pipeline/vae_eval.ipynb
```

## Stage D status (already decided without re-windowing)

See `stage_d_window_decision.md` and `models/stage_d_window_decision.pkl`.

**Final window size remains 30.** Window=60 is rejected by the joint drift bar.
