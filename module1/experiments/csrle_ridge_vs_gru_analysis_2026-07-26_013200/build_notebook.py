#!/usr/bin/env python3
"""Generate the full interactive notebook for Ridge vs GRU analysis."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "notebooks" / "csrle_ridge_vs_gru_analysis.ipynb"
ANALYSIS = "csrle_ridge_vs_gru_analysis_2026-07-26_013200"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in t.strip().split("\n")]}


def code(t: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in t.strip().split("\n")]}


cells: list[dict] = []

cells.append(md(f"""
# CSRLE Ridge vs GRU — Explanatory Analysis

**Read-only.** Explains why Ridge outperformed the frozen Hybrid GRU on CSRLE B0.

- Analysis artifacts: `experiments/{ANALYSIS}/`
- Documentation: `docs/csrle_ridge_vs_gru_analysis/`

Execute cells in order. All figures display inline and can be saved to the analysis `plots/` folder.
"""))

cells.append(code(f"""
# 1–2. Setup and paths
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path('..').resolve()
ANALYSIS_DIR = REPO_ROOT / 'experiments' / '{ANALYSIS}'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ANALYSIS_DIR))

from analysis_lib import (
    resolve_paths, verify_frozen_artifacts, load_b0_bundle,
    day1_residual_triplet, pooled_day1_residuals,
    acf_pacf_features, welch_spectrum, fft_periodogram,
    linear_predictability_scores, complexity_features,
    spectral_overlap, error_decomposition, horizon_comparison,
    save_publication_figure,
)
from utils.hybrid_config import DAY1_HORIZON

plt.rcParams.update({{'figure.figsize': (10, 4), 'font.size': 11}})
paths = resolve_paths(REPO_ROOT)
PLOTS = ANALYSIS_DIR / 'plots'
RESULTS = ANALYSIS_DIR / 'results'
"""))

cells.append(md("## 3. Verify frozen artifacts"))
cells.append(code("verify_df = verify_frozen_artifacts(paths)\ndisplay(verify_df)\nassert verify_df['exists'].all(), 'Missing frozen CSRLE artifact'"))

cells.append(md("## 4. Load Stage 2 B0 results"))
cells.append(code("""
bundle = load_b0_bundle(paths)
baseline = pd.read_csv(paths.b0_baseline_csv)
extended = pd.read_csv(paths.b0_extended_csv)
alpha = pd.read_csv(paths.alpha_map)
container_ids = sorted(extended['container_id'].unique())
print('Frozen Stage 2 baseline Ridge r mean:', baseline.loc[baseline.baseline=='ridge','residual_pearson_r'].mean())
print('Frozen Stage 2 baseline GRU r mean:', baseline.loc[baseline.baseline=='gru','residual_pearson_r'].mean())
"""))

cells.append(md("## 5. Residual statistics (Day-1 triplets)"))
cells.append(code("""
triplets = {cid: day1_residual_triplet(cid, bundle) for cid in container_ids}
rows = []
for cid, t in triplets.items():
    rows.append({
        'container_id': cid,
        'actual_std': np.std(t['actual']),
        'gru_std': np.std(t['gru']),
        'ridge_std': np.std(t['ridge']),
        'gru_r': pearsonr(t['actual'], t['gru'])[0] if np.std(t['gru'])>1e-12 else np.nan,
        'ridge_r': pearsonr(t['actual'], t['ridge'])[0] if np.std(t['ridge'])>1e-12 else np.nan,
    })
stats_df = pd.DataFrame(rows)
display(stats_df.describe())
"""))

cells.append(md("## 6. Analysis 1 — Linearity diagnostics (ACF / PACF / spectrum / AR models)"))
cells.append(code("""
pooled_b0 = pooled_day1_residuals(container_ids, 'b0_lc', paths)
lin = linear_predictability_scores(pooled_b0)
acf_out = acf_pacf_features(pooled_b0, nlags=40)
print('Linear predictability on pooled B0 validation Prophet residuals:')
print(json.dumps({**lin, 'acf_lags_1_10_mean_abs': acf_out.get('acf_lags_1_10_mean_abs')}, indent=2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].stem(range(len(acf_out['acf'])), acf_out['acf'], basefmt=' ')
axes[0].set_title('B0 pooled residual ACF'); axes[0].set_xlabel('Lag')
axes[1].stem(range(len(acf_out['pacf'])), acf_out['pacf'], basefmt=' ')
axes[1].set_title('B0 pooled residual PACF'); axes[1].set_xlabel('Lag')
plt.tight_layout(); plt.show()

f, p = welch_spectrum(pooled_b0)
fig, ax = plt.subplots(figsize=(8,4))
ax.semilogy(f[1:], p[1:]); ax.set_title('Welch spectrum — B0 pooled residuals')
ax.set_xlabel('Frequency (1/step)'); ax.set_ylabel('Power'); plt.show()
"""))

cells.append(md("## 7. Analysis 2 — Complexity (A vs B0 vs C0)"))
cells.append(code("""
complexity_rows = []
for cond in ['control', 'b0_lc', 'c0_iid']:
    pooled = pooled_day1_residuals(container_ids, cond, paths)
    row = complexity_features(pooled); row['condition'] = cond
    complexity_rows.append(row)
display(pd.DataFrame(complexity_rows))
"""))

cells.append(md("## 8. Analysis 3 — Frequency-domain comparison (representative container)"))
cells.append(code("""
rank_df = stats_df.copy()
rank_df['delta_r'] = rank_df['ridge_r'] - rank_df['gru_r']
rep_cid = rank_df.loc[(rank_df['delta_r'] - rank_df['delta_r'].median()).abs().idxmin(), 'container_id']
t = triplets[rep_cid]
fig, ax = plt.subplots(figsize=(8,4))
for label, arr in [('Actual', t['actual']), ('GRU', t['gru']), ('Ridge', t['ridge'])]:
    f, p = welch_spectrum(arr)
    ax.semilogy(f[1:], p[1:], label=label)
ax.set_title(f'Welch spectra — {rep_cid}'); ax.legend(); plt.show()
print('GRU spectral metrics:', spectral_overlap(t['actual'], t['gru']))
print('Ridge spectral metrics:', spectral_overlap(t['actual'], t['ridge']))
"""))

cells.append(md("## 9. Analysis 4 — Variance recovery"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(stats_df['actual_std'], stats_df['gru_std'], alpha=0.6, label='GRU', s=30)
ax.scatter(stats_df['actual_std'], stats_df['ridge_std'], alpha=0.6, label='Ridge', s=30)
mx = stats_df['actual_std'].max() * 1.05
ax.plot([0, mx], [0, mx], 'k--', alpha=0.4)
ax.set_xlabel('Actual residual std'); ax.set_ylabel('Predicted residual std')
ax.set_title('B0 variance recovery'); ax.legend(); plt.show()
print('Mean GRU std ratio:', (stats_df.gru_std/stats_df.actual_std).mean())
print('Mean Ridge std ratio:', (stats_df.ridge_std/stats_df.actual_std).mean())
"""))

cells.append(md("## 10. Analysis 5 — Trajectory comparison"))
cells.append(code("""
steps = np.arange(1, DAY1_HORIZON + 1)
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(steps, t['actual'], label='Actual')
ax.plot(steps, t['gru'], label='GRU')
ax.plot(steps, t['ridge'], label='Ridge')
ax.set_title(f'Residual trajectories — {rep_cid}'); ax.legend(); plt.show()

fig, ax = plt.subplots(figsize=(9,4))
sl = slice(0, 24)
ax.plot(steps[sl], t['actual'][sl], label='Actual')
ax.plot(steps[sl], t['gru'][sl], label='GRU')
ax.plot(steps[sl], t['ridge'][sl], label='Ridge')
ax.set_title('Zoom steps 1–24'); ax.legend(); plt.show()
"""))

cells.append(md("## 11. Analysis 6 — Error decomposition"))
cells.append(code("""
g = error_decomposition(t['actual'], t['gru'])
r = error_decomposition(t['actual'], t['ridge'])
display(pd.DataFrame({'GRU': g, 'Ridge': r}).T)
"""))

cells.append(md("## 12. Analysis 7 — Horizon comparison"))
cells.append(code("""
hz_parts = []
for cid in container_ids:
    tt = triplets[cid]
    h = horizon_comparison(tt['actual'], tt['gru'], tt['ridge'])
    h['container_id'] = cid
    hz_parts.append(h)
hz_cohort = pd.concat(hz_parts).groupby('horizon').mean(numeric_only=True)
display(hz_cohort)

fig, ax = plt.subplots(1, 2, figsize=(11,4))
ax[0].plot(hz_cohort.index, hz_cohort['gru_r'], marker='o', label='GRU')
ax[0].plot(hz_cohort.index, hz_cohort['ridge_r'], marker='o', label='Ridge')
ax[0].set_title('Pearson r by horizon'); ax[0].legend()
ax[1].plot(hz_cohort.index, hz_cohort['gru_std_ratio'], marker='o', label='GRU')
ax[1].plot(hz_cohort.index, hz_cohort['ridge_std_ratio'], marker='o', label='Ridge')
ax[1].set_title('Std ratio by horizon'); ax[1].legend()
plt.show()
"""))

cells.append(md("## 13. Analysis 8 — Container ranking"))
cells.append(code("""
rank_df = stats_df.merge(alpha, on='container_id', how='left')
rank_df['delta_r'] = rank_df['ridge_r'] - rank_df['gru_r']
display(rank_df.nlargest(10, 'ridge_r')[['container_id','gru_r','ridge_r','delta_r','alpha']])
display(rank_df.nsmallest(10, 'gru_r')[['container_id','gru_r','ridge_r','delta_r','alpha']])
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(rank_df['gru_r'], rank_df['ridge_r'], alpha=0.6)
lims = [-0.5, 0.6]
ax.plot(lims, lims, 'k--', alpha=0.4)
ax.set_xlabel('GRU r'); ax.set_ylabel('Ridge r'); plt.show()
"""))

cells.append(md("## 14. Analysis 9 — Generator property correlations"))
cells.append(code("""
cols = [c for c in ['alpha','kappa_effective','delta_r','actual_std','gru_std','ridge_std'] if c in rank_df.columns]
display(rank_df[cols].corr(numeric_only=True))
"""))

cells.append(md("## 15. Analysis 10 — Evidence synthesis"))
cells.append(code("""
synthesis = {
    'mean_gru_r': float(rank_df.gru_r.mean()),
    'mean_ridge_r': float(rank_df.ridge_r.mean()),
    'fraction_ridge_beats_gru': float((rank_df.delta_r > 0).mean()),
    'mean_gru_std_ratio': float((rank_df.gru_std/rank_df.actual_std).mean()),
    'mean_ridge_std_ratio': float((rank_df.ridge_std/rank_df.actual_std).mean()),
    'b0_ar1_r2_pooled': lin.get('ar1_r2'),
    'b0_ar2_r2_pooled': lin.get('ar2_r2'),
    'b0_acf_lags_1_10_mean_abs': acf_out.get('acf_lags_1_10_mean_abs'),
}
print(json.dumps(synthesis, indent=2))
"""))

cells.append(md("""
## 16. Final findings

Quantitative synthesis is written to `experiments/csrle_ridge_vs_gru_analysis_2026-07-26_013200/results/evidence_synthesis.json` when `run_analysis.py` is executed.

Thesis-ready narrative: `docs/csrle_ridge_vs_gru_analysis/summary.md`.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print("Wrote", OUT)
