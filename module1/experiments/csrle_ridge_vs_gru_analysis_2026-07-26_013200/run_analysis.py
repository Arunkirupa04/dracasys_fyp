#!/usr/bin/env python3
"""Execute read-only Ridge vs GRU analysis; write results/plots to this experiment dir."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent


def _find_repo(start: Path) -> Path:
    p = start.resolve()
    for _ in range(6):
        if (p / "utils" / "hybrid_config.py").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Repository root not found from {start}")


REPO_ROOT = _find_repo(ANALYSIS_DIR)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ANALYSIS_DIR))

from utils.hybrid_config import DAY1_HORIZON  # noqa: E402

from analysis_lib import (  # noqa: E402
    HORIZONS,
    acf_pacf_features,
    complexity_features,
    day1_residual_triplet,
    error_decomposition,
    fft_periodogram,
    horizon_comparison,
    linear_predictability_scores,
    load_b0_bundle,
    pooled_day1_residuals,
    resolve_paths,
    save_publication_figure,
    spectral_overlap,
    verify_frozen_artifacts,
    welch_spectrum,
)

PLOTS = ANALYSIS_DIR / "plots"
RESULTS = ANALYSIS_DIR / "results"
DATA = ANALYSIS_DIR / "data"


def main() -> None:
    paths = resolve_paths()
    PLOTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # --- Verification ---
    verify_df = verify_frozen_artifacts(paths)
    verify_df.to_csv(ANALYSIS_DIR / "verification" / "artifact_verification.csv", index=False)
    with (ANALYSIS_DIR / "verification" / "artifact_verification.json").open("w") as fh:
        json.dump({"all_exist": bool(verify_df["exists"].all()), "n_files": len(verify_df)}, fh, indent=2)

    # --- Load frozen Stage 2 B0 ---
    bundle = load_b0_bundle(paths)
    baseline = pd.read_csv(paths.b0_baseline_csv)
    extended = pd.read_csv(paths.b0_extended_csv)
    alpha = pd.read_csv(paths.alpha_map)
    manifest = pd.read_parquet(paths.manifest)

    container_ids = sorted(extended["container_id"].unique())
    triplets = {cid: day1_residual_triplet(cid, bundle) for cid in container_ids}

    # --- Container-level comparison table ---
    rows = []
    for cid, t in triplets.items():
        g_dec = error_decomposition(t["actual"], t["gru"])
        r_dec = error_decomposition(t["actual"], t["ridge"])
        rows.append({
            "container_id": cid,
            "gru_pearson_r": g_dec["pearson_r"],
            "ridge_pearson_r": r_dec["pearson_r"],
            "delta_r_ridge_minus_gru": r_dec["pearson_r"] - g_dec["pearson_r"],
            "gru_std_ratio": g_dec["amplitude_ratio"],
            "ridge_std_ratio": r_dec["amplitude_ratio"],
            "gru_bias": g_dec["bias"],
            "ridge_bias": r_dec["bias"],
            "gru_phase_lag": g_dec["phase_lag_steps"],
            "ridge_phase_lag": r_dec["phase_lag_steps"],
        })
    cmp_df = pd.DataFrame(rows)
    cmp_df = cmp_df.merge(alpha, on="container_id", how="left")
    cmp_df.to_csv(RESULTS / "container_ridge_gru_comparison.csv", index=False)

    # --- Analysis 1: Linearity on pooled B0 validation residuals ---
    pooled_b0 = pooled_day1_residuals(container_ids, "b0_lc", paths)
    lin_b0 = linear_predictability_scores(pooled_b0)
    acf_b0 = acf_pacf_features(pooled_b0)
    with (RESULTS / "b0_pooled_linearity.json").open("w") as fh:
        json.dump({**lin_b0, "acf_lags_1_10_mean_abs": acf_b0.get("acf_lags_1_10_mean_abs")}, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if acf_b0:
        axes[0].stem(range(len(acf_b0["acf"])), acf_b0["acf"], basefmt=" ")
        axes[0].set_title("B0 pooled validation residual ACF")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("ACF")
        axes[1].stem(range(len(acf_b0["pacf"])), acf_b0["pacf"], basefmt=" ")
        axes[1].set_title("B0 pooled validation residual PACF")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("PACF")
    fig.tight_layout()
    save_publication_figure(fig, PLOTS / "analysis1_b0_acf_pacf")
    plt.close(fig)

    f, p = welch_spectrum(pooled_b0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(f[1:], p[1:])
    ax.set_title("B0 pooled validation residual Welch spectrum")
    ax.set_xlabel("Frequency (1/step)")
    ax.set_ylabel("Power")
    save_publication_figure(fig, PLOTS / "analysis1_b0_welch_spectrum")
    plt.close(fig)

    # --- Analysis 2: Complexity A vs B0 vs C0 ---
    complexity_rows = []
    for cond in ("control", "b0_lc", "c0_iid"):
        pooled = pooled_day1_residuals(container_ids, cond, paths)
        cf = complexity_features(pooled)
        cf["condition"] = cond
        complexity_rows.append(cf)
    complexity_df = pd.DataFrame(complexity_rows)
    complexity_df.to_csv(RESULTS / "complexity_comparison.csv", index=False)

    # --- Analysis 3-6: Per-container spectral, variance, decomposition ---
    spec_rows, decomp_rows, var_rows = [], [], []
    for cid, t in triplets.items():
        for name, pred in (("gru", t["gru"]), ("ridge", t["ridge"])):
            spec_rows.append({"container_id": cid, "model": name, **spectral_overlap(t["actual"], pred)})
            decomp_rows.append({"container_id": cid, "model": name, **error_decomposition(t["actual"], pred)})
        var_rows.append({
            "container_id": cid,
            "actual_std": float(np.std(t["actual"])),
            "gru_std": float(np.std(t["gru"])),
            "ridge_std": float(np.std(t["ridge"])),
        })
    spec_df = pd.DataFrame(spec_rows)
    decomp_df = pd.DataFrame(decomp_rows)
    var_df = pd.DataFrame(var_rows)
    spec_df.to_csv(RESULTS / "spectral_comparison.csv", index=False)
    decomp_df.to_csv(RESULTS / "error_decomposition.csv", index=False)
    var_df.to_csv(RESULTS / "variance_recovery.csv", index=False)

    # Variance cohort plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(var_df["actual_std"], var_df["gru_std"], alpha=0.6, label="GRU", s=25)
    ax.scatter(var_df["actual_std"], var_df["ridge_std"], alpha=0.6, label="Ridge", s=25)
    lims = [0, max(var_df["actual_std"].max(), var_df["ridge_std"].max()) * 1.05]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("Actual residual std")
    ax.set_ylabel("Predicted residual std")
    ax.set_title("B0 variance recovery: GRU vs Ridge")
    ax.legend()
    save_publication_figure(fig, PLOTS / "analysis4_variance_recovery_scatter")
    plt.close(fig)

    # --- Representative containers: median delta-r and best ridge ---
    rep_median = cmp_df.iloc[(cmp_df["delta_r_ridge_minus_gru"] - cmp_df["delta_r_ridge_minus_gru"].median()).abs().argmin()]["container_id"]
    rep_best_ridge = cmp_df.loc[cmp_df["ridge_pearson_r"].idxmax(), "container_id"]

    for tag, cid in (("median_delta_r", rep_median), ("best_ridge_r", rep_best_ridge)):
        t = triplets[cid]
        fig, ax = plt.subplots(figsize=(11, 4))
        steps = np.arange(1, DAY1_HORIZON + 1)
        ax.plot(steps, t["actual"], label="Actual", lw=1.5)
        ax.plot(steps, t["gru"], label="GRU", lw=1.5)
        ax.plot(steps, t["ridge"], label="Ridge", lw=1.5)
        ax.set_title(f"B0 residual trajectories — {cid} ({tag})")
        ax.set_xlabel("Horizon step (15 min)")
        ax.set_ylabel("Prophet residual")
        ax.legend()
        save_publication_figure(fig, PLOTS / f"analysis5_trajectory_{tag}_{cid}")
        plt.close(fig)

        # Zoom first 24 steps
        fig, ax = plt.subplots(figsize=(9, 4))
        sl = slice(0, 24)
        ax.plot(steps[sl], t["actual"][sl], label="Actual", lw=1.5)
        ax.plot(steps[sl], t["gru"][sl], label="GRU", lw=1.5)
        ax.plot(steps[sl], t["ridge"][sl], label="Ridge", lw=1.5)
        ax.set_title(f"B0 trajectories (zoom h=1–24) — {cid}")
        ax.legend()
        save_publication_figure(fig, PLOTS / f"analysis5_trajectory_zoom_{tag}_{cid}")
        plt.close(fig)

        # Spectral
        fa, pa = welch_spectrum(t["actual"])
        fg, pg = welch_spectrum(t["gru"])
        fr, pr = welch_spectrum(t["ridge"])
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(fa[1:], pa[1:], label="Actual")
        ax.semilogy(fg[1:], pg[1:], label="GRU")
        ax.semilogy(fr[1:], pr[1:], label="Ridge")
        ax.set_title(f"Welch spectra — {cid}")
        ax.legend()
        save_publication_figure(fig, PLOTS / f"analysis3_spectral_{tag}_{cid}")
        plt.close(fig)

    # --- Analysis 7: Horizon cohort ---
    hz_rows = []
    for cid, t in triplets.items():
        hdf = horizon_comparison(t["actual"], t["gru"], t["ridge"])
        hdf["container_id"] = cid
        hz_rows.append(hdf)
    hz_all = pd.concat(hz_rows, ignore_index=True)
    hz_cohort = hz_all.groupby("horizon").mean(numeric_only=True).reset_index()
    hz_cohort.to_csv(RESULTS / "horizon_ridge_vs_gru_cohort.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hz_cohort["horizon"], hz_cohort["gru_r"], marker="o", label="GRU r")
    axes[0].plot(hz_cohort["horizon"], hz_cohort["ridge_r"], marker="o", label="Ridge r")
    axes[0].set_title("Within-trajectory Pearson r by horizon (cohort mean)")
    axes[0].set_xlabel("Horizon h")
    axes[0].legend()
    axes[1].plot(hz_cohort["horizon"], hz_cohort["gru_std_ratio"], marker="o", label="GRU std ratio")
    axes[1].plot(hz_cohort["horizon"], hz_cohort["ridge_std_ratio"], marker="o", label="Ridge std ratio")
    axes[1].set_title("Std ratio by horizon (cohort mean)")
    axes[1].legend()
    fig.tight_layout()
    save_publication_figure(fig, PLOTS / "analysis7_horizon_comparison")
    plt.close(fig)

    # --- Analysis 8: Rankings ---
    best_gru = cmp_df.nlargest(10, "gru_pearson_r")[["container_id", "gru_pearson_r", "ridge_pearson_r"]]
    best_ridge = cmp_df.nlargest(10, "ridge_pearson_r")[["container_id", "gru_pearson_r", "ridge_pearson_r"]]
    worst_gru = cmp_df.nsmallest(10, "gru_pearson_r")[["container_id", "gru_pearson_r", "ridge_pearson_r"]]
    best_gru.to_csv(RESULTS / "top10_gru_containers.csv", index=False)
    best_ridge.to_csv(RESULTS / "top10_ridge_containers.csv", index=False)
    worst_gru.to_csv(RESULTS / "bottom10_gru_containers.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(cmp_df["gru_pearson_r"], cmp_df["ridge_pearson_r"], alpha=0.6, s=30)
    lim = max(cmp_df[["gru_pearson_r", "ridge_pearson_r"]].abs().max().max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.set_xlabel("GRU Pearson r")
    ax.set_ylabel("Ridge Pearson r")
    ax.set_title("Container-level residual correlation: Ridge vs GRU (B0)")
    save_publication_figure(fig, PLOTS / "analysis8_container_scatter")
    plt.close(fig)

    # --- Analysis 9: Correlations with generator properties ---
    corr_cols = ["alpha", "kappa_effective", "delta_r_ridge_minus_gru", "gru_std_ratio", "ridge_std_ratio"]
    avail = [c for c in corr_cols if c in cmp_df.columns]
    corr_mat = cmp_df[avail].corr(numeric_only=True)
    corr_mat.to_csv(RESULTS / "generator_property_correlations.csv")

    # --- Analysis 10: Synthesis metrics ---
    gru_decomp = decomp_df[decomp_df["model"] == "gru"].mean(numeric_only=True)
    ridge_decomp = decomp_df[decomp_df["model"] == "ridge"].mean(numeric_only=True)
    synthesis = {
        "cohort_mean_gru_r": float(cmp_df["gru_pearson_r"].mean()),
        "cohort_mean_ridge_r": float(cmp_df["ridge_pearson_r"].mean()),
        "fraction_ridge_beats_gru_r": float((cmp_df["delta_r_ridge_minus_gru"] > 0).mean()),
        "cohort_mean_gru_std_ratio": float(cmp_df["gru_std_ratio"].mean()),
        "cohort_mean_ridge_std_ratio": float(cmp_df["ridge_std_ratio"].mean()),
        "b0_ar1_r2_pooled": lin_b0.get("ar1_r2"),
        "b0_ar2_r2_pooled": lin_b0.get("ar2_r2"),
        "b0_acf_lags_1_10_mean_abs": acf_b0.get("acf_lags_1_10_mean_abs"),
        "gru_mean_amplitude_ratio": float(gru_decomp.get("amplitude_ratio", np.nan)),
        "ridge_mean_amplitude_ratio": float(ridge_decomp.get("amplitude_ratio", np.nan)),
        "gru_mean_abs_bias": float(abs(gru_decomp.get("bias", np.nan))),
        "ridge_mean_abs_bias": float(abs(ridge_decomp.get("bias", np.nan))),
        "mean_spectral_corr_gru": float(spec_df[spec_df["model"] == "gru"]["spectral_corr"].mean()),
        "mean_spectral_corr_ridge": float(spec_df[spec_df["model"] == "ridge"]["spectral_corr"].mean()),
    }
    with (RESULTS / "evidence_synthesis.json").open("w") as fh:
        json.dump(synthesis, fh, indent=2)

    # Error decomposition bar chart
    metrics = ["amplitude_ratio", "pearson_r", "mae"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, [gru_decomp[m] for m in metrics], width=w, label="GRU")
    ax.bar(x + w / 2, [ridge_decomp[m] for m in metrics], width=w, label="Ridge")
    ax.set_xticks(x, metrics)
    ax.set_title("Cohort-mean error decomposition (B0 Day-1)")
    ax.legend()
    save_publication_figure(fig, PLOTS / "analysis6_error_decomposition_bars")
    plt.close(fig)

    print(json.dumps(synthesis, indent=2))


if __name__ == "__main__":
    main()
