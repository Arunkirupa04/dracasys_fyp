"""
Read-only CSRLE Ridge vs GRU explanatory analysis helpers.

Lives only under experiments/csrle_ridge_vs_gru_analysis_*/ — does not modify
frozen CSRLE or baseline code paths.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, welch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, pacf

from utils.csrle.stage2_baselines import (
    predict_baseline_residuals,
    scaled_to_unscaled_residual,
    train_ridge_residual_model,
)
from utils.csrle.stage2_training import build_condition_scalers, load_condition_frames
from utils.hybrid_artifacts import load_hybrid_artifacts
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_inference import run_hybrid_inference


def _find_repo(start: Path) -> Path:
    p = start.resolve()
    for _ in range(6):
        if (p / "utils" / "hybrid_config.py").exists():
            return p
        p = p.parent
    raise RuntimeError(f"Repository root not found from {start}")


HORIZONS = (1, 4, 12, 24, 48, 96)


@dataclass
class FrozenPaths:
    repo: Path
    csrle: Path
    stage2: Path
    stage1: Path
    b0_dir: Path
    b0_model: Path
    b0_stats: Path
    b0_baseline_csv: Path
    b0_extended_csv: Path
    alpha_map: Path
    manifest: Path
    peak_thresholds: Path


def resolve_paths(repo_root: Path | None = None) -> FrozenPaths:
    if repo_root is None:
        repo = _find_repo(Path(__file__).resolve().parent)
    else:
        repo = repo_root
    cfg_path = Path(__file__).resolve().parent / "config" / "paths.json"
    with cfg_path.open() as fh:
        cfg = json.load(fh)
    csrle = repo / cfg["csrle_experiment"]
    stage2 = repo / cfg["stage2_dir"]
    b0 = stage2 / cfg["b0_condition"]
    return FrozenPaths(
        repo=repo,
        csrle=csrle,
        stage2=stage2,
        stage1=repo / cfg["stage1_dir"],
        b0_dir=b0,
        b0_model=b0 / "models" / "hybrid_gru.keras",
        b0_stats=b0 / "models" / "residual_stats.pkl",
        b0_baseline_csv=b0 / "evaluation" / "baseline_comparison.csv",
        b0_extended_csv=b0 / "evaluation" / "extended_metrics.csv",
        alpha_map=csrle / "synthetic_validation" / "alpha_map_b0_lc.csv",
        manifest=csrle / "data" / "ground_truth" / "injection_manifest.parquet",
        peak_thresholds=repo / "experiments/peak_aware_2026-07-14_164518/peak/peak_thresholds.pkl",
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_artifacts(paths: FrozenPaths) -> pd.DataFrame:
    required = [
        paths.b0_model,
        paths.b0_stats,
        paths.b0_baseline_csv,
        paths.b0_extended_csv,
        paths.alpha_map,
        paths.manifest,
        paths.csrle / "data/b0_lc/train_syn.parquet",
        paths.csrle / "data/b0_lc/val_syn.parquet",
    ]
    rows = []
    for p in required:
        rows.append({
            "path": str(p.relative_to(paths.repo)),
            "exists": p.exists(),
            "sha256": file_sha256(p) if p.exists() else None,
            "size_bytes": p.stat().st_size if p.exists() else None,
        })
    return pd.DataFrame(rows)


def load_b0_bundle(paths: FrozenPaths) -> dict[str, Any]:
    train_df, val_df = load_condition_frames(paths.csrle, "b0_lc")
    model, res_mean, res_std, input_window = load_hybrid_artifacts(
        paths.b0_model, paths.b0_stats,
    )
    assert input_window == DEFAULT_INPUT_WINDOW
    scalers = build_condition_scalers(train_df)
    ridge = train_ridge_residual_model(
        train_df, res_mean, res_std, alpha=1.0,
    )
    return {
        "train_df": train_df,
        "val_df": val_df,
        "model": model,
        "res_mean": res_mean,
        "res_std": res_std,
        "scalers": scalers,
        "ridge": ridge,
    }


def day1_residual_triplet(
    container_id: str,
    bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    result = run_hybrid_inference(
        container_id,
        bundle["train_df"],
        bundle["val_df"],
        bundle["model"],
        bundle["scalers"],
        bundle["res_mean"],
        bundle["res_std"],
    )
    actual = np.ravel(result.actual_day1_scaled - result.day1_prophet)
    gru = np.ravel(result.day1_residual)
    ridge_scaled = predict_baseline_residuals(
        result.residual_input, "ridge", ridge_model=bundle["ridge"],
    )
    ridge_pred = scaled_to_unscaled_residual(
        ridge_scaled, bundle["res_mean"], bundle["res_std"],
    )
    return {
        "actual": actual,
        "gru": gru,
        "ridge": ridge_pred,
        "residual_input": np.ravel(result.residual_input),
        "container_id": container_id,
    }


def pooled_day1_residuals(
    container_ids: list[str],
    condition_id: str,
    paths: FrozenPaths,
) -> np.ndarray:
    """Pooled validation Day-1 Prophet residuals for a CSRLE condition."""
    if condition_id == "control":
        train_df, val_df = load_condition_frames(paths.csrle, "control")
    else:
        train_df, val_df = load_condition_frames(paths.csrle, condition_id)
    scalers = build_condition_scalers(train_df)

    chunks = []
    for cid in container_ids:
        tr = train_df[train_df["container_id"] == cid].sort_values("time_stamp")
        va = val_df[val_df["container_id"] == cid].sort_values("time_stamp")
        if len(va) < DAY1_HORIZON or len(tr) < DEFAULT_INPUT_WINDOW:
            continue
        from prophet import Prophet
        pm = Prophet(daily_seasonality=True, weekly_seasonality=False)
        pm.fit(pd.DataFrame({"ds": tr["time_stamp"], "y": tr["cpu_scaled"]}))
        fc = pm.predict(pd.DataFrame({"ds": va["time_stamp"].values[:DAY1_HORIZON]}))
        actual_scaled = va["cpu_scaled"].values[:DAY1_HORIZON]
        resid = actual_scaled - fc["yhat"].values
        chunks.append(resid)
    return np.concatenate(chunks) if chunks else np.array([])


def acf_pacf_features(x: np.ndarray, nlags: int = 40) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < nlags + 5:
        return {}
    ac = acf(x, nlags=nlags, fft=True)
    pc = pacf(x, nlags=nlags, method="ywm")
    return {
        "acf": ac,
        "pacf": pc,
        "acf_lags_1_10_mean_abs": float(np.mean(np.abs(ac[1:11]))),
        "pacf_lags_1_5_mean_abs": float(np.mean(np.abs(pc[1:6]))),
    }


def welch_spectrum(x: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    if len(x) < 16:
        return np.array([]), np.array([])
    f, p = welch(x, fs=fs, nperseg=min(256, len(x)))
    return f, p


def fft_periodogram(x: np.ndarray, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    return freqs, power


def linear_predictability_scores(x: np.ndarray) -> dict[str, float]:
    """AR(1), AR(2), linear trend R² on validation residual series."""
    x = np.asarray(x, dtype=float)
    out: dict[str, float] = {}
    if len(x) < 10:
        return out
    # AR(1)
    X1 = x[:-1].reshape(-1, 1)
    y1 = x[1:]
    r1 = LinearRegression().fit(X1, y1)
    out["ar1_r2"] = float(r2_score(y1, r1.predict(X1)))
    # AR(2)
    X2 = np.column_stack([x[1:-1], x[:-2]])
    y2 = x[2:]
    r2 = LinearRegression().fit(X2, y2)
    out["ar2_r2"] = float(r2_score(y2, r2.predict(X2)))
    # Polynomial trend on index
    t = np.arange(len(x)).reshape(-1, 1)
    lin = LinearRegression().fit(t, x)
    out["linear_trend_r2"] = float(r2_score(x, lin.predict(t)))
    t2 = np.column_stack([t, t ** 2])
    poly2 = LinearRegression().fit(t2, x)
    out["poly2_r2"] = float(r2_score(x, poly2.predict(t2)))
    out["nonlinear_gap_poly2_minus_ar2"] = out["poly2_r2"] - out["ar2_r2"]
    return out


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < order * delay + 2:
        return float("nan")
    patterns: dict[tuple[int, ...], int] = {}
    for i in range(n - delay * (order - 1)):
        window = x[i : i + delay * order : delay]
        rank = tuple(np.argsort(np.argsort(window)))
        patterns[rank] = patterns.get(rank, 0) + 1
    probs = np.array(list(patterns.values()), dtype=float)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= m + 2:
        return float("nan")
    r = r_ratio * np.std(x)
    if r < 1e-12:
        return float("nan")

    def _count(template_len: int) -> int:
        templates = np.array([x[i : i + template_len] for i in range(n - template_len)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    a = _count(m + 1)
    b = _count(m)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def complexity_features(x: np.ndarray) -> dict[str, float]:
    return {
        "permutation_entropy": permutation_entropy(x),
        "sample_entropy": sample_entropy(x),
        "std": float(np.std(x)),
    }


def spectral_overlap(actual: np.ndarray, pred: np.ndarray, fs: float = 1.0) -> dict[str, float]:
    fa, pa = welch_spectrum(actual, fs=fs)
    fp, pp = welch_spectrum(pred, fs=fs)
    if len(fa) == 0:
        return {}
    # Dominant frequency
    dom_a = float(fa[np.argmax(pa)])
    dom_p = float(fa[np.argmax(pp)])
    # Normalized spectral correlation on common grid
    min_len = min(len(pa), len(pp))
    pa_n = pa[:min_len] / (pa[:min_len].sum() + 1e-12)
    pp_n = pp[:min_len] / (pp[:min_len].sum() + 1e-12)
    spec_corr = float(np.corrcoef(pa_n, pp_n)[0, 1])
    energy_ratio = float(pp.sum() / (pa.sum() + 1e-12))
    return {
        "dom_freq_actual": dom_a,
        "dom_freq_pred": dom_p,
        "dom_freq_abs_error": abs(dom_a - dom_p),
        "spectral_corr": spec_corr,
        "spectral_energy_ratio": energy_ratio,
    }


def error_decomposition(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.ravel(actual)
    pred = np.ravel(pred)
    bias = float(np.mean(pred - actual))
    std_a, std_p = float(np.std(actual)), float(np.std(pred))
    amp_ratio = std_p / std_a if std_a > 1e-12 else float("nan")
    if std_a < 1e-12 or std_p < 1e-12:
        corr = float("nan")
        phase_lag = float("nan")
    else:
        corr = float(pearsonr(actual, pred)[0])
        lags = correlation_lags(len(actual), len(pred), mode="full")
        xcorr = correlate(actual - actual.mean(), pred - pred.mean(), mode="full")
        phase_lag = float(lags[np.argmax(xcorr)])
    return {
        "bias": bias,
        "actual_std": std_a,
        "pred_std": std_p,
        "amplitude_ratio": amp_ratio,
        "pearson_r": corr,
        "phase_lag_steps": phase_lag,
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)) if std_a > 1e-12 else float("nan"),
    }


def horizon_comparison(actual: np.ndarray, gru: np.ndarray, ridge: np.ndarray) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sl = slice(0, h)
        a, g, r = actual[sl], gru[sl], ridge[sl]
        rows.append({
            "horizon": h,
            "gru_mae": mean_absolute_error(a, g),
            "ridge_mae": mean_absolute_error(a, r),
            "gru_r": pearsonr(a, g)[0] if np.std(g) > 1e-12 else np.nan,
            "ridge_r": pearsonr(a, r)[0] if np.std(r) > 1e-12 else np.nan,
            "gru_std_ratio": np.std(g) / np.std(a) if np.std(a) > 1e-12 else np.nan,
            "ridge_std_ratio": np.std(r) / np.std(a) if np.std(a) > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows)


def save_publication_figure(fig, path_stem: Path) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
