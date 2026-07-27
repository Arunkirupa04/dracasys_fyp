"""Per-container CPU MinMax scaling."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.utils.errors import NotFoundError, ServiceUnavailableError


def load_scalers(path: Path) -> dict[str, MinMaxScaler]:
    if not path.exists():
        raise ServiceUnavailableError(
            "scalers.pkl artifact missing",
            {"path": str(path)},
        )
    with path.open("rb") as handle:
        scalers = pickle.load(handle)
    if not isinstance(scalers, dict):
        raise ServiceUnavailableError("scalers.pkl has unexpected format")
    return scalers


def resolve_scaler(
    container_id: str,
    history_df: pd.DataFrame,
    scalers: dict[str, MinMaxScaler],
) -> tuple[MinMaxScaler, str]:
    """
    Return scaler and mode ('known' uses frozen training scaler, 'new' fits on history).
    """
    if container_id in scalers:
        return scalers[container_id], "known"

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(history_df[["cpu"]])
    return scaler, "new"


def scale_history(
    history_df: pd.DataFrame,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    out = history_df.copy()
    out["cpu_scaled"] = scaler.transform(out[["cpu"]]).ravel()
    return out
