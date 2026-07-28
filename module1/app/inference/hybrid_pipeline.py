"""End-to-end Hybrid Prophet + GRU inference for API requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.config.settings import Settings
from app.inference.gru_predictor import predict_residuals
from app.preprocessing.scaling import scale_history
from app.preprocessing.validation import build_future_timestamps
from app.prophet.forecaster import fit_and_forecast
from app.utils.timestamps import format_forecast_timestamp
from app.utils.timestamps import format_forecast_timestamp


@dataclass
class ForecastResult:
    """Internal forecast result before API serialisation."""

    container_id: str
    scaler_mode: str
    history_steps: int
    horizon_steps: int
    forecast_timestamps: list[str]
    predicted_cpu: list[float]
    prophet_cpu: list[float]
    residual_cpu: list[float]
    sampling_interval_minutes: int


def run_hybrid_forecast(
    container_id: str,
    history_df: pd.DataFrame,
    horizon_steps: int,
    scaler: MinMaxScaler,
    scaler_mode: str,
    gru_model: Any,
    res_mean: float,
    res_std: float,
    settings: Settings,
) -> ForecastResult:
    """
    Execute the frozen Hybrid inference pipeline on validated history.

    Steps
    -----
    1. Scale historical CPU (MinMax)
    2. Fit Prophet on history → forecast future seasonal component
    3. Compute train residuals → global z-score → last 96 for GRU input
    4. GRU predicts future scaled residuals
    5. Combine Prophet + residual → inverse MinMax → CPU %
    """
    scaled_history = scale_history(history_df, scaler)

    future_ts = build_future_timestamps(
        scaled_history["time_stamp"].iloc[-1],
        horizon_steps,
        settings.sampling_interval_minutes,
    )

    _, prophet_future, train_yhat = fit_and_forecast(
        scaled_history,
        future_ts,
        daily_seasonality=settings.prophet_daily_seasonality,
        weekly_seasonality=settings.prophet_weekly_seasonality,
    )

    residuals = scaled_history["cpu_scaled"].values - train_yhat.values
    residual_scaled = (residuals - res_mean) / res_std

    day1_residual_scaled = predict_residuals(
        gru_model,
        residual_scaled,
        settings.input_window,
        horizon_steps,
    )
    day1_residual = (day1_residual_scaled * res_std) + res_mean

    prophet_component = prophet_future["yhat"].values[:horizon_steps]
    final_scaled = prophet_component + day1_residual

    final_real = scaler.inverse_transform(final_scaled.reshape(-1, 1)).ravel()
    prophet_real = scaler.inverse_transform(prophet_component.reshape(-1, 1)).ravel()
    residual_real = final_real - prophet_real

    final_real = np.clip(final_real, settings.cpu_min_percent, settings.cpu_max_percent)

    return ForecastResult(
        container_id=container_id,
        scaler_mode=scaler_mode,
        history_steps=len(history_df),
        horizon_steps=horizon_steps,
        forecast_timestamps=[
            format_forecast_timestamp(ts) for ts in future_ts
        ],
        predicted_cpu=[float(v) for v in final_real],
        prophet_cpu=[float(v) for v in prophet_real],
        residual_cpu=[float(v) for v in residual_real],
        sampling_interval_minutes=settings.sampling_interval_minutes,
    )
