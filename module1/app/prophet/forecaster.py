"""Prophet seasonal forecast component."""

from __future__ import annotations

from typing import Any

import pandas as pd
from prophet import Prophet


def fit_and_forecast(
    history_df: pd.DataFrame,
    future_timestamps: pd.DatetimeIndex,
    daily_seasonality: bool = True,
    weekly_seasonality: bool = False,
) -> tuple[Any, pd.DataFrame, pd.Series]:
    """
    Fit Prophet on historical scaled CPU and forecast future steps.

    Returns (model, full_forecast_df, future_yhat).
    """
    prophet_train = pd.DataFrame({
        "ds": history_df["time_stamp"],
        "y": history_df["cpu_scaled"],
    })

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
    )
    model.fit(prophet_train)

    future_df = pd.DataFrame({"ds": future_timestamps})
    forecast = model.predict(future_df)
    train_forecast = model.predict(prophet_train[["ds"]])

    return model, forecast, train_forecast["yhat"]
