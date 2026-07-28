"""Hybrid GRU residual prediction."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_residuals(
    model: Any,
    residual_window: np.ndarray,
    input_window: int,
    horizon_steps: int,
) -> np.ndarray:
    """
    Predict future scaled residuals from the last ``input_window`` values.

    Parameters
    ----------
    model:
        Loaded Keras Hybrid GRU (outputs horizon matching training, typically 96).
    residual_window:
        1-D array of scaled residuals, length >= input_window.
    horizon_steps:
        Number of future steps to return (truncated from model output).
    """
    if len(residual_window) < input_window:
        raise ValueError(f"Need at least {input_window} residual values")

    window = np.asarray(residual_window[-input_window:], dtype=float)
    x = window.reshape(1, input_window, 1)
    pred = model.predict(x, verbose=0)[0]
    return np.asarray(pred[:horizon_steps], dtype=float)
