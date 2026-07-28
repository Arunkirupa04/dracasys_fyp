"""Forecast orchestration service."""

from __future__ import annotations

import time
from typing import Any

from app.config.settings import Settings
from app.inference.hybrid_pipeline import ForecastResult, run_hybrid_forecast
from app.preprocessing.scaling import resolve_scaler
from app.preprocessing.validation import validate_forecast_request
from app.services.artifact_loader import ArtifactBundle
from app.utils.errors import ServiceUnavailableError


class ForecastOrchestrator:
    """Coordinates validation, scaling, and Hybrid inference."""

    def __init__(self, artifacts: ArtifactBundle, settings: Settings) -> None:
        self._artifacts = artifacts
        self._settings = settings

    @property
    def is_ready(self) -> bool:
        return self._artifacts is not None

    def model_info(self) -> dict[str, Any]:
        art = self._artifacts
        cfg = art.production_config
        return {
            "model_version": art.model_version,
            "model_type": cfg.get("model_type", "hybrid_prophet_gru"),
            "input_window": art.input_window,
            "default_horizon_steps": art.forecast_horizon,
            "max_horizon_steps": self._settings.max_horizon_steps,
            "sampling_interval_minutes": self._settings.sampling_interval_minutes,
            "known_containers_count": len(art.known_container_ids),
            "residual_normalization": cfg.get("residual_normalization", "global"),
            "prophet": cfg.get("prophet", {}),
            "architecture_summary": (
                "GRU(256)->Dropout(0.2)->GRU(128)->Dropout(0.2)->GRU(64)->"
                "Dense(128,relu)->Dense(96)"
            ),
        }

    def forecast(
        self,
        container_id: str,
        historical_cpu: list[float],
        timestamps: list[str],
        sampling_interval_minutes: int,
        prediction_horizon_steps: int,
    ) -> tuple[ForecastResult, float]:
        """Run forecast and return result plus processing time in milliseconds."""
        if not self.is_ready:
            raise ServiceUnavailableError("Model artifacts not loaded")

        settings = self._settings
        t0 = time.perf_counter()

        history_df = validate_forecast_request(
            container_id=container_id,
            historical_cpu=historical_cpu,
            timestamps=timestamps,
            sampling_interval_minutes=sampling_interval_minutes,
            prediction_horizon_steps=prediction_horizon_steps,
            settings=settings,
        )

        scaler, scaler_mode = resolve_scaler(
            container_id, history_df, self._artifacts.scalers
        )

        result = run_hybrid_forecast(
            container_id=container_id,
            history_df=history_df,
            horizon_steps=prediction_horizon_steps,
            scaler=scaler,
            scaler_mode=scaler_mode,
            gru_model=self._artifacts.model,
            res_mean=self._artifacts.res_mean,
            res_std=self._artifacts.res_std,
            settings=settings,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, elapsed_ms

    def forecast_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process multiple forecast requests sequentially."""
        outputs: list[dict[str, Any]] = []
        for req in requests:
            try:
                result, elapsed_ms = self.forecast(
                    container_id=req["container_id"],
                    historical_cpu=req["historical_cpu"],
                    timestamps=req["timestamps"],
                    sampling_interval_minutes=req.get(
                        "sampling_interval_minutes",
                        self._settings.sampling_interval_minutes,
                    ),
                    prediction_horizon_steps=req.get(
                        "prediction_horizon_steps",
                        self._settings.default_horizon_steps,
                    ),
                )
                outputs.append({
                    "status": "success",
                    "container_id": result.container_id,
                    "processing_time_ms": round(elapsed_ms, 2),
                    "forecast": _result_to_dict(result),
                })
            except Exception as exc:
                outputs.append({
                    "status": "error",
                    "container_id": req.get("container_id"),
                    "error": str(exc),
                })
        return outputs


def _result_to_dict(result: ForecastResult) -> dict:
    return {
        "timestamps": result.forecast_timestamps,
        "predicted_cpu_percent": result.predicted_cpu,
        "prophet_component_percent": result.prophet_cpu,
        "gru_residual_component_percent": result.residual_cpu,
        "horizon_steps": result.horizon_steps,
        "sampling_interval_minutes": result.sampling_interval_minutes,
        "scaler_mode": result.scaler_mode,
        "history_steps_used": result.history_steps,
    }
