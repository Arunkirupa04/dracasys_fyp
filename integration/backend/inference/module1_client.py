"""
Module 1 — Long-Term Trend & Seasonal Forecasting (Prophet + GRU).

Unlike M2/M3/M4, Module 1 is NOT loaded in-process. It runs as its own
FastAPI microservice (module1/run.py, default port 8000) because it depends
on TensorFlow + Prophet, a heavy/separate stack from this PyTorch-only
backend. This client calls it over HTTP, exactly as documented in
module1/docs/api/integration.md.

Demo data: module1/examples/unseen_c_10312_forecast_request.json (copied to
backend/data/m1_request.json) — a real 672-step (7-day) holdout history for
an unseen container, satisfying the API's recommended history length.
Never derived from the M2/M3 15-second synthetic stream (different units,
different sampling interval — see docs/api/schemas.md).
"""
import json

import httpx

from backend.config import M1_BASE_URL, M1_HEALTH_TIMEOUT_SECONDS, M1_REQUEST_PATH, M1_TIMEOUT_SECONDS

ERROR_LABEL = "Long-Term Forecasting"
HISTORY_TAIL_STEPS = 48  # 12h at 15-min spacing — enough recent "actual" context for the chart


class Module1Client:
    name = "Module 1 — Long-Term Forecasting"

    def __init__(self):
        with open(M1_REQUEST_PATH) as f:
            self._payload = json.load(f)

        self._base_url = M1_BASE_URL.rstrip("/")
        n_steps = len(self._payload["historical_cpu"])
        print(
            f"[M1] Client configured — target={self._base_url}, "
            f"container={self._payload['container_id']}, history_steps={n_steps}"
        )

    # ------------------------------------------------------------------
    async def check_health(self) -> bool:
        """Non-fatal readiness probe used by /api/status."""
        try:
            async with httpx.AsyncClient(timeout=M1_HEALTH_TIMEOUT_SECONDS) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
                return bool(resp.json().get("model_loaded", False))
        except Exception:
            return False

    # ------------------------------------------------------------------
    async def forecast(self) -> dict:
        """
        POST /forecast on the Module 1 service using the pre-validated
        672-step payload. Async I/O — does not block the event loop while
        waiting for Prophet's 5-30s per-request fit.
        """
        try:
            async with httpx.AsyncClient(timeout=M1_TIMEOUT_SECONDS) as client:
                resp = await client.post(f"{self._base_url}/forecast", json=self._payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            return self._error(
                f"Module 1 service unreachable at {self._base_url}. "
                f"Start it with: cd module1 && python run.py"
            )
        except httpx.TimeoutException:
            return self._error(
                f"Module 1 forecast timed out after {M1_TIMEOUT_SECONDS:.0f}s "
                f"(Prophet fit on {len(self._payload['historical_cpu'])} steps)"
            )
        except httpx.HTTPStatusError as e:
            return self._error(f"Module 1 returned HTTP {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:  # noqa: BLE001
            return self._error(str(e))

        forecast = data["forecast"]
        meta = data["metadata"]

        return {
            "module":                        "m1",
            "status":                        "ok",
            "label":                         ERROR_LABEL,
            "container_id":                  data["container_id"],
            "model_version":                 meta["model_version"],
            "processing_time_ms":            meta["processing_time_ms"],
            "scaler_mode":                   meta["scaler_mode"],
            "history_steps_used":            meta["history_steps_used"],
            "horizon_steps":                 meta["horizon_steps"],
            "timestamps":                    forecast["timestamps"],
            "predicted_cpu_percent":         forecast["predicted_cpu_percent"],
            "prophet_component_percent":     forecast["prophet_component_percent"],
            "gru_residual_component_percent": forecast["gru_residual_component_percent"],
            # Recent actual history, for an actual → predicted continuity chart.
            "history_tail_cpu_percent":      self._payload["historical_cpu"][-HISTORY_TAIL_STEPS:],
            "history_tail_timestamps":       self._payload["timestamps"][-HISTORY_TAIL_STEPS:],
        }

    @staticmethod
    def _error(message: str) -> dict:
        return {
            "module": "m1",
            "status": "error",
            "label":  ERROR_LABEL,
            "error":  message,
        }
