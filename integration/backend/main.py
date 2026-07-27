"""
DracaSys Integration Backend — FastAPI app.

Endpoints:
  GET /api/status  → model load status + health check
  GET /api/stream  → SSE stream (4 samples × 4 modules)
"""
import sys
from pathlib import Path

# Ensure the integration/ root is on sys.path so relative imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import N_DEMO_SAMPLES
from backend.data.synthetic_generator import generate as gen_container_stream
from backend.inference.module1_stub import Module1Stub
from backend.inference.module2_predictor import Module2Predictor
from backend.inference.module3_detector import Module3Detector
from backend.inference.module4_detector import Module4Detector
from backend.streaming.stream_manager import run_stream

app = FastAPI(title="DracaSys Integration Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Global state (loaded once at startup)
_state: dict = {}


@app.on_event("startup")
async def startup():
    print("Loading models…")

    _state["m1"] = Module1Stub()
    print("[M1] Stub ready")

    _state["m2"] = Module2Predictor()
    _state["m3"] = Module3Detector()
    _state["m4"] = Module4Detector()

    # Generate synthetic container stream (Stream A)
    _state["container_df"] = gen_container_stream()
    print(f"[Data] Container stream: {_state['container_df'].shape}")

    print("All models loaded. Ready.")


@app.get("/api/status")
async def status():
    return {
        "ready": True,
        "modules": {
            "m1": "stub — not implemented",
            "m2": "loaded — AdaptiveGRUModel (H1/H2/H3)",
            "m3": "loaded — VAEAloneDetector",
            "m4": "loaded — SequenceBottleneckAE HPO_best",
        },
        "demo_samples": N_DEMO_SAMPLES,
    }


@app.get("/api/stream")
async def stream(request: Request):
    return StreamingResponse(
        run_stream(
            container_df=_state["container_df"],
            m1=_state["m1"],
            m2=_state["m2"],
            m3=_state["m3"],
            m4=_state["m4"],
            request=request,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
