"""
SSE stream manager.
Runs the 4-sample demo loop and yields Server-Sent Events.
Each sample produces 4 events (one per module), separated by DEMO_DELAY_SECONDS.
"""
import asyncio
import json
import traceback
from typing import AsyncGenerator

from backend.config import DEMO_DELAY_SECONDS, N_DEMO_SAMPLES, N_HISTORY_ROWS


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def run_stream(
    container_df,
    m1, m2, m3, m4,
    request,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE strings.

    container_df : pandas DataFrame (1004, 7) from synthetic_generator
    m1, m2, m3, m4 : loaded module predictors
    request : FastAPI Request (used to detect client disconnect)
    """
    yield _sse({"event": "stream_start", "total_samples": N_DEMO_SAMPLES})

    for i in range(N_DEMO_SAMPLES):
        if await request.is_disconnected():
            break

        sample_num = i + 1   # 1-based for display

        # Slice the container history ending at this demo sample
        end_row   = N_HISTORY_ROWS + i + 1        # exclusive
        start_row = max(0, end_row - 1031)         # need 1031 for M3 (1000 + 31)
        raw_slice = container_df.values[start_row:end_row]  # (≤1031, 7) numpy array

        # ── Module 1 (stub) ───────────────────────────────────────────
        try:
            r_m1 = m1.predict(i)
        except Exception as e:
            r_m1 = {"module": "m1", "status": "error", "error": str(e)}
        yield _sse({"event": "module_result", "sample": sample_num, **r_m1})

        # ── Module 2 (GRU forecast) ───────────────────────────────────
        try:
            r_m2 = m2.predict(raw_slice)
        except Exception as e:
            traceback.print_exc()
            r_m2 = {"module": "m2", "status": "error", "error": str(e)}
        yield _sse({"event": "module_result", "sample": sample_num, **r_m2})

        # ── Module 3 (VAE system anomaly) ─────────────────────────────
        try:
            r_m3 = m3.detect(i)   # uses pre-generated calibrated windows
        except Exception as e:
            traceback.print_exc()
            r_m3 = {"module": "m3", "status": "error", "error": str(e)}
        yield _sse({"event": "module_result", "sample": sample_num, **r_m3})

        # ── Module 4 (AE security anomaly) ───────────────────────────
        try:
            r_m4 = m4.detect(i)   # uses pre-loaded windows, index 0..3
        except Exception as e:
            traceback.print_exc()
            r_m4 = {"module": "m4", "status": "error", "error": str(e)}
        yield _sse({"event": "module_result", "sample": sample_num, **r_m4})

        yield _sse({"event": "sample_complete", "sample": sample_num})

        # Pause between samples (skip after the last one)
        if i < N_DEMO_SAMPLES - 1:
            await asyncio.sleep(DEMO_DELAY_SECONDS)

    yield _sse({"event": "stream_complete"})
