"""
SSE stream manager.
Runs the N_DEMO_SAMPLES-sample demo loop and yields Server-Sent Events.
Each sample produces 4 events (one per module), separated by DEMO_DELAY_SECONDS.

Module 1 is special-cased: it lives in a separate microservice with 5-30s
latency (Prophet refit per request), so it is fired once, concurrently, at
stream start via asyncio.create_task — it never blocks the M2/M3/M4 tick
loop. Its result is emitted as soon as it resolves (checked after every
sample), or awaited with a bounded timeout if the 4-sample loop finishes
first.
"""
import asyncio
import json
import traceback
from typing import AsyncGenerator

from backend.config import DEMO_DELAY_SECONDS, N_DEMO_SAMPLES, N_HISTORY_ROWS

M1_AWAIT_TIMEOUT_SECONDS = 90.0


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
    m1 : Module1Client (HTTP, async, fired once)
    m2, m3, m4 : loaded module predictors (in-process, called every sample)
    request : FastAPI Request (used to detect client disconnect)
    """
    yield _sse({"event": "stream_start", "total_samples": N_DEMO_SAMPLES})

    # Fire Module 1's forecast concurrently — do not await here.
    m1_task = asyncio.create_task(m1.forecast())
    m1_emitted = False
    yield _sse({
        "event": "module_result", "sample": 0, "module": "m1",
        "status": "pending", "label": "Long-Term Forecasting",
        "message": "Requesting 24h forecast from the Module 1 service…",
    })

    for i in range(N_DEMO_SAMPLES):
        if await request.is_disconnected():
            m1_task.cancel()
            return

        sample_num = i + 1   # 1-based for display

        # Slice the container history ending at this demo sample.
        # Module2Predictor internally uses only the last 1000 rows.
        end_row   = N_HISTORY_ROWS + i + 1        # exclusive
        raw_slice = container_df.values[:end_row]  # (>=1000, 7) numpy array

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

        # ── Module 1 (fired earlier) — emit as soon as it resolves ────
        if not m1_emitted and m1_task.done():
            try:
                r_m1 = m1_task.result()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                r_m1 = {"module": "m1", "status": "error", "error": str(e)}
            yield _sse({"event": "module_result", "sample": sample_num, **r_m1})
            m1_emitted = True

        yield _sse({"event": "sample_complete", "sample": sample_num})

        # Pause between samples (skip after the last one)
        if i < N_DEMO_SAMPLES - 1:
            await asyncio.sleep(DEMO_DELAY_SECONDS)

    # The 4-sample loop (~15s) may finish before M1's forecast (up to ~30s).
    # Wait a bit longer, bounded, rather than dropping the result.
    if not m1_emitted:
        try:
            r_m1 = await asyncio.wait_for(m1_task, timeout=M1_AWAIT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            r_m1 = {
                "module": "m1", "status": "error",
                "error": f"Module 1 forecast timed out after {M1_AWAIT_TIMEOUT_SECONDS:.0f}s",
            }
        except Exception as e:  # noqa: BLE001
            r_m1 = {"module": "m1", "status": "error", "error": str(e)}
        yield _sse({"event": "module_result", "sample": N_DEMO_SAMPLES, **r_m1})

    yield _sse({"event": "stream_complete"})
