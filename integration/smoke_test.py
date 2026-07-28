"""Quick smoke test for all inference wrappers."""
import asyncio
import sys

sys.path.insert(0, ".")

from backend.config import N_DEMO_SAMPLES, N_HISTORY_ROWS
from backend.data.synthetic_generator import generate
from backend.inference.module1_client import Module1Client
from backend.inference.module2_predictor import Module2Predictor
from backend.inference.module3_detector import Module3Detector
from backend.inference.module4_detector import Module4Detector

print("=== Loading models ===")
m1 = Module1Client()
m2 = Module2Predictor()
m3 = Module3Detector()
m4 = Module4Detector()

print("\n=== Checking Module 1 service ===")
m1_healthy = asyncio.run(m1.check_health())
if m1_healthy:
    print("  M1 reachable — requesting one forecast (fired once, like the real demo)")
    r1 = asyncio.run(m1.forecast())
    print(f"  M1: status={r1['status']}", end="")
    if r1["status"] == "ok":
        print(f"  latency={r1['processing_time_ms']:.0f}ms  last_pred={r1['predicted_cpu_percent'][-1]:.2f}%")
    else:
        print(f"  error={r1.get('error')}")
else:
    r1 = None
    print("  M1 unreachable — start `cd module1 && python run.py` to include it in this test. Skipping.")

print("\n=== Generating data ===")
df = generate(n_history=N_HISTORY_ROWS, n_demo=N_DEMO_SAMPLES)
raw = df.values
print(f"Container stream: {raw.shape}")

print(f"\n=== Running inference on {N_DEMO_SAMPLES} samples ===")
for i in range(N_DEMO_SAMPLES):
    end = N_HISTORY_ROWS + i + 1
    sl  = raw[:end]

    r2 = m2.predict(sl)
    r3 = m3.detect(i)
    r4 = m4.detect(i)

    print(f"\n--- Sample {i+1} ---")
    print(f"  M1: {'skipped (service unreachable)' if r1 is None else r1['status']}")
    a  = r2['actual']
    h1 = r2['horizons']['h1']
    print(f"  M2: actual_cpu={a['cpu_usage']} s  actual_mem={a['mem_usage_mb']} MB  ->  pred_cpu(H1)={h1['cpu_usage']} s  pred_mem(H1)={h1['mem_usage_mb']} MB")
    print(f"  M3: mse={r3['reconstruction_mse']:.5f}  anomaly={r3['is_anomaly']}  thr={r3['threshold']:.5f}  {r3['anomaly_type'] or ''}")
    print(f"  M4: score={r4['score']:.5f}  anomaly={r4['is_anomaly']}  thr={r4['threshold']:.5f}  attack={r4['attack_type']}")

print("\n=== All OK ===")
