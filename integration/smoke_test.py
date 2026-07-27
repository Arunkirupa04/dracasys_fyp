"""Quick smoke test for all inference wrappers."""
import sys
sys.path.insert(0, ".")

from backend.data.synthetic_generator import generate
from backend.inference.module1_stub import Module1Stub
from backend.inference.module2_predictor import Module2Predictor
from backend.inference.module3_detector import Module3Detector
from backend.inference.module4_detector import Module4Detector

print("=== Loading models ===")
m1 = Module1Stub()
m2 = Module2Predictor()
m3 = Module3Detector()
m4 = Module4Detector()

print("\n=== Generating data ===")
df = generate()
raw = df.values
print(f"Container stream: {raw.shape}")

print("\n=== Running inference on 4 samples ===")
for i in range(4):
    end   = 1000 + i + 1
    start = max(0, end - 1031)
    sl    = raw[start:end]

    r1 = m1.predict(i)
    r2 = m2.predict(sl)
    r3 = m3.detect(i)
    r4 = m4.detect(i)

    print(f"\n--- Sample {i+1} ---")
    print(f"  M1: {r1['status']}")
    h1 = r2['horizons']['h1']
    print(f"  M2: cpu={h1['cpu_usage']} s  mem={h1['mem_usage_mb']} MB  (H1=15s)")
    print(f"  M3: mse={r3['reconstruction_mse']:.5f}  anomaly={r3['is_anomaly']}  thr={r3['threshold']:.5f}")
    print(f"  M4: score={r4['score']:.5f}  anomaly={r4['is_anomaly']}  thr={r4['threshold']:.5f}  attack={r4['attack_type']}")

print("\n=== All OK ===")
