"""
Bug reproducer: torch.xpu.Event.elapsed_time() returns incorrect (negative) values on Aurora.
Compare XPU event timing against time.perf_counter() wall-clock timing.

Run on Aurora:
    python test/repro_xpu_event_timing.py
"""
import time
import torch

device = torch.device("xpu:0")
torch.xpu.set_device(0)

size = (1024, 1024)
a = torch.randn(size, device=device, dtype=torch.float32)
b = torch.randn(size, device=device, dtype=torch.float32)

# Warmup
for _ in range(3):
    c = torch.mm(a, b)
torch.xpu.synchronize()

NUM_ITER = 20

# --- Wall-clock timing (expected correct) ---
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(NUM_ITER):
    c = torch.mm(a, b)
torch.xpu.synchronize()
wall_sec = time.perf_counter() - t0

# --- XPU Event timing ---
begin = torch.xpu.Event(enable_timing=True)
end = torch.xpu.Event(enable_timing=True)
begin.record()
for _ in range(NUM_ITER):
    c = torch.mm(a, b)
end.record()
torch.xpu.synchronize()
event_ms = begin.elapsed_time(end)

print(f"Wall-clock : {wall_sec*1000:.3f} ms  ({wall_sec/NUM_ITER*1000:.3f} ms/iter)")
print(f"XPU Event  : {event_ms:.3f} ms  ({event_ms/NUM_ITER:.3f} ms/iter)")
print(f"Match      : {'YES' if abs(event_ms - wall_sec*1000) / (wall_sec*1000) < 0.1 else 'NO -- event timing is wrong'}")
