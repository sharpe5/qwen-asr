#!/usr/bin/env python3
"""One GPU decode worker: builds the stateful decoder on CPU_AND_GPU, warms up,
then runs decode steps in a tight loop for SECS seconds and prints its tok/s.
Used by gpu_scaling_test.py to measure aggregate throughput of N parallel
workers all sharing the single GPU."""
import sys, time
import numpy as np, torch, coremltools as ct
import decoder_stateful_bench as d

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
torch.manual_seed(0)
m = d.DecoderStep().eval()
x = torch.randn(1, 1, d.H)
with torch.no_grad():
    traced = torch.jit.trace(m, (x,))
states = []
for i in range(d.L_LAYERS):
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, d.NKV, d.CTX, d.HD)), name=f"kc_{i}"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, d.NKV, d.CTX, d.HD)), name=f"vc_{i}"))
ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
                states=states, compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_GPU,
                minimum_deployment_target=ct.target.macOS15)
st = ml.make_state(); feed = {"x": x.numpy()}
for _ in range(8):
    ml.predict(feed, state=st)
n = 0; t0 = time.perf_counter()
while time.perf_counter() - t0 < SECS:
    ml.predict(feed, state=st); n += 1
dt = time.perf_counter() - t0
print(f"TOKS={n/dt:.2f}", flush=True)
