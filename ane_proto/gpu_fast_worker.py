#!/usr/bin/env python3
"""One fast-path decode worker: builds the gpu_fast stateful model (real weights,
on-device KV State, masked write), reuses a constant input feed in a tight loop
for SECS seconds, prints tok/s. Used by gpu_fast_scaling.py for N-parallel test."""
import sys, time
import numpy as np, torch, coremltools as ct
import gpu_fast as g

SECS = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
w = g.load_w()
step = g.Step(w).eval()
ex = (torch.zeros(1, 1, g.H), torch.zeros(1, 1, 1, g.HD), torch.zeros(1, 1, 1, g.HD),
      torch.zeros(1, 1, g.MAX, 1), torch.zeros(1, 1, 1, g.MAX))
with torch.no_grad():
    traced = torch.jit.trace(step, ex)
states = []
for i in range(g.NL):
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, g.NKV, g.MAX, g.HD)), name=f"kc_{i}"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, g.NKV, g.MAX, g.HD)), name=f"vc_{i}"))
ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32)
                                for n, t in zip(["x", "cos", "sin", "wmask", "amask"], ex)],
                states=states, compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_GPU,
                minimum_deployment_target=ct.target.macOS15)
stt = ml.make_state()
pos = 300
cos, sin = g.rope_at(pos)
feed = {"x": np.zeros((1, 1, g.H), np.float32), "cos": cos, "sin": sin,
        "wmask": (np.arange(g.MAX) == pos).astype(np.float32).reshape(1, 1, g.MAX, 1),
        "amask": np.where(np.arange(g.MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, g.MAX)}
for _ in range(8):
    ml.predict(feed, state=stt)
n = 0; t0 = time.perf_counter()
while time.perf_counter() - t0 < SECS:
    ml.predict(feed, state=stt); n += 1
print(f"TOKS={n/(time.perf_counter()-t0):.2f}", flush=True)
