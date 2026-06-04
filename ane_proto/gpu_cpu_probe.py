#!/usr/bin/env python3
"""Probe: how much CPU does sustained CoreML GPU inference consume?
Builds the stateful decoder on CPU_AND_GPU and runs predict in a tight loop for
~20s. Run `ps` against this PID in parallel to read %CPU (1 core = 100%)."""
import sys, time, os
import numpy as np, torch, coremltools as ct
import decoder_stateful_bench as d   # reuse the model + dims

print(f"PID={os.getpid()}", flush=True)
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
st = ml.make_state()
feed = {"x": x.numpy()}
print("LOOPING (sample CPU now)", flush=True)
n = 0; t0 = time.perf_counter()
while time.perf_counter() - t0 < 20:
    ml.predict(feed, state=st); n += 1
print(f"did {n} predicts in {time.perf_counter()-t0:.1f}s ({n/(time.perf_counter()-t0):.0f}/s)", flush=True)
