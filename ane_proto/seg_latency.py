#!/usr/bin/env python3
"""Per-segment encoder latency on the REAL graph for a dumped mel segment.
Reports pure-fp16 on CPU_ONLY and CPU_AND_NE (the GPU-free ANE path)."""
import sys, time, statistics
import numpy as np, torch, coremltools as ct
import encoder_fidelity as ef

PREF = sys.argv[1]
n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
_, mf, mel = ef.read_bin(f"{PREF}.mel.bin")
w = ef.load_weights(); enc = ef.Encoder(w, mf).eval()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf); mel_np = mel_t.numpy()
with torch.no_grad():
    traced = torch.jit.trace(enc, mel_t)
print(f"segment graph: mel 128x{mf} -> {enc.total} tokens")
for label, units in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                     ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
    ml = ct.convert(traced, inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT16, compute_units=units,
                    minimum_deployment_target=ct.target.macOS15)
    for _ in range(5): ml.predict({"mel": mel_np})
    ts = [ (lambda t0: (ml.predict({"mel": mel_np}), (time.perf_counter()-t0)*1000)[1])(time.perf_counter()) for _ in range(20)]
    med = statistics.median(ts)
    print(f"  {label:12s} per-segment={med:6.1f} ms   x{n_seg} segments = {med*n_seg/1000:6.3f} s", flush=True)
