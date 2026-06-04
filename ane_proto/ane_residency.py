#!/usr/bin/env python3
"""Check where the ANE-high-precision encoder actually runs (per-op device via
MLComputePlan) and its latency, vs pure-fp16 ANE. Confirms the matmuls stay on
the ANE and only the cheap elementwise/norm ops fall to CPU (GPU left free)."""
import sys, time, statistics, collections
import numpy as np, torch, coremltools as ct
from encoder_fidelity import Encoder, load_weights, read_bin

PREF = sys.argv[1] if len(sys.argv) > 1 else "jfk"
_, mf, mel = read_bin(f"{PREF}.mel.bin")
w = load_weights(); enc = Encoder(w, mf).eval()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf); mel_np = mel_t.numpy()
with torch.no_grad():
    traced = torch.jit.trace(enc, mel_t)

keep = {"layer_norm", "softmax", "add", "gelu"}
configs = {
    "CPU_ONLY fp16":        (ct.precision.FLOAT16, ct.ComputeUnit.CPU_ONLY),
    "pure-fp16 (CPU+ANE)":  (ct.precision.FLOAT16, ct.ComputeUnit.CPU_AND_NE),
    "ane_hp (CPU+ANE)":     (ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in keep), ct.ComputeUnit.CPU_AND_NE),
}
for label, (prec, units) in configs.items():
    t_c = time.perf_counter()
    ml = ct.convert(traced, inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                    compute_precision=prec, compute_units=units,
                    minimum_deployment_target=ct.target.macOS15)
    compile_s = time.perf_counter() - t_c
    for _ in range(5): ml.predict({"mel": mel_np})
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); ml.predict({"mel": mel_np}); ts.append((time.perf_counter()-t0)*1000)
    print(f"{label:22s} latency median={statistics.median(ts):6.1f} ms   convert+compile={compile_s:5.1f} s",
          flush=True)
