#!/usr/bin/env python3
"""Source-level fp16-stability test. All variants are PURE fp16 on CPU_AND_NE
(clean single-precision graph -> compiles for ANE, stays fast). Only the GELU
*formulation* changes. Question: does a different GELU recover accuracy vs the
C f32 (tanh-GELU) reference WITHOUT leaving the ANE?

For each variant: fidelity (cos, rel_l2) vs C ref, latency, + write encout bin.
"""
import sys, time, statistics, struct
import numpy as np, torch, coremltools as ct
import encoder_fidelity as ef

PREF = sys.argv[1] if len(sys.argv) > 1 else "zh_28s"
_, mf, mel = ef.read_bin(f"{PREF}.mel.bin")
T, OD, ref = ef.read_bin(f"{PREF}.encout.bin")
ref_flat = ref.reshape(-1)
w = ef.load_weights()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf); mel_np = mel_t.numpy()

print(f"\nGELU fp16-stability sweep on {PREF} (ref {T}x{OD}); all PURE fp16 on CPU_AND_NE")
print(f"reference activation = tanh-GELU in f32 (the C engine)\n")
print(f"  {'gelu':9s} {'cos':>9s} {'rel_l2':>8s} {'latency':>9s}")

for mode in ["tanh", "exact", "sigmoid"]:
    ef.GELU_MODE = mode                       # gelu() reads this global at call time
    enc = ef.Encoder(w, mf).eval()
    with torch.no_grad():
        traced = torch.jit.trace(enc, mel_t)
    ml = ct.convert(traced, inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.macOS15)
    for _ in range(5): ml.predict({"mel": mel_np})
    ts = []
    for _ in range(15):
        t0 = time.perf_counter(); out = ml.predict({"mel": mel_np}); ts.append((time.perf_counter()-t0)*1000)
    got = list(out.values())[0].astype(np.float32).reshape(-1)
    cos = float(np.dot(ref_flat, got)/(np.linalg.norm(ref_flat)*np.linalg.norm(got)))
    rel = float(np.linalg.norm(ref_flat-got)/np.linalg.norm(ref_flat))
    print(f"  {mode:9s} {cos:9.6f} {rel:8.5f} {statistics.median(ts):7.1f}ms", flush=True)
    with open(f"{PREF}.encout.gelu_{mode}.bin", "wb") as fp:
        fp.write(struct.pack("<ii", T, OD)); fp.write(np.ascontiguousarray(got.reshape(T, OD)).tobytes())
