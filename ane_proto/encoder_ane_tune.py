#!/usr/bin/env python3
"""Sweep mixed-precision policies that KEEP the heavy matmuls/convs in fp16 on
the ANE, but selectively promote chosen op-types to fp32 (which fall to the CPU
under CPU_AND_NE, leaving the GPU free). Goal: recover the f32-equivalent
encoder output while staying on the ANE.

For each policy: report encoder-output fidelity (cos, rel_l2 vs the C f32
reference) and write <prefix>.encout.<tag>.bin for an end-to-end decode CER check.
"""
import sys, struct
import numpy as np, torch, coremltools as ct
from encoder_fidelity import Encoder, load_weights, read_bin

PREF = sys.argv[1] if len(sys.argv) > 1 else "zh_28s"

# policies: tag -> set of op_types to keep in fp32 (everything else fp16 on ANE)
POLICIES = {
    "fp16":            set(),
    "norm_sm":         {"layer_norm", "softmax"},
    "add":             {"add"},
    "norm_sm_add":     {"layer_norm", "softmax", "add"},
    "norm_sm_add_mul": {"layer_norm", "softmax", "add", "mul"},
    "norm_sm_add_gelu":{"layer_norm", "softmax", "add", "gelu"},
}

_, mf, mel = read_bin(f"{PREF}.mel.bin")
T, OD, ref = read_bin(f"{PREF}.encout.bin")
ref_flat = ref.reshape(-1)
w = load_weights()
enc = Encoder(w, mf).eval()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf)
mel_np = mel_t.numpy()
with torch.no_grad():
    traced = torch.jit.trace(enc, mel_t)

print(f"\npolicy sweep on {PREF}  (ref {T}x{OD}, CPU_AND_NE so fp32 ops -> CPU, GPU free)\n")
print(f"  {'policy':18s} {'fp32 op-types':32s} {'cos':>9s} {'rel_l2':>8s}")
for tag, keep in POLICIES.items():
    if not keep:
        prec = ct.precision.FLOAT16
    else:
        prec = ct.transform.FP16ComputePrecision(op_selector=lambda op, k=keep: op.op_type not in k)
    ml = ct.convert(traced,
                    inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                    compute_precision=prec,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.macOS15)
    got = list(ml.predict({"mel": mel_np}).values())[0].astype(np.float32).reshape(-1)
    cos = float(np.dot(ref_flat, got) / (np.linalg.norm(ref_flat) * np.linalg.norm(got)))
    rel = float(np.linalg.norm(ref_flat - got) / np.linalg.norm(ref_flat))
    print(f"  {tag:18s} {','.join(sorted(keep)) or '(none)':32s} {cos:9.6f} {rel:8.5f}")
    out = got.reshape(T, OD)
    with open(f"{PREF}.encout.{tag}.bin", "wb") as f:
        f.write(struct.pack("<ii", T, OD)); f.write(np.ascontiguousarray(out).tobytes())
