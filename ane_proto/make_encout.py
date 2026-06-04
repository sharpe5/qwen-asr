#!/usr/bin/env python3
"""make_encout.py <prefix> <mode>
Build the faithful encoder as CoreML and write <prefix>.encout.<mode>.bin
(C engine format) for QWEN_LOAD_ENC injection.

mode:
  fp16   - full float16 (pure ANE baseline)
  mixed  - float16 matmuls/convs on ANE, but layer_norm + softmax kept float32
"""
import sys, struct
import numpy as np, torch
import coremltools as ct
from encoder_fidelity import Encoder, load_weights, read_bin

pref, mode = sys.argv[1], sys.argv[2]
_, mf, mel = read_bin(f"{pref}.mel.bin")
w = load_weights()
enc = Encoder(w, mf).eval()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf)
with torch.no_grad():
    traced = torch.jit.trace(enc, mel_t)

FP32_KEEP = {"layer_norm", "softmax"}
units = ct.ComputeUnit.CPU_AND_NE
if mode == "fp16":
    precision = ct.precision.FLOAT16
elif mode == "mixed":
    precision = ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_KEEP)
elif mode == "gpu":                       # fp16 storage, but GPU = fp32 accumulation
    precision = ct.precision.FLOAT16
    units = ct.ComputeUnit.CPU_AND_GPU
elif mode == "ane_hp":                     # ANE high-precision: matmuls/convs fp16 on ANE,
    keep = {"layer_norm", "softmax", "add", "gelu"}   # elementwise/norm path fp32 on CPU
    precision = ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep)
    units = ct.ComputeUnit.CPU_AND_NE
else:
    raise SystemExit("mode must be fp16 | mixed | gpu | ane_hp")

ml = ct.convert(traced,
                inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                compute_precision=precision,
                compute_units=units,
                minimum_deployment_target=ct.target.macOS15)
out = list(ml.predict({"mel": mel_t.numpy()}).values())[0].astype(np.float32)
out = np.ascontiguousarray(out.reshape(out.shape[-2], out.shape[-1]))
with open(f"{pref}.encout.{mode}.bin", "wb") as f:
    f.write(struct.pack("<ii", out.shape[0], out.shape[1]))
    f.write(out.tobytes())
print(f"wrote {pref}.encout.{mode}.bin  {out.shape}  (mode={mode})")
