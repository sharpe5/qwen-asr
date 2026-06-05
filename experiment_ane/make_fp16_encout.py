#!/usr/bin/env python3
"""Read <prefix>.mel.bin (dumped by the C engine), run the faithful encoder as
CoreML fp16 on CPU+ANE, and write <prefix>.encout.fp16.bin in the C engine's
binary format ([int tokens, int dim] header + f32 data) so it can be injected
back via QWEN_LOAD_ENC for an end-to-end WER check."""
import sys, struct
import numpy as np, torch
import coremltools as ct
from encoder_fidelity import Encoder, load_weights, read_bin

pref = sys.argv[1]
_, mf, mel = read_bin(f"{pref}.mel.bin")
w = load_weights()
enc = Encoder(w, mf).eval()
mel_t = torch.from_numpy(mel).view(1, 1, 128, mf)
with torch.no_grad():
    traced = torch.jit.trace(enc, mel_t)
ml = ct.convert(traced,
                inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.macOS15)
out = list(ml.predict({"mel": mel_t.numpy()}).values())[0].astype(np.float32)
out = np.ascontiguousarray(out.reshape(out.shape[-2], out.shape[-1]))
with open(f"{pref}.encout.fp16.bin", "wb") as f:
    f.write(struct.pack("<ii", out.shape[0], out.shape[1]))
    f.write(out.tobytes())
print(f"wrote {pref}.encout.fp16.bin  {out.shape}")
