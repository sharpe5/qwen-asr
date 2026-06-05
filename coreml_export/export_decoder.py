#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["coremltools==9.0", "torch==2.5.1", "numpy"]
# ///
# Run with `uv run export_decoder.py` — uv provisions the interpreter + deps from
# the metadata above (coremltools 9.0's BlobWriter has no wheel for Python >= 3.13).
"""Export the flexible-sequence GPU decoder (gpu_chunk.Chunk, real weights) to a
CoreML .mlpackage the C bridge loads. One model serves BOTH batched prefill
(S=N, one call) and decode (S=1) via a RangeDim sequence axis + matmul-scatter
write.

Output: <outdir>/qwen_decoder_gpu.mlpackage
Inputs (S = 1..512 flexible): x[1,S,1024] cos[1,S,128] sin[1,S,128]
                              wmask[512,S] amask[1,1,S,512]
Output: hidden[1,S,1024]   States: kc_0..kc_27, vc_0..vc_27  [1,8,512,128]
"""
import os, sys, time
import numpy as np, torch, coremltools as ct
import gpu_chunk as gc
import gpu_fast as g

_tag = g.model_tag()
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(g._REPO, f"qwen_decoder_gpu_{_tag}.mlpackage")
H, HD, MAX, NL, NKV = g.H, g.HD, g.MAX, g.NL, g.NKV

w = g.load_w()
chunk = gc.Chunk(w).eval()
Sx = 381
ex = (torch.zeros(1, Sx, H), torch.zeros(1, Sx, HD), torch.zeros(1, Sx, HD),
      torch.zeros(MAX, Sx), torch.zeros(1, 1, Sx, MAX))
with torch.no_grad():
    traced = torch.jit.trace(chunk, ex)
states = []
for i in range(NL):
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"kc_{i}"))
    states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"vc_{i}"))
sr = ct.RangeDim(lower_bound=1, upper_bound=MAX, default=Sx)
inputs = [
    ct.TensorType(name="x",     shape=(1, sr, H),  dtype=np.float32),
    ct.TensorType(name="cos",   shape=(1, sr, HD), dtype=np.float32),
    ct.TensorType(name="sin",   shape=(1, sr, HD), dtype=np.float32),
    ct.TensorType(name="wmask", shape=(MAX, sr),   dtype=np.float32),
    ct.TensorType(name="amask", shape=(1, 1, sr, MAX), dtype=np.float32),
]
QUANT = (len(sys.argv) > 2 and sys.argv[2] == "int8")
t0 = time.perf_counter()
ml = ct.convert(traced, inputs=inputs,
                outputs=[ct.TensorType(name="hidden", dtype=np.float32)],
                states=states, compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15)
if QUANT:
    from coremltools.optimize.coreml import (linear_quantize_weights,
                                             OptimizationConfig, OpLinearQuantizerConfig)
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel"))
    ml = linear_quantize_weights(ml, cfg)   # weight-only int8 -> ~half per-token bandwidth
    print("applied int8 weight-only quantization (per-channel symmetric)")
ml.save(OUT)
print(f"saved {OUT} ({time.perf_counter()-t0:.1f}s) flexible S=1..{MAX}{' int8' if QUANT else ' fp16'}")
