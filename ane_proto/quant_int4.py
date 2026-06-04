#!/usr/bin/env python3
"""Quantize an existing decoder .mlpackage to 4-bit weights (no re-trace) to test
whether CoreML's GPU path executes low-bit weights at reduced memory bandwidth or
(like the int8 dud) dequantizes to fp16 before the matmul.

  python quant_int4.py <src.mlpackage> <dst.mlpackage> [linear|palette]
"""
import sys, time
import coremltools as ct

src = sys.argv[1]
dst = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "linear"

ml = ct.models.MLModel(src)
t0 = time.time()
if mode == "palette":
    from coremltools.optimize.coreml import palettize_weights, OptimizationConfig, OpPalettizerConfig
    cfg = OptimizationConfig(global_config=OpPalettizerConfig(nbits=4, mode="kmeans"))
    mlq = palettize_weights(ml, cfg)
    print("palettize 4-bit kmeans")
else:
    from coremltools.optimize.coreml import linear_quantize_weights, OptimizationConfig, OpLinearQuantizerConfig
    cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32))
    mlq = linear_quantize_weights(ml, cfg)
    print("int4 linear_symmetric per-block(32)")
mlq.save(dst)
print(f"saved {dst} in {time.time()-t0:.0f}s")
