#!/usr/bin/env python3
"""Stage 1 of C integration: export the validated stateful GPU decoder
(gpu_fast.Step, real weights) to a CoreML .mlpackage + compiled .mlmodelc that
the C engine's Obj-C++ bridge loads.

Output: <outdir>/qwen_decoder_gpu.mlpackage  (and .mlmodelc compiled alongside)
Inputs (per token):  x[1,1,1024], cos[1,1,1,128], sin[1,1,1,128],
                     wmask[1,1,512,1], amask[1,1,1,512]
Output: hidden[1,1024]  (final-normed; C does lm_head argmax + tokenizer)
States: kc_0..kc_27, vc_0..vc_27  each [1,8,512,128]
"""
import sys, time
import numpy as np, torch, coremltools as ct
import gpu_fast as g

OUT = sys.argv[1] if len(sys.argv) > 1 else "qwen_decoder_gpu.mlpackage"

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
names = ["x", "cos", "sin", "wmask", "amask"]
t0 = time.perf_counter()
ml = ct.convert(traced,
                inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, ex)],
                outputs=[ct.TensorType(name="hidden", dtype=np.float32)],
                states=states, compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15)
ml.save(OUT)
print(f"saved {OUT}  ({time.perf_counter()-t0:.1f}s)")
print("inputs:", names, " states: kc_0..kc_27, vc_0..vc_27  output: hidden[1,1024]")
