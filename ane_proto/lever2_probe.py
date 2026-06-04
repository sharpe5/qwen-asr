#!/usr/bin/env python3
"""Lever 2 de-risk: is a FIXED S=1 decode model meaningfully faster per token
than the FLEXIBLE RangeDim (S=1..512) model at S=1?

Flexible RangeDim can force CoreML to re-plan per shape and inserts dynamic-shape
glue; a fixed S=1 graph is fully specialized. This measures the per-token decode
latency of both on CPU_AND_GPU. If the gap is small (<~15%), Lever 2 is a dud
(like int8) and we skip to Lever 3.

  fixed   = gpu_fast.Step   (x[1,1,H], wmask[1,1,MAX,1], amask[1,1,1,MAX])
  flexible= ../qwen_decoder_gpu.mlpackage (gpu_chunk.Chunk, S=1..512)
"""
import time
import numpy as np, torch, coremltools as ct
import gpu_fast as g

H, HD, MAX, NL, NKV = g.H, g.HD, g.MAX, g.NL, g.NKV
N = 300  # timed steps


def conv_fixed_s1(w):
    step = g.Step(w).eval()
    ex = (torch.zeros(1, 1, H), torch.zeros(1, 1, 1, HD), torch.zeros(1, 1, 1, HD),
          torch.zeros(1, 1, MAX, 1), torch.zeros(1, 1, 1, MAX))
    with torch.no_grad():
        tr = torch.jit.trace(step, ex)
    states = []
    for i in range(NL):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"vc_{i}"))
    names = ["x", "cos", "sin", "wmask", "amask"]
    t0 = time.perf_counter()
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, ex)],
                    states=states, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU,
                    minimum_deployment_target=ct.target.macOS15)
    ml.save("../qwen_decoder_gpu_s1.mlpackage")
    print(f"  fixed S=1 converted+saved in {time.perf_counter()-t0:.1f}s")
    return ml


def time_fixed(ml, pos=200):
    st = ml.make_state()
    cos, sin = g.rope_at(pos)
    wmask = (np.arange(MAX) == pos).astype(np.float32).reshape(1, 1, MAX, 1)
    amask = np.where(np.arange(MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, MAX)
    feed = {"x": np.zeros((1, 1, H), np.float32), "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}
    for _ in range(10):
        ml.predict(feed, state=st)
    t0 = time.perf_counter()
    for _ in range(N):
        ml.predict(feed, state=st)
    return (time.perf_counter() - t0) / N * 1000


def time_flex(ml, pos=200):
    st = ml.make_state()
    cos = np.zeros((1, 1, HD), np.float32); sin = np.zeros((1, 1, HD), np.float32)
    c, s = g.rope_at(pos); cos[0, 0] = c.reshape(HD); sin[0, 0] = s.reshape(HD)
    wmask = np.zeros((MAX, 1), np.float32); wmask[pos, 0] = 1.0
    amask = np.where(np.arange(MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, MAX)
    feed = {"x": np.zeros((1, 1, H), np.float32), "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}
    for _ in range(10):
        ml.predict(feed, state=st)
    t0 = time.perf_counter()
    for _ in range(N):
        ml.predict(feed, state=st)
    return (time.perf_counter() - t0) / N * 1000


def main():
    w = g.load_w()
    print("converting fixed S=1 ...", flush=True)
    mlf = conv_fixed_s1(w)
    print("loading flexible (../qwen_decoder_gpu.mlpackage) ...", flush=True)
    mlx = ct.models.MLModel("../qwen_decoder_gpu.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_AND_GPU)
    print(f"\ntiming {N} S=1 decode steps each (warm) ...", flush=True)
    tf = time_fixed(mlf)
    tx = time_flex(mlx)
    print(f"\n  FIXED   S=1 : {tf:.2f} ms/token")
    print(f"  FLEXIBLE S=1: {tx:.2f} ms/token")
    print(f"  speedup of fixed over flexible: {tx/tf:.2f}x  ({100*(tx-tf)/tx:+.1f}%)")
    print(f"\n  verdict: {'WORTH IT' if tx/tf > 1.15 else 'DUD (skip Lever 2, go to Lever 3)'}")


if __name__ == "__main__":
    main()
