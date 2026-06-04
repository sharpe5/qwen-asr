#!/usr/bin/env python3
"""Steady-state throughput: N concurrent --gpu processes (-t 1), each transcribing
a 30-min clip (startup amortized ~ like the real 6h files), fp16 batched B=4.
aggregate x = N * 1800s / wall. Shows where the shared machine saturates (likely
CPU encode before the GPU). Run from qwen-asr-fork."""
import subprocess, time, os

M = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b"
WAV = "ane_proto/aja_30min.wav"; AUDIO = 1800.0; BIN = "./qwen_asr"

def run(N, enc):
    env = dict(os.environ, QWEN_GPU_MODEL="qwen_decoder_gpu.mlpackage",
               QWEN_GPU_BATCH_MODEL="qwen_decoder_gpu_b4.mlpackage",
               QWEN_GPU_ENC_THREADS=str(enc), QWEN_GPU_REFILL_MIN="3")
    t0 = time.perf_counter()
    ps = [subprocess.Popen([BIN, "-d", M, "-i", WAV, "--gpu", "-t", "1", "--silent"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
          for _ in range(N)]
    for p in ps: p.wait()
    wall = time.perf_counter() - t0
    return wall, N * AUDIO / wall

for N in [1, 2, 4, 6, 8]:
    wall, agg = run(N, 2)
    print(f"enc=2 N={N}: wall={wall:6.1f}s aggregate={agg:6.1f}x  (per-proc {agg/N:5.1f}x)", flush=True)
