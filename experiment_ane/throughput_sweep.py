#!/usr/bin/env python3
"""Throughput sweep: launch N concurrent --gpu processes (each -t 1, 1 encode
thread = true thread=1), each transcribing the 5-min file, and measure aggregate
realtime x = (N * 300s audio) / wall. Shows how fast we can decode N files and
where the single shared GPU saturates. Compares fp16 vs int4 batched models.
Run from the qwen-asr-fork directory.
"""
import subprocess, time, os, sys

M = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b"
WAV = "ane_proto/aja_5min.wav"
BIN = "./qwen_asr"
NS = [1, 2, 4, 8, 12]

def run(model, N):
    env = dict(os.environ,
               QWEN_GPU_MODEL="qwen_decoder_gpu.mlpackage",
               QWEN_GPU_BATCH_MODEL=model,
               QWEN_GPU_ENC_THREADS="1",
               QWEN_GPU_REFILL_MIN="3")
    t0 = time.perf_counter()
    procs = [subprocess.Popen([BIN, "-d", M, "-i", WAV, "--gpu", "-t", "1", "--silent"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
             for _ in range(N)]
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0
    return wall, N * 300.0 / wall

for label, model in [("fp16", "qwen_decoder_gpu_b4.mlpackage"),
                     ("int4", "qwen_decoder_gpu_b4_int4.mlpackage")]:
    for N in NS:
        wall, agg = run(model, N)
        print(f"{label}  N={N:2d}  wall={wall:6.1f}s  aggregate={agg:6.1f}x realtime  "
              f"(per-proc {agg/N:.1f}x)", flush=True)
