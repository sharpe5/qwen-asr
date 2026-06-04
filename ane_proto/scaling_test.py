#!/usr/bin/env python3
"""CPU process-parallelism scaling test. Runs N copies of the C engine
(qwen_asr -t1, truly single-threaded incl. Accelerate via VECLIB/OMP=1) on the
5-min Arabic clip in parallel, for N=1..8, and reports aggregate realtime x.

Each process transcribes 300s of audio. With N in parallel finishing in `wall`
seconds, the box processed N*300 audio-seconds in `wall` -> aggregate = N*300/wall.
Linear scaling => aggregate = N * single-process rate; bandwidth saturation =>
aggregate plateaus.
"""
import subprocess, time, os, sys

BIN = "/Users/t/PyCharmProjects/mrecord/qwen-asr-fork/qwen_asr"
MODEL = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b"
CLIP = "/Users/t/PyCharmProjects/mrecord/qwen-asr-fork/ane_proto/aja_5min.wav"
AUDIO_S = 300.0

env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1"}
cmd = [BIN, "-t", "1", "-S", "30", "--past-text", "no", "-d", MODEL, "-i", CLIP, "--silent"]

print(f"\nCPU process-parallelism scaling (qwen_asr -t1, 5-min Arabic, decode bandwidth-bound)")
print(f"  {'N':>2} {'wall(s)':>8} {'per-proc x':>11} {'AGGREGATE x':>12} {'scale-eff':>10}")
base = None
for N in range(1, 9):
    t0 = time.perf_counter()
    procs = [subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
             for _ in range(N)]
    for p in procs:
        p.wait()
    wall = time.perf_counter() - t0
    per = AUDIO_S / wall
    agg = N * AUDIO_S / wall
    if base is None:
        base = per
    eff = agg / (N * base) * 100.0
    print(f"  {N:>2} {wall:8.1f} {per:11.2f} {agg:12.1f} {eff:9.0f}%", flush=True)
