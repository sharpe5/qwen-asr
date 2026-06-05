#!/usr/bin/env python3
"""N=1..8 parallel scaling of the gpu_fast decode worker (real-weights stateful
GPU fast path) sharing the ONE 60-core GPU. realtime x = aggregate tok/s / 3.58
(5-min Arabic = 1074 tok / 300s)."""
import subprocess, re, os, sys

PY = sys.executable
WORKER = os.path.join(os.path.dirname(__file__), "gpu_fast_worker.py")
SECS = "12"
TOK_PER_AUDIO_S = 1074 / 300.0

print(f"\nGPU FAST-PATH sharing scaling (N gpu_fast workers on ONE 60-core GPU, {SECS}s each)")
print(f"  {'N':>2} {'per-proc tok/s':>14} {'TOTAL tok/s':>12} {'AGGREGATE x':>12} {'scale-eff':>10}")
base = None
for N in range(1, 9):
    procs = [subprocess.Popen([PY, WORKER, SECS], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
             for _ in range(N)]
    rates = []
    for p in procs:
        out = p.communicate()[0].decode()
        m = re.search(r"TOKS=([\d.]+)", out)
        rates.append(float(m.group(1)) if m else 0.0)
    total = sum(rates); per = total / N
    if base is None:
        base = rates[0] if rates and rates[0] else 1.0
    print(f"  {N:>2} {per:14.1f} {total:12.1f} {total/TOK_PER_AUDIO_S:12.1f} {total/(N*base)*100:9.0f}%", flush=True)
