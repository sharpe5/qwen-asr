#!/usr/bin/env python3
"""GPU sharing/contention scaling test. Launches N parallel gpu_worker.py
processes (each a stateful GPU decode loop) for N=1..8, all sharing the single
60-core GPU, and reports aggregate decode throughput -> realtime x.

realtime x = aggregate tok/s / 3.58 (the 5-min Arabic clip = 1074 tok / 300s).
Linear scaling => GPU was underused and parallelism fills it; plateau => the
GPU (memory bandwidth) is the shared bottleneck."""
import subprocess, re, os, sys

PY = sys.executable
WORKER = os.path.join(os.path.dirname(__file__), "gpu_worker.py")
SECS = "15"
TOK_PER_AUDIO_S = 1074 / 300.0

print(f"\nGPU sharing scaling (N stateful decode workers on ONE 60-core GPU, {SECS}s each)")
print(f"  {'N':>2} {'per-proc tok/s':>14} {'TOTAL tok/s':>12} {'AGGREGATE x':>12} {'scale-eff':>10}")
base = None
for N in range(1, 9):
    procs = [subprocess.Popen([PY, WORKER, SECS], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
             for _ in range(N)]
    rates = []
    for p in procs:
        out = p.communicate()[0].decode()
        mobj = re.search(r"TOKS=([\d.]+)", out)
        rates.append(float(mobj.group(1)) if mobj else 0.0)
    total = sum(rates)
    per = total / N
    if base is None:
        base = rates[0] if rates else 1.0
    eff = total / (N * base) * 100.0
    print(f"  {N:>2} {per:14.1f} {total:12.1f} {total/TOK_PER_AUDIO_S:12.1f} {eff:9.0f}%", flush=True)
