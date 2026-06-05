# experiment_ane — Apple Neural Engine research (archived)

**Conclusion: we never got the ANE working as a viable path.** This directory is
the research archive that reached that conclusion. The accelerated path that
actually shipped runs the decoder on the **Metal GPU** (CoreML, `CPU_AND_GPU`),
not the Neural Engine. The encoder stays pure C for both model sizes.

If you're looking for the tooling that builds the GPU `.mlpackage` files the
engine loads, it has moved to **`../coreml_export/`** and is driven by
`make gpu`. This directory is kept for history only.

## Why the ANE didn't pan out

Two independent problems, both measured here, killed it:

1. **It was slow for our workload.** Decoding is autoregressive — one token at a
   time (`seq_q = 1`), 28 sequential layers per step. The ANE is built for large,
   batched matmuls; on tiny per-step dispatch its latency lost to CPU+GPU. The
   per-step benchmarks (`decoder_step_bench.py`, `decoder_stateful_bench.py`,
   comparing `CPU_ONLY` / `CPU_AND_GPU` / `CPU_AND_NE`) showed the GPU winning.

2. **fp16 precision killed quality.** The ANE runs fp16. On the audio encoder
   that degraded output enough to matter, and the only way to recover fidelity
   was to promote selected ops back to fp32 — which fall to the **CPU** under
   `CPU_AND_NE`, so the supposed ANE win evaporated. The "ane_hp" (ANE
   high-precision) experiments chased this and confirmed the trade-off was not
   worth it: `encoder_ane_tune.py`, `ane_residency.py`, `gelu_experiment.py`,
   `encoder_fidelity.py`.

Net: the ANE was neither fast enough (single-token decode) nor accurate enough
(fp16 encoder) to beat the GPU, so the shipping `--gpu` path uses
`MLComputeUnitsCPUAndGPU` and deliberately *leaves the ANE free*
(see `../coreml_decoder.mm`).

## What's in here (all archival)

- **ANE feasibility / precision:** `encoder_ane_bench.py`, `encoder_ane_tune.py`,
  `ane_residency.py`, `gelu_experiment.py`, `encoder_fidelity.py`,
  `make_encout.py`, `make_fp16_encout.py`, `seg_latency.py`.
- **GPU benchmarks / scaling / probes (predate the shipped path):**
  `decoder_step_bench.py`, `decoder_stateful_bench.py`, `gpu_decoder.py`,
  `gpu_transcribe.py`, `gpu_worker.py`, `gpu_scaling_test.py`,
  `gpu_fast_worker.py`, `gpu_fast_scaling.py`, `gpu_cpu_probe.py`,
  `lever2_probe.py`, `lever3_probe.py`, `lever3_validate.py`,
  `scaling_test.py`, `throughput_long.py`, `throughput_sweep.py`.

## Caveat

These scripts are **not maintained**. Many hardcode absolute paths from the
original author's machine and some import `gpu_fast`, which now lives in
`../coreml_export/`. Expect to fix paths/imports before any of them run. For the
supported, working flow, use `../coreml_export/` via `make gpu`.
