#!/usr/bin/env python3
"""Stateful KV-cache decoder step (CoreML States, macOS15+). The KV cache lives
on-device as model State instead of being passed as input each step, eliminating
the ~350 MB/step host I/O that dominated the previous (stateless) benchmark.

One autoregressive step (seq_q=1) through all 28 layers + tied lm_head, with
per-layer k/v cache as State. Random weights (per-step latency is data-
independent). Measures CPU / GPU / ANE per-step latency and projects 1074-token
decode. Baseline: C engine single-thread = ~43 ms/token (46.2s).
"""
import sys, time, statistics
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct

H, L_LAYERS, NH, NKV, HD = 1024, 28, 16, 8, 128
QD, KVD, INTER, VOCAB = NH*HD, NKV*HD, 3072, 151936
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 512        # max KV-cache length


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class DecLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Parameter(torch.ones(H)); self.n2 = nn.Parameter(torch.ones(H))
        self.q = nn.Linear(H, QD, bias=False); self.k = nn.Linear(H, KVD, bias=False)
        self.v = nn.Linear(H, KVD, bias=False); self.o = nn.Linear(QD, H, bias=False)
        self.qn = nn.Parameter(torch.ones(HD)); self.kn = nn.Parameter(torch.ones(HD))
        self.gate = nn.Linear(H, INTER, bias=False); self.up = nn.Linear(H, INTER, bias=False)
        self.down = nn.Linear(INTER, H, bias=False)
        self.scale = HD ** -0.5

    def forward(self, x, kc, vc):                       # x:[1,1,H]; kc,vc: State buffers
        h = rms(x, self.n1)
        q = self.q(h).view(1, 1, NH, HD).transpose(1, 2)        # [1,NH,1,HD]
        k = self.k(h).view(1, 1, NKV, HD).transpose(1, 2)       # [1,NKV,1,HD]
        v = self.v(h).view(1, 1, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        # write new k/v into the resident cache (fixed slot 0 — latency is slot-independent)
        kc[:, :, 0:1, :] = k
        vc[:, :, 0:1, :] = v
        rep = NH // NKV
        kfull = kc.repeat_interleave(rep, dim=1)                # [1,NH,CTX,HD]
        vfull = vc.repeat_interleave(rep, dim=1)
        att = torch.softmax(torch.matmul(q, kfull.transpose(-1, -2)) * self.scale, dim=-1)
        o = torch.matmul(att, vfull).transpose(1, 2).reshape(1, 1, QD)
        x = x + self.o(o)
        h = rms(x, self.n2)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class DecoderStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecLayer() for _ in range(L_LAYERS)])
        for i in range(L_LAYERS):           # flat top-level buffer names (no dots) = States
            self.register_buffer(f"kc_{i}", torch.zeros(1, NKV, CTX, HD))
            self.register_buffer(f"vc_{i}", torch.zeros(1, NKV, CTX, HD))
        self.nf = nn.Parameter(torch.ones(H))
        self.lm = nn.Linear(H, VOCAB, bias=False)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"))
        return self.lm(rms(x, self.nf))


def main():
    torch.manual_seed(0)
    m = DecoderStep().eval()
    x = torch.randn(1, 1, H)
    with torch.no_grad():
        traced = torch.jit.trace(m, (x,))
    states = []
    for i in range(L_LAYERS):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, CTX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, CTX, HD)), name=f"vc_{i}"))
    print(f"\nstateful decoder step: hidden={H} layers={L_LAYERS} GQA {NH}/{NKV}x{HD} "
          f"vocab={VOCAB} ctx={CTX}  (KV cache = on-device State)")
    print(f"C-engine single-thread baseline: ~43 ms/token (46,167ms / 1074 tok)\n")
    print(f"  {'backend':14s} {'ms/step':>9s} {'1074-tok decode':>16s}")
    for label, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                      ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
                      ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        try:
            ml = ct.convert(traced, inputs=[ct.TensorType(name="x", shape=x.shape, dtype=np.float32)],
                            states=states, compute_precision=ct.precision.FLOAT16,
                            compute_units=cu, minimum_deployment_target=ct.target.macOS15)
            st = ml.make_state()
            feed = {"x": x.numpy()}
            for _ in range(5): ml.predict(feed, state=st)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter(); ml.predict(feed, state=st); ts.append((time.perf_counter()-t0)*1000)
            med = statistics.median(ts)
            print(f"  {label:14s} {med:9.2f} {med*1074/1000:14.2f} s", flush=True)
        except Exception as e:
            print(f"  {label:14s} FAILED: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
