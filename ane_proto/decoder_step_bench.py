#!/usr/bin/env python3
"""Feasibility: per-decode-step latency of the Qwen3-ASR-0.6B *decoder* on
CPU / GPU / ANE. One autoregressive step (seq_q=1) through all 28 layers with a
fixed-length KV cache, plus the big tied lm_head (1024 -> 151936). Random weights
(per-step latency is data-independent: it's dominated by streaming ~0.6B fp16
weights from memory + the lm_head matmul + per-call dispatch).

Decoder dims (config thinker.text_config):
  hidden=1024, layers=28, heads=16, kv_heads=8 (GQA), head_dim=128,
  q_dim=2048, kv_dim=1024, intermediate=3072 (SwiGLU), vocab=151936, RoPE theta=1e6.

Baseline to beat: C engine single-thread decode = 46,167 ms / 1074 tok = ~43 ms/token.
"""
import sys, time, statistics
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct

H, L_LAYERS, NH, NKV, HD = 1024, 28, 16, 8, 128
QD, KVD, INTER, VOCAB = NH*HD, NKV*HD, 3072, 151936
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 390     # KV-cache length (one chunk)


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

    def forward(self, x, kc, vc):                 # x:[1,1,H]  kc,vc:[1,NKV,CTX,HD]
        h = rms(x, self.n1)
        q = self.q(h).view(1, 1, NH, HD).transpose(1, 2)         # [1,NH,1,HD]
        k = self.k(h).view(1, 1, NKV, HD).transpose(1, 2)        # [1,NKV,1,HD]
        v = self.v(h).view(1, 1, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        # append new k/v to cache, expand KV for GQA
        kfull = torch.cat([kc, k], dim=2)                        # [1,NKV,CTX+1,HD]
        vfull = torch.cat([vc, v], dim=2)
        rep = NH // NKV
        kfull = kfull.repeat_interleave(rep, dim=1)              # [1,NH,CTX+1,HD]
        vfull = vfull.repeat_interleave(rep, dim=1)
        att = torch.matmul(q, kfull.transpose(-1, -2)) * self.scale
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, vfull).transpose(1, 2).reshape(1, 1, QD)
        x = x + self.o(o)
        h = rms(x, self.n2)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class DecoderStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecLayer() for _ in range(L_LAYERS)])
        self.nf = nn.Parameter(torch.ones(H))
        self.lm = nn.Linear(H, VOCAB, bias=False)        # tied embed/lm_head

    def forward(self, x, kc, vc):                        # kc,vc:[L,1,NKV,CTX,HD]
        for i, layer in enumerate(self.layers):
            x = layer(x, kc[i], vc[i])
        x = rms(x, self.nf)
        return self.lm(x)                                # [1,1,VOCAB]


def main():
    torch.manual_seed(0)
    m = DecoderStep().eval()
    x = torch.randn(1, 1, H)
    kc = torch.randn(L_LAYERS, 1, NKV, CTX, HD)
    vc = torch.randn(L_LAYERS, 1, NKV, CTX, HD)
    with torch.no_grad():
        traced = torch.jit.trace(m, (x, kc, vc))
    print(f"\ndecoder step: hidden={H} layers={L_LAYERS} GQA {NH}/{NKV}x{HD} "
          f"inter={INTER} vocab={VOCAB} ctx={CTX}")
    print(f"C-engine single-thread baseline: ~43 ms/token (46,167ms / 1074 tok)\n")
    print(f"  {'backend':18s} {'ms/step':>9s} {'1074-tok decode':>16s}")
    inp = [ct.TensorType(name=n, shape=t.shape, dtype=np.float32)
           for n, t in (("x", x), ("kc", kc), ("vc", vc))]
    for label, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
                      ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
                      ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
        try:
            ml = ct.convert(traced, inputs=inp, compute_precision=ct.precision.FLOAT16,
                            compute_units=cu, minimum_deployment_target=ct.target.macOS15)
            feed = {"x": x.numpy(), "kc": kc.numpy(), "vc": vc.numpy()}
            for _ in range(5): ml.predict(feed)
            ts = []
            for _ in range(20):
                t0 = time.perf_counter(); ml.predict(feed); ts.append((time.perf_counter()-t0)*1000)
            med = statistics.median(ts)
            print(f"  {label:18s} {med:9.2f} {med*1074/1000:14.2f} s", flush=True)
        except Exception as e:
            print(f"  {label:18s} FAILED: {str(e)[:80]}", flush=True)


if __name__ == "__main__":
    main()
