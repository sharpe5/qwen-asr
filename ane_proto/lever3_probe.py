#!/usr/bin/env python3
"""Lever 3 GO/NO-GO: does BATCHED S=1 decode throughput scale with batch B?

Decode is overhead/under-utilization-bound at B=1 (~9% of the memory roofline),
so batching B independent segments (the --past-text no contract) should amortize
dispatch and fill the GPU -> near-linear tok/s until the bandwidth roofline.
This converts a batched (fixed B, S=1) stateful decoder and times one decode step
for B in {1,2,4,8,12}. If aggregate tok/s rises with B, Lever 3 is GO.

Batched model dims (vs gpu_fast.Step batch-1):
  x[B,1,H] cos/sin[B,1,1,HD] wmask[B,1,MAX,1] amask[B,1,1,MAX]; states [B,NKV,MAX,HD]
"""
import time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct
import gpu_fast as g

H, NL, NH, NKV, HD, INTER = g.H, g.NL, g.NH, g.NKV, g.HD, g.INTER
QD, MAX, EPS = NH * HD, g.MAX, g.EPS
rms, rotate_half = g.rms, g.rotate_half


class BLayer(nn.Module):
    def __init__(self, w, i):
        super().__init__()
        p = f"{g.P}layers.{i}."
        self.iln = w[p+"input_layernorm.weight"]; self.pln = w[p+"post_attention_layernorm.weight"]
        self.wq = w[p+"self_attn.q_proj.weight"]; self.wk = w[p+"self_attn.k_proj.weight"]
        self.wv = w[p+"self_attn.v_proj.weight"]; self.wo = w[p+"self_attn.o_proj.weight"]
        self.qn = w[p+"self_attn.q_norm.weight"]; self.kn = w[p+"self_attn.k_norm.weight"]
        self.gate = w[p+"mlp.gate_proj.weight"]; self.up = w[p+"mlp.up_proj.weight"]
        self.down = w[p+"mlp.down_proj.weight"]
        self.scale = HD ** -0.5

    def forward(self, x, kc, vc, cos, sin, wmask, amask):
        # x:[B,1,H] kc,vc:[B,NKV,MAX,HD] cos/sin:[B,1,1,HD] wmask:[B,1,MAX,1] amask:[B,1,1,MAX]
        B = x.shape[0]
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(B, 1, NH, HD).transpose(1, 2)   # [B,NH,1,HD]
        k = F.linear(h, self.wk).view(B, 1, NKV, HD).transpose(1, 2)  # [B,NKV,1,HD]
        v = F.linear(h, self.wv).view(B, 1, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        kc[:] = kc * (1.0 - wmask) + k * wmask        # [B,NKV,MAX,HD]
        vc[:] = vc * (1.0 - wmask) + v * wmask
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=1)        # [B,NH,MAX,HD]
        vc_e = vc.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kc_e.transpose(-1, -2)) * self.scale + amask   # [B,NH,1,MAX]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vc_e).transpose(1, 2).reshape(B, 1, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x


class BStep(nn.Module):
    def __init__(self, w, B):
        super().__init__()
        self.B = B
        self.layers = nn.ModuleList([BLayer(w, i) for i in range(NL)])
        for i in range(NL):
            self.register_buffer(f"kc_{i}", torch.zeros(B, NKV, MAX, HD))
            self.register_buffer(f"vc_{i}", torch.zeros(B, NKV, MAX, HD))
        self.norm = w[f"{g.P}norm.weight"]

    def forward(self, x, cos, sin, wmask, amask):
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"), cos, sin, wmask, amask)
        return rms(x, self.norm)            # [B,1,H]


def conv(w, B):
    step = BStep(w, B).eval()
    ex = (torch.zeros(B, 1, H), torch.zeros(B, 1, 1, HD), torch.zeros(B, 1, 1, HD),
          torch.zeros(B, 1, MAX, 1), torch.zeros(B, 1, 1, MAX))
    with torch.no_grad():
        tr = torch.jit.trace(step, ex)
    states = []
    for i in range(NL):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(B, NKV, MAX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(B, NKV, MAX, HD)), name=f"vc_{i}"))
    names = ["x", "cos", "sin", "wmask", "amask"]
    ml = ct.convert(tr, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, ex)],
                    states=states, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU,
                    minimum_deployment_target=ct.target.macOS15)
    return ml


def time_step(ml, B, n=200, pos=200):
    st = ml.make_state()
    c, s = g.rope_at(pos)                       # [1,1,1,HD]
    cos = np.repeat(c, B, axis=0).astype(np.float32)  # [B,1,1,HD]
    sin = np.repeat(s, B, axis=0).astype(np.float32)
    wmask = np.zeros((B, 1, MAX, 1), np.float32); wmask[:, 0, pos, 0] = 1.0
    amask = np.where(np.arange(MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, MAX)
    amask = np.repeat(amask, B, axis=0)
    feed = {"x": np.zeros((B, 1, H), np.float32), "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}
    for _ in range(10):
        ml.predict(feed, state=st)
    t0 = time.perf_counter()
    for _ in range(n):
        ml.predict(feed, state=st)
    ms = (time.perf_counter() - t0) / n * 1000
    return ms


def main():
    w = g.load_w()
    print("B | ms/step | tok/s (=B/step) | vs B=1")
    base = None
    for B in [1, 2, 4, 8, 12]:
        t0 = time.perf_counter()
        ml = conv(w, B)
        ct_s = time.perf_counter() - t0
        ms = time_step(ml, B)
        toks = B / ms * 1000
        if base is None:
            base = toks
        print(f"{B:2d} | {ms:7.2f} | {toks:10.1f}     | {toks/base:.2f}x   (convert {ct_s:.0f}s)", flush=True)
    print("\nGO if tok/s rises strongly with B (ideally near-linear); "
          "NO-GO if flat (CoreML serializes the batch).")


if __name__ == "__main__":
    main()
