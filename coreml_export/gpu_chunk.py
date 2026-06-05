#!/usr/bin/env python3
"""De-risk batched prefill: a single flexible-sequence (S=1..MAX) stateful
decoder that handles BOTH prefill (S=N, one call) and decode (S=1), using a
matmul-scatter write (wmask[MAX,S] @ k) so any write positions work with no
dynamic-index op. One forward reads the weights once for all S tokens.

Test: feed the real 381-token prefill (decref.demb.bin) as ONE chunk, check the
last-row hidden -> lm_head argmax == C reference (1098), and time it vs the
per-token path (gpu_fast: ~350 calls).
"""
import sys, time, json, struct
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct
import gpu_fast as g

H, NL, NH, NKV, HD, INTER = g.H, g.NL, g.NH, g.NKV, g.HD, g.INTER
QD, MAX, EPS = NH*HD, g.MAX, g.EPS
rms, rotate_half = g.rms, g.rotate_half


class ChunkLayer(nn.Module):
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

    def forward(self, x, kc, vc, cos, sin, wmask, rowmask, amask):
        # x:[1,S,H] cos/sin:[1,S,HD] wmask:[MAX,S] rowmask:[1,1,MAX,1] amask:[1,1,S,MAX]
        S = x.shape[1]
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(1, S, NH, HD).transpose(1, 2)   # [1,NH,S,HD]
        k = F.linear(h, self.wk).view(1, S, NKV, HD).transpose(1, 2)  # [1,NKV,S,HD]
        v = F.linear(h, self.wv).view(1, S, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        c = cos.view(1, 1, S, HD); s = sin.view(1, 1, S, HD)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        # scatter the S new keys/values into the resident cache via wmask[MAX,S]
        kf = k[0].permute(1, 0, 2).reshape(S, NKV*HD)        # [S, NKV*HD]
        vf = v[0].permute(1, 0, 2).reshape(S, NKV*HD)
        sk = torch.matmul(wmask, kf).view(MAX, NKV, HD).permute(1, 0, 2).unsqueeze(0)  # [1,NKV,MAX,HD]
        sv = torch.matmul(wmask, vf).view(MAX, NKV, HD).permute(1, 0, 2).unsqueeze(0)
        kc[:] = kc * (1.0 - rowmask) + sk
        vc[:] = vc * (1.0 - rowmask) + sv
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=1)              # [1,NH,MAX,HD]
        vc_e = vc.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kc_e.transpose(-1, -2)) * self.scale + amask  # [1,NH,S,MAX]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vc_e).transpose(1, 2).reshape(1, S, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x


class Chunk(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.layers = nn.ModuleList([ChunkLayer(w, i) for i in range(NL)])
        for i in range(NL):
            self.register_buffer(f"kc_{i}", torch.zeros(1, NKV, MAX, HD))
            self.register_buffer(f"vc_{i}", torch.zeros(1, NKV, MAX, HD))
        self.norm = w[f"{g.P}norm.weight"]

    def forward(self, x, cos, sin, wmask, amask):
        rowmask = wmask.sum(1).view(1, 1, MAX, 1)            # [1,1,MAX,1]
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"),
                      cos, sin, wmask, rowmask, amask)
        return rms(x, self.norm)                              # [1,S,H]


def main():
    with open("decref.demb.bin", "rb") as f:
        seq, dim, start = struct.unpack("<iii", f.read(12))
        emb = np.frombuffer(f.read(), np.float32).reshape(seq, dim).copy()
    with open("decref.dtok.bin", "rb") as f:
        c_tok0 = struct.unpack("<i", f.read(4))[0]
    w = g.load_w()
    embed = w[f"{g.P}embed_tokens.weight"].numpy()
    chunk = Chunk(w).eval()

    # trace with a representative S, mark S flexible
    Sx = 381
    ex = (torch.zeros(1, Sx, H), torch.zeros(1, Sx, HD), torch.zeros(1, Sx, HD),
          torch.zeros(MAX, Sx), torch.zeros(1, 1, Sx, MAX))
    with torch.no_grad():
        traced = torch.jit.trace(chunk, ex)
    states = []
    for i in range(NL):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"vc_{i}"))
    srange = ct.RangeDim(lower_bound=1, upper_bound=MAX, default=Sx)
    inputs = [
        ct.TensorType(name="x",     shape=(1, srange, H),   dtype=np.float32),
        ct.TensorType(name="cos",   shape=(1, srange, HD),  dtype=np.float32),
        ct.TensorType(name="sin",   shape=(1, srange, HD),  dtype=np.float32),
        ct.TensorType(name="wmask", shape=(MAX, srange),    dtype=np.float32),
        ct.TensorType(name="amask", shape=(1, 1, srange, MAX), dtype=np.float32),
    ]
    t0 = time.perf_counter()
    ml = ct.convert(traced, inputs=inputs, states=states, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU, minimum_deployment_target=ct.target.macOS15)
    print(f"convert+compile: {time.perf_counter()-t0:.1f}s (flexible S=1..{MAX})", flush=True)
    st = ml.make_state()

    # ---- batched prefill: one call, S = seq ----
    S = seq
    cos = np.zeros((1, S, HD), np.float32); sin = np.zeros((1, S, HD), np.float32)
    for pos in range(S):
        c, s = g.rope_at(pos); cos[0, pos] = c.reshape(HD); sin[0, pos] = s.reshape(HD)
    wmask = np.zeros((MAX, S), np.float32)
    for pos in range(S):
        wmask[pos, pos] = 1.0                                # prefill positions 0..S-1
    amask = np.full((1, 1, S, MAX), -1e9, np.float32)
    for i in range(S):
        amask[0, 0, i, :i+1] = 0.0                           # causal
    feed = {"x": emb.reshape(1, S, H), "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}
    for _ in range(2):
        ml.predict(feed, state=ml.make_state())              # warmup (fresh state)
    st = ml.make_state()
    tp = time.perf_counter()
    out = ml.predict(feed, state=st)
    prefill_ms = (time.perf_counter() - tp) * 1000
    hid = list(out.values())[0].reshape(S, H)
    last = hid[S-1]
    tok = int(np.argmax(last @ embed.T))
    print(f"batched prefill: S={S} in {prefill_ms:.1f} ms (one call)  -> first token {tok} "
          f"(C ref {c_tok0}) {'MATCH' if tok==c_tok0 else 'DIFF'}")
    print(f"  vs per-token prefill would be ~{S} calls (gpu_fast did {S} x ~11ms = ~{S*11/1000:.1f}s)")


if __name__ == "__main__":
    main()
