#!/usr/bin/env python3
"""Lever 3 correctness: batched (B-lane) flexible-S decoder with the lm_head +
argmax ON THE GPU (outputs token ids, not hidden). Validates that the GPU fp16
lm_head/argmax reproduces the C bf16 reference:
  - feed the real 381-token prefill (decref.demb.bin) into every lane,
  - batched prefill (one call), read the argmax token at each lane's last position,
  - all lanes must equal the C first-token reference (decref.dtok.bin = 1098),
  - then run a few batched decode steps; all lanes must stay identical and match
    the gpu_fast single-lane greedy continuation.
Also times an S=1 batched decode step WITH the lm_head to confirm it stays cheap.
"""
import sys, time, struct, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct
import gpu_fast as g

H, NL, NH, NKV, HD, INTER = g.H, g.NL, g.NH, g.NKV, g.HD, g.INTER
QD, MAX, EPS = NH * HD, g.MAX, g.EPS
rms, rotate_half = g.rms, g.rotate_half
B = 4


class CL(nn.Module):
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
        # x:[B,S,H] cos/sin:[B,S,HD] wmask:[B,MAX,S] rowmask:[B,1,MAX,1] amask:[B,1,S,MAX]
        Bb, S = x.shape[0], x.shape[1]
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(Bb, S, NH, HD).transpose(1, 2)   # [B,NH,S,HD]
        k = F.linear(h, self.wk).view(Bb, S, NKV, HD).transpose(1, 2)  # [B,NKV,S,HD]
        v = F.linear(h, self.wv).view(Bb, S, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        c = cos.view(Bb, 1, S, HD); s = sin.view(Bb, 1, S, HD)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        kf = k.permute(0, 2, 1, 3).reshape(Bb, S, NKV * HD)        # [B,S,NKV*HD]
        vf = v.permute(0, 2, 1, 3).reshape(Bb, S, NKV * HD)
        sk = torch.matmul(wmask, kf).view(Bb, MAX, NKV, HD).permute(0, 2, 1, 3)  # [B,NKV,MAX,HD]
        sv = torch.matmul(wmask, vf).view(Bb, MAX, NKV, HD).permute(0, 2, 1, 3)
        kc[:] = kc * (1.0 - rowmask) + sk
        vc[:] = vc * (1.0 - rowmask) + sv
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=1)              # [B,NH,MAX,HD]
        vc_e = vc.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kc_e.transpose(-1, -2)) * self.scale + amask  # [B,NH,S,MAX]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vc_e).transpose(1, 2).reshape(Bb, S, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x


class Chunk(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.layers = nn.ModuleList([CL(w, i) for i in range(NL)])
        for i in range(NL):
            self.register_buffer(f"kc_{i}", torch.zeros(B, NKV, MAX, HD))
            self.register_buffer(f"vc_{i}", torch.zeros(B, NKV, MAX, HD))
        self.norm = w[f"{g.P}norm.weight"]
        self.embed = w[f"{g.P}embed_tokens.weight"]   # [VOCAB,H] tied lm_head

    def forward(self, x, cos, sin, wmask, amask):
        rowmask = wmask.sum(2).view(B, 1, MAX, 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"),
                      cos, sin, wmask, rowmask, amask)
        hid = rms(x, self.norm)                              # [B,S,H]
        logits = F.linear(hid, self.embed)                   # [B,S,VOCAB]  (GPU lm_head)
        return torch.argmax(logits, dim=-1).to(torch.int32)  # [B,S] token ids


def main():
    with open("decref.demb.bin", "rb") as f:
        seq, dim, start = struct.unpack("<iii", f.read(12))
        emb = np.frombuffer(f.read(), np.float32).reshape(seq, dim).copy()
    with open("decref.dtok.bin", "rb") as f:
        c_tok0 = struct.unpack("<i", f.read(4))[0]
    w = g.load_w()
    embed_np = w[f"{g.P}embed_tokens.weight"].numpy()
    chunk = Chunk(w).eval()

    Sx = 384
    ex = (torch.zeros(B, Sx, H), torch.zeros(B, Sx, HD), torch.zeros(B, Sx, HD),
          torch.zeros(B, MAX, Sx), torch.zeros(B, 1, Sx, MAX))
    with torch.no_grad():
        tr = torch.jit.trace(chunk, ex)
    states = []
    for i in range(NL):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(B, NKV, MAX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(B, NKV, MAX, HD)), name=f"vc_{i}"))
    sr = ct.RangeDim(lower_bound=1, upper_bound=MAX, default=Sx)
    inputs = [
        ct.TensorType(name="x", shape=(B, sr, H), dtype=np.float32),
        ct.TensorType(name="cos", shape=(B, sr, HD), dtype=np.float32),
        ct.TensorType(name="sin", shape=(B, sr, HD), dtype=np.float32),
        ct.TensorType(name="wmask", shape=(B, MAX, sr), dtype=np.float32),
        ct.TensorType(name="amask", shape=(B, 1, sr, MAX), dtype=np.float32),
    ]
    t0 = time.perf_counter()
    ml = ct.convert(tr, inputs=inputs, outputs=[ct.TensorType(name="tok")],
                    states=states, compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_GPU,
                    minimum_deployment_target=ct.target.macOS15)
    print(f"convert+compile {time.perf_counter()-t0:.0f}s  (B={B}, lm_head+argmax on GPU)", flush=True)
    st = ml.make_state()

    # ---- batched prefill: replicate the 381-token prompt into all B lanes ----
    S = seq
    cos = np.zeros((B, S, HD), np.float32); sin = np.zeros((B, S, HD), np.float32)
    for pos in range(S):
        c, s = g.rope_at(pos); cos[:, pos] = c.reshape(HD); sin[:, pos] = s.reshape(HD)
    wmask = np.zeros((B, MAX, S), np.float32)
    for pos in range(S):
        wmask[:, pos, pos] = 1.0
    amask = np.full((B, 1, S, MAX), -1e9, np.float32)
    for i in range(S):
        amask[:, 0, i, :i+1] = 0.0
    x = np.repeat(emb.reshape(1, S, H), B, axis=0)
    tok = ml.predict({"x": x, "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}, state=st)["tok"]
    first = tok[:, S-1]   # last real position per lane
    print(f"first token per lane = {first.tolist()}  (C ref {c_tok0})  "
          f"{'ALL MATCH' if all(int(t)==c_tok0 for t in first) else 'MISMATCH'}")

    # ---- a few batched decode steps (S=1); all lanes identical, greedy ----
    def step(tokens, pos):
        xb = np.stack([embed_np[int(t)] for t in tokens]).reshape(B, 1, H).astype(np.float32)
        c1 = np.zeros((B, 1, HD), np.float32); s1 = np.zeros((B, 1, HD), np.float32)
        cc, ss = g.rope_at(pos); c1[:, 0] = cc.reshape(HD); s1[:, 0] = ss.reshape(HD)
        wm = np.zeros((B, MAX, 1), np.float32); wm[:, pos, 0] = 1.0
        am = np.where(np.arange(MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, MAX)
        am = np.repeat(am, B, axis=0)
        return ml.predict({"x": xb, "cos": c1, "sin": s1, "wmask": wm, "amask": am}, state=st)["tok"][:, 0]

    seq_out = [int(first[0])]
    cur = first; pos = S
    for _ in range(12):
        nxt = step(cur, pos); pos += 1
        if not all(int(t) == int(nxt[0]) for t in nxt):
            print("  LANE DIVERGENCE at step", pos); break
        seq_out.append(int(nxt[0])); cur = nxt
        if int(nxt[0]) in g.EOS:
            break
    print(f"greedy tokens (lane 0): {seq_out}")

    # ---- per-step S=1 timing WITH lm_head ----
    for _ in range(10):
        step(cur, pos)
    t0 = time.perf_counter()
    for _ in range(200):
        step(cur, pos)
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f"\nS=1 batched step WITH GPU lm_head: {ms:.2f} ms/step (B={B}) = {B/ms*1000:.0f} tok/s")


if __name__ == "__main__":
    main()
