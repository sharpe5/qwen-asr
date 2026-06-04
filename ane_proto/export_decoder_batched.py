#!/usr/bin/env python3
"""Export the Lever 3 BATCHED decoder: fixed batch B, flexible S (RangeDim
1..MAX), lm_head + argmax ON THE GPU. One model serves batched prefill (S=Lmax,
one call) and batched decode (S=1) for B independent segments (--past-text no).

Output: ../qwen_decoder_gpu_b{B}.mlpackage
Inputs : x[B,S,H] cos[B,S,HD] sin[B,S,HD] wmask[B,MAX,S] amask[B,1,S,MAX]
Output : tok[B,S] int32   (greedy argmax token id per (lane,position))
States : kc_0..kc_27, vc_0..vc_27  [B,NKV,MAX,HD]
"""
import sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct
import gpu_fast as g

H, NL, NH, NKV, HD = g.H, g.NL, g.NH, g.NKV, g.HD
QD, MAX = NH * HD, g.MAX
rms, rotate_half = g.rms, g.rotate_half
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
HIDDEN = (len(sys.argv) > 2 and sys.argv[2] == "hidden")  # output hidden (CPU argmax) vs tok (GPU argmax)


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
        Bb, S = x.shape[0], x.shape[1]
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(Bb, S, NH, HD).transpose(1, 2)
        k = F.linear(h, self.wk).view(Bb, S, NKV, HD).transpose(1, 2)
        v = F.linear(h, self.wv).view(Bb, S, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        c = cos.view(Bb, 1, S, HD); s = sin.view(Bb, 1, S, HD)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        kf = k.permute(0, 2, 1, 3).reshape(Bb, S, NKV * HD)
        vf = v.permute(0, 2, 1, 3).reshape(Bb, S, NKV * HD)
        sk = torch.matmul(wmask, kf).view(Bb, MAX, NKV, HD).permute(0, 2, 1, 3)
        sv = torch.matmul(wmask, vf).view(Bb, MAX, NKV, HD).permute(0, 2, 1, 3)
        kc[:] = kc * (1.0 - rowmask) + sk
        vc[:] = vc * (1.0 - rowmask) + sv
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=1)
        vc_e = vc.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kc_e.transpose(-1, -2)) * self.scale + amask
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
        if not HIDDEN:
            self.embed = w[f"{g.P}embed_tokens.weight"]   # tied lm_head (only for GPU-argmax build)

    def forward(self, x, cos, sin, wmask, amask):
        rowmask = wmask.sum(2).view(B, 1, MAX, 1)
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"),
                      cos, sin, wmask, rowmask, amask)
        hid = rms(x, self.norm)                           # [B,S,H]
        if HIDDEN:
            return hid                                    # CPU argmax in the C engine (throughput)
        logits = F.linear(hid, self.embed)
        return torch.argmax(logits, dim=-1).to(torch.int32)


def main():
    OUT = f"../qwen_decoder_gpu_b{B}{'_hidden' if HIDDEN else ''}.mlpackage"
    w = g.load_w()
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
    out_t = (ct.TensorType(name="hidden", dtype=np.float32) if HIDDEN  # C reads fp32 hidden
             else ct.TensorType(name="tok"))                            # int32 token ids
    ml = ct.convert(tr, inputs=inputs, outputs=[out_t],
                    states=states, compute_precision=ct.precision.FLOAT16,
                    minimum_deployment_target=ct.target.macOS15)
    ml.save(OUT)
    print(f"saved {OUT} ({time.perf_counter()-t0:.0f}s) B={B} flexible S=1..{MAX}, "
          f"{'hidden output (CPU argmax)' if HIDDEN else 'GPU lm_head+argmax'}")


if __name__ == "__main__":
    main()
