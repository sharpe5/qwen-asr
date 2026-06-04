#!/usr/bin/env python3
"""Standalone CoreML decoder runner with a --gpu flag, + single-step numerical
fidelity validation against the C engine.

Faithful PyTorch port of the Qwen3-ASR-0.6B text decoder (real weights), run as
a full causal forward over a real prefill sequence captured from the C engine
(QWEN_DUMP_DEC). Validates that the decoder's next-token prediction (final-normed
hidden + lm_head argmax) matches the C decoder within fp16 tolerance, and reports
per-device latency.

Usage:
  python gpu_decoder.py            # validate: PyTorch-f32 vs C, and CoreML fp16 (GPU + CPU)
  python gpu_decoder.py --gpu      # CoreML fp16 on GPU only
  python gpu_decoder.py --cpu      # CoreML fp16 on CPU only

Decoder: hidden 1024, 28 layers, GQA 16/8 x128, q/k per-head RMSNorm, NeoX RoPE
theta=1e6, SwiGLU 3072, RMSNorm eps 1e-6, tied lm_head (= embed_tokens), vocab 151936.
"""
import sys, json, struct, time, math, statistics
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct

MODEL = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b/model.safetensors"
P = "thinker.model."
H, NL, NH, NKV, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
QD, KVD = NH*HD, NKV*HD
EPS, THETA = 1e-6, 1e6


def load_dec_weights():
    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = f.tell()
        out = {}
        want = lambda k: (k.startswith(P) or k == "thinker.lm_head.weight") and k != "__metadata__"
        for k, meta in hdr.items():
            if not want(k) or meta.get("dtype") != "BF16":
                continue
            s, e = meta["data_offsets"]; f.seek(base + s)
            raw = np.frombuffer(f.read(e - s), dtype=np.uint16)
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            out[k] = torch.from_numpy(f32.reshape(meta["shape"]).copy())
    return out


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope_tables(seq, start_pos):
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half) * 2.0 / HD))
    pos = np.arange(start_pos, start_pos + seq)[:, None]
    ang = pos * inv[None, :]                       # [seq, half]
    cos = np.concatenate([np.cos(ang), np.cos(ang)], -1)   # duplicate across halves
    sin = np.concatenate([np.sin(ang), np.sin(ang)], -1)
    return torch.tensor(cos, dtype=torch.float32), torch.tensor(sin, dtype=torch.float32)


def rotate_half(x):
    half = HD // 2                                 # baked constant (head_dim fixed)
    return torch.cat([-x[..., half:], x[..., :half]], -1)


class Layer(nn.Module):
    def __init__(self, w, i, seq):
        super().__init__()
        self.S = seq
        p = f"{P}layers.{i}."
        self.iln = w[p+"input_layernorm.weight"]; self.pln = w[p+"post_attention_layernorm.weight"]
        self.wq = w[p+"self_attn.q_proj.weight"]; self.wk = w[p+"self_attn.k_proj.weight"]
        self.wv = w[p+"self_attn.v_proj.weight"]; self.wo = w[p+"self_attn.o_proj.weight"]
        self.qn = w[p+"self_attn.q_norm.weight"]; self.kn = w[p+"self_attn.k_norm.weight"]
        self.gate = w[p+"mlp.gate_proj.weight"]; self.up = w[p+"mlp.up_proj.weight"]
        self.down = w[p+"mlp.down_proj.weight"]
        self.scale = HD ** -0.5

    def forward(self, x, cos, sin, mask):              # x:[1,S,H]
        S = self.S                                     # baked constant (fixed seq)
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(1, S, NH, HD).transpose(1, 2)    # [1,NH,S,HD]
        k = F.linear(h, self.wk).view(1, S, NKV, HD).transpose(1, 2)
        v = F.linear(h, self.wv).view(1, S, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)       # per-head RMSNorm (before RoPE)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        rep = NH // NKV
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(1, S, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x


class Decoder(nn.Module):
    def __init__(self, w, seq, start_pos):
        super().__init__()
        self.seq = seq
        self.layers = nn.ModuleList([Layer(w, i, seq) for i in range(NL)])
        self.norm = w[f"{P}norm.weight"]
        self.lm = w[f"{P}embed_tokens.weight"]         # C uses tied embed_tokens as lm_head
        cos, sin = rope_tables(seq, start_pos)
        self.register_buffer("cos", cos.view(1, 1, seq, HD))
        self.register_buffer("sin", sin.view(1, 1, seq, HD))
        m = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
        self.register_buffer("mask", m.view(1, 1, seq, seq))

    def forward(self, emb):                            # emb:[1,S,H]
        x = emb
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, self.mask)
        x = rms(x, self.norm)                          # [1,S,H]
        last = x[:, self.seq - 1, :]                   # [1,H] last position (fixed index)
        logits = F.linear(last, self.lm)               # [1,VOCAB]
        return last, logits


def read_demb(pref):
    with open(f"{pref}.demb.bin", "rb") as f:
        seq, dim, start = struct.unpack("<iii", f.read(12))
        emb = np.frombuffer(f.read(), np.float32).reshape(seq, dim).copy()
    with open(f"{pref}.dhid.bin", "rb") as f:
        dim = struct.unpack("<i", f.read(4))[0]
        hid = np.frombuffer(f.read(), np.float32).reshape(dim).copy()
    with open(f"{pref}.dtok.bin", "rb") as f:
        tok = struct.unpack("<i", f.read(4))[0]
    return seq, dim, start, emb, hid, tok


def fidelity(tag, ref_hid, ref_tok, hid, logits):
    hid = hid.reshape(-1); rh = ref_hid.reshape(-1)
    cos = float(np.dot(rh, hid)/(np.linalg.norm(rh)*np.linalg.norm(hid)))
    rel = float(np.linalg.norm(rh-hid)/np.linalg.norm(rh))
    tok = int(np.argmax(logits))
    print(f"  {tag:20s} hidden cos={cos:.6f} rel_l2={rel:.5f}  argmax={tok} "
          f"{'== C(' + str(ref_tok) + ') OK' if tok==ref_tok else '!= C('+str(ref_tok)+')'}")


def main():
    pref = "decref"
    want = {"--gpu": ["gpu"], "--cpu": ["cpu"]}.get(sys.argv[1] if len(sys.argv) > 1 else "", ["gpu", "cpu"])
    seq, dim, start, emb, ref_hid, ref_tok = read_demb(pref)
    print(f"\nreal decode-step reference: seq={seq} dim={dim} start_pos={start} C-argmax-token={ref_tok}")
    w = load_dec_weights()
    dec = Decoder(w, seq, start).eval()
    emb_t = torch.from_numpy(emb).view(1, seq, H)

    # (1) PyTorch f32 — port correctness vs C
    with torch.no_grad():
        last, logits = dec(emb_t)
    print("\n--- PyTorch f32 vs C (port correctness) ---")
    fidelity("pytorch-f32", ref_hid, ref_tok, last.numpy(), logits.numpy()[0])

    # (2) CoreML fp16 on requested device(s) — the runnable --gpu path
    with torch.no_grad():
        traced = torch.jit.trace(dec, (emb_t,))
    units = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}
    print("\n--- CoreML fp16 vs C (the --gpu runner path) ---")
    for dev in want:
        ml = ct.convert(traced, inputs=[ct.TensorType(name="emb", shape=emb_t.shape, dtype=np.float32)],
                        compute_precision=ct.precision.FLOAT16, compute_units=units[dev],
                        minimum_deployment_target=ct.target.macOS15)
        feed = {"emb": emb_t.numpy()}
        for _ in range(3): ml.predict(feed)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); out = ml.predict(feed); ts.append((time.perf_counter()-t0)*1000)
        vals = list(out.values())
        # identify outputs by shape
        h_out = next(v for v in vals if v.size == H)
        l_out = next(v for v in vals if v.size == VOCAB)
        fidelity(f"coreml-fp16-{dev}", ref_hid, ref_tok, h_out, l_out.reshape(-1))
        print(f"      prefill latency median={statistics.median(ts):.1f} ms (seq={seq})")


if __name__ == "__main__":
    main()
