#!/usr/bin/env python3
"""Working autoregressive GPU transcriber (decoder side) for Qwen3-ASR-0.6B.

Reuses the C engine's front-end: takes the real decoder input_embeds for one
chunk (captured via QWEN_DUMP_DEC), then runs the FULL autoregressive generation
on the GPU via CoreML fp16 — prefill + greedy token-by-token with an incremental
KV cache — and decodes to text with the byte-level BPE tokenizer.

Design (per the agreed --gpu contract): independent chunks (--past-text no),
greedy decode (do_sample=false), stop on EOS {151643, 151645}. --past-text yes
is rejected upstream (GPU mode decodes chunks independently for parallelism/quality).

KV cache is a fixed [L,NKV,MAX,HD] buffer passed in/out; the new token's own K/V
is folded into attention via an appended self-score (no dynamic-index op), so the
model is fully static-shape and converts cleanly for the GPU.

Usage: python gpu_transcribe.py [--gpu|--cpu] [max_new]
"""
import sys, json, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct

MODELDIR = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b"
SAFET = f"{MODELDIR}/model.safetensors"
P = "thinker.model."
H, NL, NH, NKV, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
QD, KVD = NH*HD, NKV*HD
EPS, THETA = 1e-6, 1e6
MAX = 512
EOS = {151643, 151645}

dev = "gpu"
args = sys.argv[1:]
if args and args[0] in ("--gpu", "--cpu"):
    dev = args.pop(0)[2:]
MAX_NEW = int(args[0]) if args else 220


def load_w():
    with open(SAFET, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = f.tell()
        out = {}
        for k, m in hdr.items():
            if k == "__metadata__" or "audio_tower" in k or m.get("dtype") != "BF16":
                continue
            s, e = m["data_offsets"]; f.seek(base + s)
            raw = np.frombuffer(f.read(e - s), np.uint16)
            out[k] = torch.from_numpy(((raw.astype(np.uint32) << 16).view(np.float32)).reshape(m["shape"]).copy())
    return out


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rotate_half(x):
    return torch.cat([-x[..., HD//2:], x[..., :HD//2]], -1)


class Layer(nn.Module):
    def __init__(self, w, i):
        super().__init__()
        p = f"{P}layers.{i}."
        self.iln = w[p+"input_layernorm.weight"]; self.pln = w[p+"post_attention_layernorm.weight"]
        self.wq = w[p+"self_attn.q_proj.weight"]; self.wk = w[p+"self_attn.k_proj.weight"]
        self.wv = w[p+"self_attn.v_proj.weight"]; self.wo = w[p+"self_attn.o_proj.weight"]
        self.qn = w[p+"self_attn.q_norm.weight"]; self.kn = w[p+"self_attn.k_norm.weight"]
        self.gate = w[p+"mlp.gate_proj.weight"]; self.up = w[p+"mlp.up_proj.weight"]
        self.down = w[p+"mlp.down_proj.weight"]
        self.scale = HD ** -0.5

    def forward(self, x, kc, vc, cos, sin, kmask):
        # x:[1,1,H]  kc,vc:[NKV,MAX,HD]  cos,sin:[1,HD]  kmask:[1,MAX]
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(1, NH, HD)        # [1,NH,HD]
        k = F.linear(h, self.wk).view(1, NKV, HD)
        v = F.linear(h, self.wv).view(1, NKV, HD)
        q = rms(q, self.qn); k = rms(k, self.kn)
        q = q * cos.view(1, 1, HD) + rotate_half(q) * sin.view(1, 1, HD)
        k = k * cos.view(1, 1, HD) + rotate_half(k) * sin.view(1, 1, HD)
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=0)         # [NH,MAX,HD]
        vc_e = vc.repeat_interleave(rep, dim=0)
        k_e = k.repeat_interleave(rep, dim=1)           # [1,NH,HD]
        v_e = v.repeat_interleave(rep, dim=1)
        qh = q.transpose(0, 1)                          # [NH,1,HD]
        past = torch.bmm(qh, kc_e.transpose(1, 2)) * self.scale   # [NH,1,MAX]
        past = past + kmask.view(1, 1, MAX)
        self_s = (qh * k_e.transpose(0, 1)).sum(-1, keepdim=True) * self.scale  # [NH,1,1]
        scores = torch.cat([past, self_s], dim=-1)      # [NH,1,MAX+1]
        p = torch.softmax(scores, dim=-1)
        o = torch.bmm(p[..., :MAX], vc_e) + p[..., MAX:] * v_e.transpose(0, 1)  # [NH,1,HD]
        o = o.transpose(0, 1).reshape(1, 1, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x, k, v                                  # return new k,v (pre-expand) for cache


class Step(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.layers = nn.ModuleList([Layer(w, i) for i in range(NL)])
        self.norm = w[f"{P}norm.weight"]

    def forward(self, emb, kc, vc, cos, sin, kmask):    # kc,vc:[L,NKV,MAX,HD]
        x = emb
        nks, nvs = [], []
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, kc[i], vc[i], cos, sin, kmask)
            nks.append(nk); nvs.append(nv)
        x = rms(x, self.norm)[:, 0, :]                  # [1,H] (only position)
        newk = torch.stack(nks, 0).view(NL, NKV, HD)    # [L,NKV,HD]
        newv = torch.stack(nvs, 0).view(NL, NKV, HD)
        return x, newk, newv


def rope_at(pos):
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half) * 2.0 / HD))
    ang = pos * inv
    cos = np.concatenate([np.cos(ang), np.cos(ang)]).astype(np.float32)
    sin = np.concatenate([np.sin(ang), np.sin(ang)]).astype(np.float32)
    return cos[None, :], sin[None, :]


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    with open("decref.demb.bin", "rb") as f:
        seq, dim, start = struct.unpack("<iii", f.read(12))
        emb = np.frombuffer(f.read(), np.float32).reshape(seq, dim).copy()
    with open("decref.dtok.bin", "rb") as f:
        c_tok0 = struct.unpack("<i", f.read(4))[0]

    print(f"\nGPU transcriber  device={dev}  chunk prefill seq={seq}  max_new={MAX_NEW}")
    w = load_w()
    embed = w[f"{P}embed_tokens.weight"].numpy()         # [VOCAB,H] (= lm_head, tied)
    id2tok = {v: k for k, v in json.load(open(f"{MODELDIR}/vocab.json")).items()}
    bd = byte_decoder()

    step = Step(w).eval()
    ex = (torch.zeros(1, 1, H), torch.zeros(NL, NKV, MAX, HD), torch.zeros(NL, NKV, MAX, HD),
          torch.zeros(1, HD), torch.zeros(1, HD), torch.zeros(1, MAX))
    with torch.no_grad():
        traced = torch.jit.trace(step, ex)
    units = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}[dev]
    names = ["emb", "kc", "vc", "cos", "sin", "kmask"]
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, ex)],
                    compute_precision=ct.precision.FLOAT16, compute_units=units,
                    minimum_deployment_target=ct.target.macOS15)

    kc = np.zeros((NL, NKV, MAX, HD), np.float32); vc = np.zeros_like(kc)
    def predict(emb_row, pos):
        cos, sin = rope_at(pos)
        kmask = np.where(np.arange(MAX) < pos, 0.0, -1e9).astype(np.float32)[None, :]
        return ml.predict({"emb": emb_row.reshape(1, 1, H).astype(np.float32),
                           "kc": kc, "vc": vc, "cos": cos, "sin": sin, "kmask": kmask})

    out_names = None
    t0 = time.perf_counter()
    # prefill: feed each input_embed row
    last_hid = None
    for pos in range(seq):
        o = predict(emb[pos], pos)
        if out_names is None:
            out_names = {k: v.shape for k, v in o.items()}
        hid = o[[k for k, s in out_names.items() if np.prod(s) == H][0]]
        nk = o[[k for k, s in out_names.items() if np.prod(s) == NL*NKV*HD][0]]
        nv = o[[k for k, s in out_names.items() if np.prod(s) == NL*NKV*HD][1]]
        kc[:, :, pos, :] = nk.reshape(NL, NKV, HD); vc[:, :, pos, :] = nv.reshape(NL, NKV, HD)
        last_hid = hid.reshape(H)
    prefill_s = time.perf_counter() - t0

    # generate
    toks = []
    kH = [k for k, s in out_names.items() if np.prod(s) == H][0]
    kKs = [k for k, s in out_names.items() if np.prod(s) == NL*NKV*HD]
    pos = seq
    g0 = time.perf_counter()
    cur_hid = last_hid
    for _ in range(MAX_NEW):
        logits = cur_hid @ embed.T                       # [VOCAB]
        t = int(np.argmax(logits))
        if t in EOS:
            break
        toks.append(t)
        o = predict(embed[t], pos)
        cur_hid = o[kH].reshape(H)
        kc[:, :, pos, :] = o[kKs[0]].reshape(NL, NKV, HD)
        vc[:, :, pos, :] = o[kKs[1]].reshape(NL, NKV, HD)
        pos += 1
    gen_s = time.perf_counter() - g0

    # first predicted token (from prefill's last hidden) for fidelity vs C
    first_tok = int(np.argmax(last_hid @ embed.T))
    s = "".join(id2tok.get(t, "") for t in toks)
    text = bytes(bd.get(ch, 63) for ch in s).decode("utf-8", "replace")
    print(f"first token = {first_tok}  (C ref = {c_tok0}) {'MATCH' if first_tok==c_tok0 else 'DIFF'}")
    print(f"prefill {seq} tok in {prefill_s:.2f}s; generated {len(toks)} tok in {gen_s:.2f}s "
          f"({len(toks)/max(gen_s,1e-9):.1f} tok/s, {gen_s/max(len(toks),1)*1000:.1f} ms/tok)")
    print(f"\n--- transcript ({dev}) ---\n{text}\n")


if __name__ == "__main__":
    main()
