#!/usr/bin/env python3
"""GPU FAST PATH (--gpu): fully GPU-optimised Qwen3-ASR decoder.

Key difference vs gpu_transcribe.py: the KV cache lives ON-DEVICE as CoreML
State (not passed in/out each step), eliminating the ~357 MB/step host I/O that
throttled the correctness-first runner. Writes the new token's K/V at a runtime
position via a one-hot MASKED write (arithmetic, no dynamic-index op) so it
converts cleanly and stays static-shape.

Same loop as before: per-chunk prefill (one stateful pass per prompt token) ->
greedy autoregressive decode -> EOS{151643,151645} -> byte-level BPE detokenize.
Independent chunks (--past-text no contract). Validates first token == C engine.

Usage: python gpu_fast.py [--gpu|--cpu] [max_new]
"""
import sys, json, struct, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import coremltools as ct

MODELDIR = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b"
SAFET = f"{MODELDIR}/model.safetensors"
P = "thinker.model."
H, NL, NH, NKV, HD, INTER, VOCAB = 1024, 28, 16, 8, 128, 3072, 151936
QD, KVD = NH*HD, NKV*HD
EPS, THETA, MAX = 1e-6, 1e6, 512
EOS = {151643, 151645}

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

    def forward(self, x, kc, vc, cos, sin, wmask, amask):
        # x:[1,1,H]; kc,vc:[1,NKV,MAX,HD] (State); cos/sin:[1,1,1,HD];
        # wmask:[1,1,MAX,1] one-hot write pos; amask:[1,1,1,MAX] causal/validity
        h = rms(x, self.iln)
        q = F.linear(h, self.wq).view(1, 1, NH, HD).transpose(1, 2)   # [1,NH,1,HD]
        k = F.linear(h, self.wk).view(1, 1, NKV, HD).transpose(1, 2)  # [1,NKV,1,HD]
        v = F.linear(h, self.wv).view(1, 1, NKV, HD).transpose(1, 2)
        q = rms(q, self.qn); k = rms(k, self.kn)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # masked write of new k/v into the resident cache at the one-hot position
        kc[:] = kc * (1.0 - wmask) + k * wmask        # k[1,NKV,1,HD]*wmask[1,1,MAX,1]->[1,NKV,MAX,HD]
        vc[:] = vc * (1.0 - wmask) + v * wmask
        rep = NH // NKV
        kc_e = kc.repeat_interleave(rep, dim=1)        # [1,NH,MAX,HD]
        vc_e = vc.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kc_e.transpose(-1, -2)) * self.scale + amask   # [1,NH,1,MAX]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vc_e).transpose(1, 2).reshape(1, 1, QD)
        x = x + F.linear(o, self.wo)
        h = rms(x, self.pln)
        x = x + F.linear(F.silu(F.linear(h, self.gate)) * F.linear(h, self.up), self.down)
        return x


class Step(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.layers = nn.ModuleList([Layer(w, i) for i in range(NL)])
        for i in range(NL):
            self.register_buffer(f"kc_{i}", torch.zeros(1, NKV, MAX, HD))
            self.register_buffer(f"vc_{i}", torch.zeros(1, NKV, MAX, HD))
        self.norm = w[f"{P}norm.weight"]

    def forward(self, x, cos, sin, wmask, amask):
        for i, layer in enumerate(self.layers):
            x = layer(x, getattr(self, f"kc_{i}"), getattr(self, f"vc_{i}"), cos, sin, wmask, amask)
        return rms(x, self.norm)[:, 0, :]              # [1,H]


def rope_at(pos):
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half) * 2.0 / HD))
    ang = pos * inv
    c = np.concatenate([np.cos(ang), np.cos(ang)]).astype(np.float32)
    s = np.concatenate([np.sin(ang), np.sin(ang)]).astype(np.float32)
    return c.reshape(1, 1, 1, HD), s.reshape(1, 1, 1, HD)


def byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    dev = "gpu"; a = sys.argv[1:]
    if a and a[0] in ("--gpu", "--cpu"):
        dev = a.pop(0)[2:]
    MAX_NEW = int(a[0]) if a else 200
    with open("decref.demb.bin", "rb") as f:
        seq, dim, start = struct.unpack("<iii", f.read(12))
        emb = np.frombuffer(f.read(), np.float32).reshape(seq, dim).copy()
    with open("decref.dtok.bin", "rb") as f:
        c_tok0 = struct.unpack("<i", f.read(4))[0]

    print(f"\nGPU FAST decoder  device={dev}  on-device KV State  prefill seq={seq}  max_new={MAX_NEW}")
    w = load_w()
    embed = w[f"{P}embed_tokens.weight"].numpy()
    id2tok = {v: k for k, v in json.load(open(f"{MODELDIR}/vocab.json")).items()}
    bd = byte_decoder()

    step = Step(w).eval()
    ex = (torch.zeros(1, 1, H), torch.zeros(1, 1, 1, HD), torch.zeros(1, 1, 1, HD),
          torch.zeros(1, 1, MAX, 1), torch.zeros(1, 1, 1, MAX))
    with torch.no_grad():
        traced = torch.jit.trace(step, ex)
    states = []
    for i in range(NL):
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"kc_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NKV, MAX, HD)), name=f"vc_{i}"))
    names = ["x", "cos", "sin", "wmask", "amask"]
    units = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}[dev]
    t_c = time.perf_counter()
    ml = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape, dtype=np.float32) for n, t in zip(names, ex)],
                    states=states, compute_precision=ct.precision.FLOAT16, compute_units=units,
                    minimum_deployment_target=ct.target.macOS15)
    print(f"convert+compile: {time.perf_counter()-t_c:.1f}s", flush=True)
    st = ml.make_state()

    def run(emb_row, pos):
        cos, sin = rope_at(pos)
        wmask = (np.arange(MAX) == pos).astype(np.float32).reshape(1, 1, MAX, 1)
        amask = np.where(np.arange(MAX) <= pos, 0.0, -1e9).astype(np.float32).reshape(1, 1, 1, MAX)
        out = ml.predict({"x": emb_row.reshape(1, 1, H).astype(np.float32),
                          "cos": cos, "sin": sin, "wmask": wmask, "amask": amask}, state=st)
        return list(out.values())[0].reshape(H)

    # prefill (one stateful pass per prompt token; cache stays on-device)
    t0 = time.perf_counter()
    last_hid = None
    for pos in range(seq):
        last_hid = run(emb[pos], pos)
    prefill_s = time.perf_counter() - t0

    # greedy generate
    toks = []; pos = seq; g0 = time.perf_counter(); cur = last_hid
    for _ in range(MAX_NEW):
        t = int(np.argmax(cur @ embed.T))
        if t in EOS:
            break
        toks.append(t)
        cur = run(embed[t], pos); pos += 1
    gen_s = time.perf_counter() - g0

    first_tok = int(np.argmax(last_hid @ embed.T))
    s = "".join(id2tok.get(t, "") for t in toks)
    text = bytes(bd.get(ch, 63) for ch in s).decode("utf-8", "replace")
    print(f"first token = {first_tok} (C ref {c_tok0}) {'MATCH' if first_tok==c_tok0 else 'DIFF'}")
    print(f"prefill {seq} tok in {prefill_s:.2f}s ({seq/prefill_s:.0f} tok/s); "
          f"gen {len(toks)} tok in {gen_s:.2f}s ({len(toks)/max(gen_s,1e-9):.1f} tok/s, {gen_s/max(len(toks),1)*1000:.1f} ms/tok)")
    print(f"\n--- transcript ({dev}) ---\n{text}\n")


if __name__ == "__main__":
    main()
