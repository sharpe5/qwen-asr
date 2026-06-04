#!/usr/bin/env python3
"""
Fidelity check: faithful PyTorch port of the Qwen3-ASR-0.6B audio encoder,
loaded with the REAL weights, compared against the C engine's f32 encoder
output (the production reference), in two precisions:

  (1) PyTorch f32   -> validates the port is functionally correct vs C
  (2) CoreML  fp16  -> the actual ANE-path fidelity vs C

Reference files produced by the C engine (QWEN_DUMP_ENC=ane_proto/jfk):
  jfk.mel.bin    : [128, mel_frames] f32  (encoder input)
  jfk.encout.bin : [total_tokens, 1024] f32  (encoder output, gold reference)

Architecture replicated exactly from qwen_asr_encoder.c:
  - per-chunk (100 mel frames) Conv2D stem 1->480->480->480 (k3,s2,p1) + tanh-GELU
  - reshape [480,16,T]->[T,7680] (channel-major, freq-minor), conv_out 7680->896 (no bias)
  - per-chunk sinusoidal positional embedding added (positions reset to 0 each chunk)
  - windowed bidirectional attention, window = tokens_per_chunk * (n_window_infer//chunk_size)
  - 18 pre-norm layers (self_attn_layer_norm / final_layer_norm), tanh-GELU FFN
  - ln_post -> proj1 -> tanh-GELU -> proj2 -> [T,1024]
"""
import json, struct, sys, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

MODEL = "/Users/t/PyCharmProjects/qwen-asr/qwen3-asr-0.6b/model.safetensors"
PREFIX = "thinker.audio_tower."
D_MODEL, N_LAYERS, N_HEADS, HEAD_DIM, FFN = 896, 18, 14, 64, 3584
CONV_CH, OUTPUT_DIM, CONV_PROJ = 480, 1024, 7680
CHUNK = 100            # enc_chunk_size = enc_n_window*2
N_WINDOW_INFER = 800

# GELU variant — the C engine uses tanh-approx. Switchable via QWEN_GELU to test
# whether a different formulation is more fp16-stable on the ANE (pure-fp16 graph).
GELU_MODE = os.environ.get("QWEN_GELU", "tanh")

def gelu(x):
    if GELU_MODE == "tanh":
        return F.gelu(x, approximate="tanh")
    if GELU_MODE == "exact":
        return F.gelu(x, approximate="none")           # erf-based
    if GELU_MODE == "sigmoid":
        return x * torch.sigmoid(1.702 * x)            # SiLU-style approx, no x^3
    raise SystemExit(f"unknown QWEN_GELU={GELU_MODE}")


# ----- safetensors loader (bf16 -> f32) -----
def load_weights():
    with open(MODEL, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = f.tell()
        out = {}
        for k, meta in hdr.items():
            if not k.startswith(PREFIX) or meta.get("dtype") != "BF16":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = np.frombuffer(f.read(e - s), dtype=np.uint16)
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)   # bf16 -> f32
            out[k[len(PREFIX):]] = torch.from_numpy(f32.reshape(meta["shape"]).copy())
    return out


def sinusoidal_pe(n_pos, d_model):
    half = d_model // 2
    log_ts = math.log(10000.0) / (half - 1)
    pe = np.zeros((n_pos, d_model), dtype=np.float32)
    d = np.arange(half)
    inv = np.exp(-d * log_ts)
    for p in range(n_pos):
        ang = p * inv
        pe[p, :half] = np.sin(ang)
        pe[p, half:] = np.cos(ang)
    return torch.from_numpy(pe)


def conv_out_len(t):
    return (t - 1) // 2 + 1


class Layer(nn.Module):
    def __init__(self, w, i):
        super().__init__()
        p = f"layers.{i}."
        self.an_w, self.an_b = w[p+"self_attn_layer_norm.weight"], w[p+"self_attn_layer_norm.bias"]
        self.fn_w, self.fn_b = w[p+"final_layer_norm.weight"], w[p+"final_layer_norm.bias"]
        self.qw, self.qb = w[p+"self_attn.q_proj.weight"], w[p+"self_attn.q_proj.bias"]
        self.kw, self.kb = w[p+"self_attn.k_proj.weight"], w[p+"self_attn.k_proj.bias"]
        self.vw, self.vb = w[p+"self_attn.v_proj.weight"], w[p+"self_attn.v_proj.bias"]
        self.ow, self.ob = w[p+"self_attn.out_proj.weight"], w[p+"self_attn.out_proj.bias"]
        self.f1w, self.f1b = w[p+"fc1.weight"], w[p+"fc1.bias"]
        self.f2w, self.f2b = w[p+"fc2.weight"], w[p+"fc2.bias"]
        self.scale = HEAD_DIM ** -0.5

    def forward(self, x, mask):                     # x: [1,S,D], mask: [1,1,S,S] additive
        S = x.shape[1]
        h = F.layer_norm(x, (D_MODEL,), self.an_w, self.an_b, 1e-5)
        q = F.linear(h, self.qw, self.qb).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = F.linear(h, self.kw, self.kb).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = F.linear(h, self.vw, self.vb).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(1, S, D_MODEL)
        x = x + F.linear(o, self.ow, self.ob)
        h = F.layer_norm(x, (D_MODEL,), self.fn_w, self.fn_b, 1e-5)
        x = x + F.linear(gelu(F.linear(h, self.f1w, self.f1b)), self.f2w, self.f2b)
        return x


class Encoder(nn.Module):
    def __init__(self, w, mel_frames):
        super().__init__()
        self.w = w
        self.mel_frames = mel_frames
        self.conv1 = nn.Conv2d(1, CONV_CH, 3, 2, 1); self._set(self.conv1, "conv2d1")
        self.conv2 = nn.Conv2d(CONV_CH, CONV_CH, 3, 2, 1); self._set(self.conv2, "conv2d2")
        self.conv3 = nn.Conv2d(CONV_CH, CONV_CH, 3, 2, 1); self._set(self.conv3, "conv2d3")
        self.layers = nn.ModuleList([Layer(w, i) for i in range(N_LAYERS)])
        # precompute geometry + PE + window mask for this fixed input length
        self.tpc = conv_out_len(conv_out_len(conv_out_len(CHUNK)))   # tokens per full chunk
        n_chunks = (mel_frames + CHUNK - 1) // CHUNK
        toks = []
        for c in range(n_chunks):
            cw = min(CHUNK, mel_frames - c * CHUNK)
            toks.append(conv_out_len(conv_out_len(conv_out_len(cw))))
        self.chunk_tokens = toks
        self.total = sum(toks)
        wsize = self.tpc * (N_WINDOW_INFER // CHUNK)
        mask = torch.full((self.total, self.total), float("-inf"))
        for s in range(0, self.total, wsize):
            e = min(s + wsize, self.total)
            mask[s:e, s:e] = 0.0
        self.register_buffer("mask", mask.view(1, 1, self.total, self.total))
        self.pe_full = sinusoidal_pe(self.tpc, D_MODEL)

    def _set(self, conv, name):
        conv.weight.data = self.w[name+".weight"]; conv.bias.data = self.w[name+".bias"]

    def forward(self, mel):                          # mel: [1,1,128,mel_frames]
        outs = []
        for ci, t in enumerate(self.chunk_tokens):
            s = ci * CHUNK
            cw = min(CHUNK, self.mel_frames - s)
            chunk = mel[:, :, :, s:s + cw]
            x = gelu(self.conv1(chunk))
            x = gelu(self.conv2(x))
            x = gelu(self.conv3(x))        # [1,480,16,t]
            x = x.permute(0, 3, 1, 2).reshape(1, t, CONV_PROJ)   # channel-major, freq-minor
            x = F.linear(x, self.w["conv_out.weight"])           # no bias
            x = x + self.pe_full[:t].unsqueeze(0)
            outs.append(x)
        x = torch.cat(outs, dim=1)                   # [1, total, 896]
        for layer in self.layers:
            x = layer(x, self.mask)
        x = F.layer_norm(x, (D_MODEL,), self.w["ln_post.weight"], self.w["ln_post.bias"], 1e-5)
        x = gelu(F.linear(x, self.w["proj1.weight"], self.w["proj1.bias"]))
        x = F.linear(x, self.w["proj2.weight"], self.w["proj2.bias"])
        return x                                     # [1, total, 1024]


def read_bin(path):
    with open(path, "rb") as f:
        a, b = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(a, b)
    return a, b, data.copy()


def metrics(name, ref, got):
    ref = ref.reshape(-1); got = got.reshape(-1)
    diff = np.abs(ref - got)
    cos = float(np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    rel = float(np.linalg.norm(ref - got) / np.linalg.norm(ref))
    print(f"  {name:16s} cos={cos:.6f}  rel_l2={rel:.5f}  max|Δ|={diff.max():.5f}  "
          f"mean|Δ|={diff.mean():.6f}  (ref range [{ref.min():.2f},{ref.max():.2f}])")
    return cos, rel


def main():
    pref = sys.argv[1] if len(sys.argv) > 1 else "jfk"
    _, mf, mel = read_bin(f"{pref}.mel.bin")          # [128, mel_frames]
    T, OD, ref = read_bin(f"{pref}.encout.bin")       # [tokens, 1024]
    print(f"\nreference: mel 128x{mf}, encoder out {T}x{OD}")

    w = load_weights()
    enc = Encoder(w, mf).eval()
    print(f"port geometry: tokens/chunk={enc.tpc}, total tokens={enc.total} "
          f"(C ref tokens={T}), window={enc.tpc*(N_WINDOW_INFER//CHUNK)}")
    assert enc.total == T, "token count mismatch — geometry wrong"

    mel_t = torch.from_numpy(mel).view(1, 1, 128, mf)
    with torch.no_grad():
        out_f32 = enc(mel_t).numpy()
    print("\n--- (1) PyTorch f32 vs C f32 (functional correctness of port) ---")
    metrics("pytorch-f32", ref, out_f32)

    print("\n--- (2) CoreML fp16 (ANE path) vs C f32 (production fidelity) ---")
    with torch.no_grad():
        traced = torch.jit.trace(enc, mel_t)
    for label, cu in [("fp16 CPU+NE", ct.ComputeUnit.CPU_AND_NE),
                      ("fp16 CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
        ml = ct.convert(traced,
                        inputs=[ct.TensorType(name="mel", shape=mel_t.shape, dtype=np.float32)],
                        compute_precision=ct.precision.FLOAT16,
                        compute_units=cu,
                        minimum_deployment_target=ct.target.macOS15)
        got = ml.predict({"mel": mel_t.numpy()})
        got = list(got.values())[0]
        metrics(label, ref, got)


if __name__ == "__main__":
    main()
