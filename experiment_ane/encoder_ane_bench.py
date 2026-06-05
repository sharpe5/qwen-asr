#!/usr/bin/env python3
"""
ANE feasibility prototype for the Qwen3-ASR-0.6B *audio encoder*.

Builds a faithful PyTorch replica of the encoder compute graph at the exact
config dims (NOT loading real weights — this is a latency/feasibility benchmark,
random init is fine because matmul/conv timing is data-independent), exports it
to CoreML fp16, and times it on CPU-only vs CPU+ANE vs ALL (CPU+GPU+ANE).

Baseline to beat: the pure-C engine reported `encoding: 236ms` for the 11s JFK
clip on this machine.

Encoder dims (from config.json thinker_config.audio_config):
  num_mel_bins=128, d_model=896, encoder_layers=18, encoder_attention_heads=14
  (head_dim=64), encoder_ffn_dim=3584, downsample_hidden_size=480, output_dim=1024
Conv stem (from safetensors shapes): conv2d1 [480,1,3,3], conv2d2/3 [480,480,3,3],
  conv_out [896,7680]. Three stride-2 convs: freq 128->16, time/8. reshape 480*16=7680 -> 896.
Audio framing: 16kHz, hop 160 -> 100 mel frames/sec.
"""
import sys, time, statistics
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

D_MODEL = 896
N_LAYERS = 18
N_HEADS = 14
HEAD_DIM = D_MODEL // N_HEADS          # 64
FFN = 3584
CONV_CH = 480
N_MEL = 128
OUTPUT_DIM = 1024
CONV_PROJ = CONV_CH * 16               # 7680


def _conv_out(t):                              # stride2, pad1, k3
    return (t - 1) // 2 + 1


class EncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder layer (full bidirectional attention).
    FLOP profile matches the C encoder's per-layer work; window attention would be
    cheaper, so this is a conservative (upper-bound) latency estimate at short seq.
    Seq length S is baked in (fixed input) to avoid dynamic-shape ops that CoreML
    conversion can't fold."""
    def __init__(self, S):
        super().__init__()
        self.S = S
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, FFN)
        self.fc2 = nn.Linear(FFN, D_MODEL)
        self.scale = HEAD_DIM ** -0.5

    def forward(self, x):                       # x: [1, S, D]
        S = self.S
        h = self.ln1(x)
        q = self.q(h).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(h).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v(h).view(1, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(1, S, D_MODEL)
        x = x + self.o(out)
        h = self.ln2(x)
        x = x + self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, seq_mel):
        super().__init__()
        self.S = _conv_out(_conv_out(_conv_out(seq_mel)))   # token count after 3 convs
        self.conv1 = nn.Conv2d(1, CONV_CH, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(CONV_CH, CONV_CH, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(CONV_CH, CONV_CH, 3, stride=2, padding=1)
        self.conv_out = nn.Linear(CONV_PROJ, D_MODEL, bias=False)
        self.layers = nn.ModuleList([EncoderLayer(self.S) for _ in range(N_LAYERS)])
        self.ln_post = nn.LayerNorm(D_MODEL)
        self.proj_out = nn.Linear(D_MODEL, OUTPUT_DIM, bias=False)

    def forward(self, mel):                     # mel: [1, 1, 128, T]
        x = torch.nn.functional.gelu(self.conv1(mel))    # [1,480,64,T/2]
        x = torch.nn.functional.gelu(self.conv2(x))      # [1,480,32,T/4]
        x = torch.nn.functional.gelu(self.conv3(x))      # [1,480,16,T/8]
        # [1,480,16,S] -> [1,S,480*16]; permute then reshape with literal dims
        x = x.permute(0, 3, 1, 2).reshape(1, self.S, CONV_PROJ)
        x = self.conv_out(x)                             # [1, S, 896]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_post(x)
        return self.proj_out(x)                          # [1, S, 1024]


def build_mlmodel(seq_mel, compute_units):
    torch.manual_seed(0)
    model = AudioEncoder(seq_mel).eval()
    ex = torch.randn(1, 1, N_MEL, seq_mel)
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=ex.shape, dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=compute_units,
        minimum_deployment_target=ct.target.macOS15,
    )
    return mlmodel, ex.numpy()


def bench(mlmodel, mel_np, iters=30, warmup=5):
    inp = {"mel": mel_np}
    for _ in range(warmup):
        mlmodel.predict(inp)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mlmodel.predict(inp)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    audio_s = float(sys.argv[1]) if len(sys.argv) > 1 else 11.0
    seq_mel = int(round(audio_s * 100))          # 100 mel frames / sec
    tokens = seq_mel // 8
    print(f"\n=== Qwen3-ASR-0.6B encoder ANE prototype ===")
    print(f"audio: {audio_s}s -> mel frames: {seq_mel} -> encoder tokens: ~{tokens}")
    print(f"dims: d_model={D_MODEL} layers={N_LAYERS} heads={N_HEADS} ffn={FFN}")
    print(f"C-engine baseline for 11s JFK: encoding=236ms (CPU + Accelerate/AMX)\n")

    units = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL (CPU+GPU+NE)": ct.ComputeUnit.ALL,
    }
    results = {}
    for label, cu in units.items():
        print(f"[building + benchmarking: {label}] ...", flush=True)
        try:
            m, mel = build_mlmodel(seq_mel, cu)
            med, lo, hi = bench(m, mel)
            results[label] = med
            print(f"   {label:18s}  median={med:7.1f} ms   (min {lo:.1f}, max {hi:.1f})\n")
        except Exception as e:
            print(f"   {label}: FAILED -> {e}\n")

    print("=== summary (encoder forward latency, 11s-equivalent input) ===")
    print(f"   {'C engine (Accelerate/AMX)':28s} {236.0:7.1f} ms   [baseline]")
    for label, med in results.items():
        delta = 236.0 / med if med else 0
        print(f"   {'CoreML '+label:28s} {med:7.1f} ms   ({delta:.2f}x vs baseline)")


if __name__ == "__main__":
    main()
