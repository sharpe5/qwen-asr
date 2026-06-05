/*
 * main.c - CLI entry point for Qwen3-ASR
 *
 * Usage: qwen_asr -d <model_dir> -i <input.wav> [options]
 */

#include "qwen_asr.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_kernels.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QWEN_GPU
#include "coreml_decoder.h"   /* GPU fast path (CoreML); only in the `gpu` build */
#include <libgen.h>           /* dirname */
#include <unistd.h>           /* access  */
#include <limits.h>           /* PATH_MAX */

/* Resolve a bare .mlpackage filename to a path that works regardless of cwd:
 *   --gpu-model-dir override, else the executable's own directory (from argv[0]),
 *   else the bare name (cwd-relative fallback). Writes into out (size n). */
static const char *resolve_gpu_model(char *out, size_t n, const char *prog,
                                     const char *model_dir_override, const char *filename) {
    if (model_dir_override && model_dir_override[0]) {
        snprintf(out, n, "%s/%s", model_dir_override, filename);
        return out;
    }
    char rp[PATH_MAX];
    if (realpath(prog, rp)) {
        char tmp[PATH_MAX];
        snprintf(tmp, sizeof(tmp), "%s", rp);
        char *dir = dirname(tmp);          /* dirname may mutate tmp; use the return value now */
        snprintf(out, n, "%s/%s", dir, filename);
        if (access(out, F_OK) == 0) return out;
    }
    snprintf(out, n, "%s", filename);      /* cwd-relative fallback */
    return out;
}
#endif

/* Token streaming callback: print each piece as it's decoded */
static void stream_token(const char *piece, void *userdata) {
    (void)userdata;
    fputs(piece, stdout);
    fflush(stdout);
}

/* Parse --past-text value.
 * out_mode:  1=yes, 0=no, -1=auto */
static int parse_past_text_mode(const char *s, int *out_mode) {
    char buf[16];
    size_t n = strlen(s);
    if (n == 0 || n >= sizeof(buf)) return -1;
    for (size_t i = 0; i < n; i++) {
        buf[i] = (char)tolower((unsigned char)s[i]);
    }
    buf[n] = '\0';

    if (strcmp(buf, "yes") == 0) {
        *out_mode = 1;
        return 0;
    }
    if (strcmp(buf, "no") == 0) {
        *out_mode = 0;
        return 0;
    }
    if (strcmp(buf, "auto") == 0) {
        *out_mode = -1;
        return 0;
    }
    return -1;
}

static void usage(const char *prog) {
    fprintf(stderr, "qwen_asr — Qwen3-ASR speech-to-text (pure C) [build %s]\n\n", __DATE__);
    fprintf(stderr, "Usage: %s -d <model_dir> (-i <input.wav> | --stdin) [options]\n\n", prog);
    fprintf(stderr, "Required:\n");
    fprintf(stderr, "  -d <dir>      Model directory (with *.safetensors, vocab.json)\n");
    fprintf(stderr, "  -i <file>     Input WAV file (16-bit PCM, any sample rate)\n");
    fprintf(stderr, "  --stdin       Read audio from stdin (auto-detect WAV or raw s16le 16kHz mono)\n");
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -t <n>        Number of threads (default: all CPUs)\n");
    fprintf(stderr, "  -S <secs>     Segment target seconds (default: 0 = full-audio decode)\n");
    fprintf(stderr, "  -W <secs>     Segment-cutting silence search window ± seconds (default: 3.0)\n");
    fprintf(stderr, "  --stream      Streaming mode: process in chunks with prefix rollback\n");
    fprintf(stderr, "  --stream-max-new-tokens <n>  Max generated tokens per stream step (default: 32)\n");
    fprintf(stderr, "  --enc-window-sec <secs>    Encoder attention window in seconds (1..8, default 8)\n");
    fprintf(stderr, "  --past-text <yes|no|auto>  Reuse previously decoded text as context for the next\n");
    fprintf(stderr, "                             segment/chunk (continuity bias; auto=yes for --stream)\n");
    fprintf(stderr, "  --skip-silence              Drop long silent spans before inference (off by default)\n");
    fprintf(stderr, "  --json                      Emit JSON {\"text\":..,\"segments\":[{start,end,text}]} with\n");
    fprintf(stderr, "                              per-segment silence-cut timestamps (suppresses token streaming;\n");
    fprintf(stderr, "                              use WITHOUT --skip-silence for broadcast-aligned times)\n");
    fprintf(stderr, "  --prompt <text>            System prompt for biasing (example: \"Preserve spelling: CPU, CUDA, PostgreSQL, Redis\")\n");
    fprintf(stderr, "  --language <lang>          Force output language via token conditioning\n");
    fprintf(stderr, "                             (usually auto-detected if omitted)\n");
    fprintf(stderr, "  --monitor     Show inline Unicode symbols on stderr (streaming diagnostics)\n");
    fprintf(stderr, "  --debug       Debug output (per-layer details)\n");
    fprintf(stderr, "  --silent      No status output (only final transcription on stdout)\n");
    fprintf(stderr, "                 with -i + --stream, uses non-interactive final refinement\n");
    fprintf(stderr, "  --gpu         Run the decoder on the GPU via CoreML (needs `make gpu`;\n");
    fprintf(stderr, "                segmented, --past-text no only; 0.6B or 1.7B). Auto-loads\n");
    fprintf(stderr, "                qwen_decoder_gpu_<size>_b4_hidden.mlpackage from the exe dir.\n");
    fprintf(stderr, "  --profile <throughput>   GPU preset (with --gpu): lean encode for many\n");
    fprintf(stderr, "                           concurrent processes (enc-threads 1).\n");
    fprintf(stderr, "  --gpu-model-dir <dir>    Directory holding the .mlpackage (default: exe dir)\n");
    fprintf(stderr, "  --gpu-model <path>       Explicit .mlpackage path (overrides the default)\n");
    fprintf(stderr, "  --gpu-refill-min <n>     Batched refill threshold 1..B (default 3)\n");
    fprintf(stderr, "  --gpu-enc-threads <n>    Encode worker threads 1..16 (default 4)\n");
    fprintf(stderr, "  GPU throughput example (run several in parallel, one file each):\n");
    fprintf(stderr, "    %s -d <model_dir> -i file.opus --gpu --profile throughput -t 2 --silent\n", prog);
    fprintf(stderr, "  -h            Show this help\n");
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    const char *input_wav = NULL;
    int verbosity = 1;
    int use_stdin = 0;
    int n_threads = 0; /* 0 = auto-detect */
    float segment_sec = -1; /* -1 = use default (0) */
    float search_sec = -1;  /* -1 = use default (3) */
    int stream_mode = 0;
    int stream_max_new_tokens = -1; /* -1 = use default (32) */
    float enc_window_sec = -1;   /* -1 = use default (8s) */
    const char *prompt_text = NULL;
    const char *force_language = NULL;
    int past_text_conditioning_mode = -1; /* -1 auto, 0 off, 1 on */
    int skip_silence = 0;
    int json_output = 0;
    int emit_tokens = 1;
    int use_gpu = 0;
    const char *profile = NULL;          /* --profile throughput */
    const char *gpu_model_dir = NULL;    /* --gpu-model-dir <dir> */
    const char *gpu_model_flag = NULL;   /* --gpu-model <path> override */
    int gpu_refill_min = 0;              /* --gpu-refill-min (0 = auto) */
    int gpu_enc_threads = 0;             /* --gpu-enc-threads (0 = auto) */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_wav = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-S") == 0 && i + 1 < argc) {
            segment_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-W") == 0 && i + 1 < argc) {
            search_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream_mode = 1;
        } else if (strcmp(argv[i], "--stream-max-new-tokens") == 0 && i + 1 < argc) {
            stream_max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--enc-window-sec") == 0 && i + 1 < argc) {
            enc_window_sec = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--past-text") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (parse_past_text_mode(mode, &past_text_conditioning_mode) != 0) {
                fprintf(stderr, "Error: --past-text must be one of yes|no|auto, got '%s'\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--past-text") == 0) {
            fprintf(stderr, "Error: --past-text requires an argument: yes|no|auto\n");
            return 1;
        } else if (strcmp(argv[i], "--skip-silence") == 0) {
            skip_silence = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = 1;
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            force_language = argv[++i];
        } else if (strcmp(argv[i], "--stdin") == 0) {
            use_stdin = 1;
        } else if (strcmp(argv[i], "--monitor") == 0) {
            qwen_monitor = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            verbosity = 2;
        } else if (strcmp(argv[i], "--silent") == 0) {
            verbosity = 0;
        } else if (strcmp(argv[i], "--gpu") == 0) {
            use_gpu = 1;
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--gpu-model-dir") == 0 && i + 1 < argc) {
            gpu_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--gpu-model") == 0 && i + 1 < argc) {
            gpu_model_flag = argv[++i];
        } else if (strcmp(argv[i], "--gpu-refill-min") == 0 && i + 1 < argc) {
            gpu_refill_min = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu-enc-threads") == 0 && i + 1 < argc) {
            gpu_enc_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir || (!input_wav && !use_stdin)) {
        usage(argv[0]);
        return 1;
    }
    if (enc_window_sec >= 0 && (enc_window_sec < 1.0f || enc_window_sec > 8.0f)) {
        fprintf(stderr, "Error: --enc-window-sec must be in [1, 8], got %.3f\n", enc_window_sec);
        return 1;
    }
    if (stream_max_new_tokens == 0 || stream_max_new_tokens < -1) {
        fprintf(stderr, "Error: --stream-max-new-tokens must be > 0\n");
        return 1;
    }
    if (input_wav && use_stdin) {
        fprintf(stderr, "Error: -i and --stdin are mutually exclusive\n");
        return 1;
    }
    if (profile && strcmp(profile, "throughput") != 0) {
        fprintf(stderr, "Error: --profile must be 'throughput' (got '%s')\n", profile);
        return 1;
    }
    if ((profile || gpu_model_dir || gpu_model_flag || gpu_refill_min || gpu_enc_threads) && !use_gpu) {
        fprintf(stderr, "Error: --profile / --gpu-* options require --gpu\n");
        return 1;
    }

    qwen_verbose = verbosity;
    emit_tokens = (verbosity > 0);
    if (json_output) emit_tokens = 0;  /* clean JSON on stdout: no live token stream */

    /* Initialize thread pool */
    if (n_threads <= 0) n_threads = qwen_get_num_cpus();
    qwen_set_threads(n_threads);

    /* Load model */
    qwen_ctx_t *ctx = qwen_load(model_dir);
    if (!ctx) {
        fprintf(stderr, "Failed to load model from %s\n", model_dir);
        return 1;
    }

    /* Apply segmentation settings */
    if (segment_sec >= 0) ctx->segment_sec = segment_sec;
    if (search_sec >= 0) ctx->search_sec = search_sec;
    if (enc_window_sec >= 0) {
        int window_frames = (int)(enc_window_sec * 100.0f + 0.5f);
        if (window_frames < 100) window_frames = 100;
        if (window_frames > 800) window_frames = 800;
        ctx->config.enc_n_window_infer = window_frames;
    }
    if (stream_max_new_tokens > 0) ctx->stream_max_new_tokens = stream_max_new_tokens;
    if (past_text_conditioning_mode >= 0)
        ctx->past_text_conditioning = past_text_conditioning_mode;
    else if (stream_mode)
        /* Official streaming path uses prefix rollback by default.
         * Keep segmented mode default unchanged (off). */
        ctx->past_text_conditioning = 1;
    if (skip_silence) ctx->skip_silence = 1;
    if (json_output) ctx->emit_json = 1;

    /* GPU fast path: decoder on CoreML/GPU (independent chunks only). */
    if (use_gpu) {
#ifdef QWEN_GPU
        if (ctx->past_text_conditioning == 1) {
            fprintf(stderr, "Error: --gpu does not support --past-text yes "
                            "(GPU decodes chunks independently for parallelism and quality)\n");
            qwen_free(ctx); return 1;
        }
        if (stream_mode) {
            fprintf(stderr, "Error: --gpu is for segmented decode, not --stream\n");
            qwen_free(ctx); return 1;
        }
        if (ctx->segment_sec <= 0.0f) ctx->segment_sec = 28.0f;  /* bounded GPU cache */
        if (ctx->segment_sec > 40.0f) {
            fprintf(stderr, "Error: --gpu requires -S <= 40 (GPU KV cache holds ~512 tokens)\n");
            qwen_free(ctx); return 1;
        }
        /* One batched B-lane model serves all --gpu decode (handles 1..B segments).
         * The default package is chosen to match the engine model size (0.6B
         * hidden=1024, 1.7B hidden=2048); --gpu-model / env overrides win, and a
         * post-load hidden check rejects a mismatched .mlpackage.
         * Path: --gpu-model > env (back-compat) > default resolved next to the binary. */
        char mbuf[PATH_MAX];
        const char *mpath = gpu_model_flag;
        if (!mpath) mpath = getenv("QWEN_GPU_BATCH_MODEL");
        if (!mpath) mpath = getenv("QWEN_GPU_MODEL");
        if (!mpath) {
            const char *tag = (ctx->config.dec_hidden == 1024) ? "0.6b"
                            : (ctx->config.dec_hidden == 2048) ? "1.7b" : NULL;
            if (!tag) {
                fprintf(stderr, "Error: --gpu has no package for engine hidden=%d "
                                "(supported: 0.6B hidden=1024, 1.7B hidden=2048)\n",
                                ctx->config.dec_hidden);
                qwen_free(ctx); return 1;
            }
            char fname[64];
            snprintf(fname, sizeof(fname), "qwen_decoder_gpu_%s_b4_hidden.mlpackage", tag);
            mpath = resolve_gpu_model(mbuf, sizeof(mbuf), argv[0], gpu_model_dir, fname);
        } else if (!strchr(mpath, '/')) {  /* bare filename from flag/env -> resolve like the default */
            mpath = resolve_gpu_model(mbuf, sizeof(mbuf), argv[0], gpu_model_dir, mpath);
        }
        if (gpu_decb_init(mpath) != 0) {
            fprintf(stderr, "Error: failed to load GPU model %s\n"
                            "  (run `make gpu` to generate it, or pass --gpu-model <path>)\n", mpath);
            qwen_free(ctx); return 1;
        }
        if (gpu_dec_hidden() != ctx->config.dec_hidden) {
            fprintf(stderr, "Error: --gpu model %s has hidden=%d but the engine model "
                            "is hidden=%d (wrong-size .mlpackage)\n",
                            mpath, gpu_dec_hidden(), ctx->config.dec_hidden);
            qwen_free(ctx); return 1;
        }
        ctx->config.use_gpu = 1;
        /* --profile throughput => lean encode for many concurrent processes.
         * gpu_refill_min/enc_threads stay 0 (auto: refill 3, enc 4) unless set. */
        if (profile && strcmp(profile, "throughput") == 0 && gpu_enc_threads == 0)
            gpu_enc_threads = 1;
        ctx->config.gpu_refill_min = gpu_refill_min;
        ctx->config.gpu_enc_threads = gpu_enc_threads;
        if (qwen_verbose > 0)
            fprintf(stderr, "GPU decode enabled (CoreML, batched B=%d): %s\n",
                    gpu_decb_batch(), mpath);
#else
        fprintf(stderr, "Error: --gpu not available in this build; rebuild with `make gpu`\n");
        qwen_free(ctx); return 1;
#endif
    }

    if (prompt_text && qwen_set_prompt(ctx, prompt_text) != 0) {
        fprintf(stderr, "Failed to set --prompt text\n");
        qwen_free(ctx);
        return 1;
    }
    if (force_language && qwen_set_force_language(ctx, force_language) != 0) {
        fprintf(stderr, "Unsupported language for --language: %s\n",
                force_language);
        fprintf(stderr, "Supported languages: %s\n", qwen_supported_languages_csv());
        qwen_free(ctx);
        return 1;
    }

    /* Stream tokens to stdout only in non-silent mode.
     * In silent mode we print the final string returned by the API. */
    if (emit_tokens) qwen_set_token_callback(ctx, stream_token, NULL);
    else qwen_set_token_callback(ctx, NULL, NULL);

    /* Transcribe */
    char *text = NULL;
    if (stream_mode && use_stdin) {
        /* Live incremental streaming from stdin */
        qwen_live_audio_t *live = qwen_live_audio_start_stdin();
        if (live) {
            text = qwen_transcribe_stream_live(ctx, live);
            qwen_live_audio_free(live);
        }
    } else if (stream_mode) {
        /* File-based streaming: load audio fully, then stream-transcribe */
        int ns = 0;
        float *samps = qwen_load_wav(input_wav, &ns);
        if (samps) {
            text = qwen_transcribe_stream(ctx, samps, ns);
            free(samps);
        }
    } else if (use_stdin) {
        text = qwen_transcribe_stdin(ctx);
    } else {
        text = qwen_transcribe(ctx, input_wav);
    }

    if (text) {
        if (emit_tokens) printf("\n");
        else printf("%s\n", text);
        free(text);
    } else {
        fprintf(stderr, "Transcription failed\n");
        qwen_free(ctx);
        return 1;
    }

    if (verbosity >= 1) {
        double tokens_per_sec = 0.0;
        if (ctx->perf_total_ms > 0) {
            tokens_per_sec = (1000.0 * (double)ctx->perf_text_tokens) / ctx->perf_total_ms;
        }
        fprintf(stderr,
                "Inference: %.0f ms, %d text tokens (%.2f tok/s, encoding: %.0fms, decoding: %.0fms)\n",
                ctx->perf_total_ms, ctx->perf_text_tokens, tokens_per_sec,
                ctx->perf_encode_ms, ctx->perf_decode_ms);
        if (ctx->perf_audio_ms > 0 && ctx->perf_total_ms > 0) {
            double audio_s = ctx->perf_audio_ms / 1000.0;
            double infer_s = ctx->perf_total_ms / 1000.0;
            double realtime_x = audio_s / infer_s;
            fprintf(stderr, "Audio: %.1f s processed in %.1f s (%.2fx realtime)\n",
                    audio_s, infer_s, realtime_x);
        }
    }

    qwen_free(ctx);
    return 0;
}
