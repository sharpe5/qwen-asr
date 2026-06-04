/*
 * coreml_decoder.h - C API for the GPU (CoreML) decoder fast path.
 *
 * Drives the exported stateful Qwen3-ASR decoder (qwen_decoder_gpu.mlpackage)
 * on the GPU (CPU_AND_GPU) from the pure-C engine. The KV cache lives on-device
 * as CoreML State; each call decodes one token.
 *
 * The C engine keeps the front-end (mel + encoder + prompt assembly) and the
 * back-end (lm_head argmax + tokenizer); this bridge only replaces the
 * autoregressive transformer step (qwen_decoder_forward).
 *
 * Implemented in coreml_decoder.mm (Objective-C++); only compiled/linked on
 * macOS with the `gpu` build target.
 */
#ifndef COREML_DECODER_H
#define COREML_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Load + compile the .mlpackage and configure CPU_AND_GPU. Returns 0 on success. */
int gpu_dec_init(const char *mlpackage_path);

/* Start a fresh, zeroed KV cache (one independent chunk). Returns 0 on success. */
int gpu_dec_reset(void);

/* Process S tokens in ONE call (S=N for batched prefill, S=1 for a decode step).
 *   emb:       [S * hidden]     input embeddings, row-major
 *   cos,sin:   [S * head_dim]   NeoX RoPE tables per token (duplicated halves)
 *   positions: [S]              absolute cache position of each token (< gpu_dec_max())
 *   S:         token count (1 .. gpu_dec_max())
 *   out:       [S * hidden]     final-normed hidden per token (may be NULL to discard)
 * Returns 0 on success, non-zero on error. */
int gpu_dec_chunk(const float *emb, const float *cos, const float *sin,
                  const int *positions, int S, float *out);

int gpu_dec_hidden(void);    /* model hidden size   (1024) */
int gpu_dec_head_dim(void);  /* rope head dim       (128)  */
int gpu_dec_max(void);       /* max cache length    (512)  */

void gpu_dec_free(void);

/* ========================================================================
 * Lever 3: BATCHED decoder (B independent segments per forward) with the
 * lm_head + argmax fused on the GPU (returns greedy token ids, not hidden).
 * One model serves batched prefill (S=Lmax, one call) and batched decode
 * (S=1). B is fixed at export and queried via gpu_decb_batch().
 * ======================================================================== */

/* Load + compile the batched .mlpackage (CPU_AND_GPU). Returns 0 on success. */
int gpu_decb_init(const char *mlpackage_path);
int gpu_decb_ready(void);    /* 1 if a batched model is loaded */
int gpu_decb_batch(void);    /* fixed batch size B baked into the model */
int gpu_decb_reset(void);    /* fresh zeroed KV cache for all B lanes */

/* Batched prefill: lane b feeds emb[b, 0..lens[b]-1] at positions 0..lens[b]-1.
 *   emb:        [B * Lmax * hidden]  per-lane prompt embeds, padded to Lmax (row-major)
 *   lens:       [B]                  real prompt length per lane (1..Lmax)
 *   write_lane: [B] or NULL          1=prefill this lane, 0=PRESERVE its KV (refill of
 *                                    freed lanes mid-stream); NULL = all lanes prefill.
 *   Lmax:       max prompt length over written lanes (1..gpu_dec_max())
 *   cos,sin:    [Lmax * head_dim]    RoPE for positions 0..Lmax-1 (shared across lanes)
 *   out_tok:    [B]                  greedy token at each written lane's last real position
 * Returns 0 on success. */
int gpu_decb_prefill(const float *emb, const int *lens, const int *write_lane, int Lmax,
                     const float *cos, const float *sin, int *out_tok);

/* Batched decode step (S=1): lane b feeds emb[b] at write position positions[b].
 *   emb:       [B * hidden]    embedding of each lane's current token
 *   positions: [B]             cache write position per lane (< gpu_dec_max())
 *   cos,sin:   [B * head_dim]  per-lane RoPE at positions[b]
 *   out_tok:   [B]             greedy next token per lane
 * Returns 0 on success. */
int gpu_decb_step(const float *emb, const int *positions,
                  const float *cos, const float *sin, int *out_tok);

/* 1 if the loaded batched model outputs "hidden" (lm_head/argmax done on the CPU,
 * for throughput) instead of "tok" (argmax fused on the GPU, for single-stream). */
int gpu_decb_is_hidden(void);

/* Hidden-output variants: same as prefill/step but return the final-normed hidden
 * vector per lane (out_hidden = [B * hidden]) — the C engine does the lm_head argmax
 * on the (idle, at high concurrency) CPU. Only valid when gpu_decb_is_hidden().
 *   prefill_h: out_hidden[b] = hidden at lane b's last real position (lens[b]-1)
 *   step_h:    out_hidden[b] = hidden at the single decoded position */
int gpu_decb_prefill_h(const float *emb, const int *lens, const int *write_lane, int Lmax,
                       const float *cos, const float *sin, float *out_hidden);
int gpu_decb_step_h(const float *emb, const int *positions,
                    const float *cos, const float *sin, float *out_hidden);

void gpu_decb_free(void);

#ifdef __cplusplus
}
#endif

#endif /* COREML_DECODER_H */
