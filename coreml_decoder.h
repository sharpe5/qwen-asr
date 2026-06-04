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

/* One decode step.
 *   x:        input hidden state [hidden]         (token embedding)
 *   cos,sin:  NeoX RoPE tables for this position  [head_dim] (duplicated halves)
 *   pos:      absolute position in the cache (0-based; must be < gpu_dec_max())
 *   out:      receives the final-normed hidden     [hidden]
 * Returns 0 on success, non-zero on error. */
int gpu_dec_step(const float *x, const float *cos, const float *sin, int pos, float *out);

int gpu_dec_hidden(void);    /* model hidden size   (1024) */
int gpu_dec_head_dim(void);  /* rope head dim       (128)  */
int gpu_dec_max(void);       /* max cache length    (512)  */

void gpu_dec_free(void);

#ifdef __cplusplus
}
#endif

#endif /* COREML_DECODER_H */
