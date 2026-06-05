/*
 * fuzz_wav.c - libFuzzer harness for the WAV/audio buffer parser.
 *
 * Drives qwen_parse_wav_buffer() with arbitrary bytes. Untrusted RIFF/WAV
 * header fields (sizes, channel counts, sample rate, bit depth) flow straight
 * into allocation and resampling math, so this is the prime OOB / integer-
 * overflow surface in qwen_asr_audio.c.
 *
 * Build/run via the Makefile:  make fuzz && make fuzz-wav
 */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "qwen_asr_audio.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    int n_samples = 0;
    float *samples = qwen_parse_wav_buffer(data, size, &n_samples);
    if (samples) free(samples);
    return 0;
}
