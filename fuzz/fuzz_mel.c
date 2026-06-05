/*
 * fuzz_mel.c - libFuzzer harness for the log-mel front-end.
 *
 * Feeds arbitrary float samples (including NaN/Inf bit patterns and boundary
 * lengths: 0, 1, a few) to qwen_mel_spectrogram() to exercise the windowing /
 * FFT / frame-count math. The input is copied into an aligned, exactly-sized
 * buffer so we test the mel code, not a misaligned read in the harness.
 *
 * Build/run:  make fuzz-mel
 */
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "qwen_asr_audio.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    int n = (int)(size / sizeof(float));
    int frames = 0;

    if (n <= 0) {                 /* boundary: empty input */
        free(qwen_mel_spectrogram(NULL, 0, &frames));
        return 0;
    }

    float *samples = (float *)malloc((size_t)n * sizeof(float));
    if (!samples) return 0;
    memcpy(samples, data, (size_t)n * sizeof(float));

    free(qwen_mel_spectrogram(samples, n, &frames));
    free(samples);
    return 0;
}
