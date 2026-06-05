/*
 * fuzz_safetensors.c - libFuzzer harness for the safetensors reader.
 *
 * safetensors_open() takes a path and mmaps it, so each input is written to a
 * reusable temp file first. We then exercise the parts that do pointer math
 * against attacker-controlled header offsets/sizes: safetensors_data(),
 * safetensors_get_f32() (reads tensor bytes out of the mmap), and
 * safetensors_get_bf16_direct(). A header that lies about an offset/size is the
 * classic out-of-bounds read this is meant to catch.
 *
 * Build/run via the Makefile:  make fuzz && make fuzz-safetensors
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include "qwen_asr_safetensors.h"

static char g_path[] = "/tmp/qwen_fuzz_st_XXXXXX";
static int  g_fd = -1;

static void ensure_tmp(void) {
    if (g_fd < 0) {
        g_fd = mkstemp(g_path);
        if (g_fd < 0) { perror("mkstemp"); abort(); }
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ensure_tmp();
    /* Rewrite the temp file with this input (reuse one fd to keep exec/s up). */
    if (ftruncate(g_fd, 0) != 0) return 0;
    if (lseek(g_fd, 0, SEEK_SET) != 0) return 0;
    if (size > 0 && write(g_fd, data, size) < 0) return 0;

    safetensors_file_t *sf = safetensors_open(g_path);
    if (!sf) return 0;

    for (int i = 0; i < sf->num_tensors && i < SAFETENSORS_MAX_TENSORS; i++) {
        const safetensor_t *t = &sf->tensors[i];
        const void *raw = safetensors_data(sf, t);
        (void)raw;
        if (safetensor_is_bf16(t)) {
            uint16_t *bf = safetensors_get_bf16_direct(sf, t);
            (void)bf;  /* points into the mmap; offset math is the surface */
        } else {
            float *f = safetensors_get_f32(sf, t);  /* copies tensor bytes out */
            if (f) free(f);
        }
    }
    safetensors_close(sf);
    return 0;
}
