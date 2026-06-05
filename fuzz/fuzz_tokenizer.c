/*
 * fuzz_tokenizer.c - libFuzzer harness for the BPE tokenizer.
 *
 * Loads the real tokenizer once (vocab.json), then fuzzes:
 *   - qwen_tokenizer_encode() with arbitrary text — exercises the byte-level
 *     UTF-8 decode + BPE merge logic on adversarial / truncated multibyte input;
 *   - qwen_tokenizer_decode() with an arbitrary token id — checks the id is
 *     bounds-checked against vocab_size (an unchecked id would read OOB).
 *
 * Vocab path: $QWEN_VOCAB or qwen3-asr-0.6b/vocab.json. If it can't load, the
 * harness no-ops (so the build/run still works without the model).
 *
 * Build/run:  make fuzz-tokenizer
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "qwen_asr_tokenizer.h"

static qwen_tokenizer_t *g_tok = NULL;

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    const char *p = getenv("QWEN_VOCAB");
    if (!p) p = "qwen3-asr-0.6b/vocab.json";
    g_tok = qwen_tokenizer_load(p);
    if (!g_tok)
        fprintf(stderr, "fuzz_tokenizer: could not load tokenizer from %s "
                        "(encode/decode fuzzing disabled)\n", p);
    return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!g_tok) return 0;

    char *text = (char *)malloc(size + 1);
    if (!text) return 0;
    if (size) memcpy(text, data, size);
    text[size] = '\0';

    int n = 0;
    int *ids = qwen_tokenizer_encode(g_tok, text, &n);
    free(ids);

    /* Decode with an attacker-controlled id (incl. negative / huge). */
    if (size >= sizeof(int)) {
        int id;
        memcpy(&id, data, sizeof(int));
        (void)qwen_tokenizer_decode(g_tok, id);
    }

    free(text);
    return 0;
}
