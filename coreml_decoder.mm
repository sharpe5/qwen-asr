/*
 * coreml_decoder.mm - Objective-C++ bridge: drives the exported stateful
 * Qwen3-ASR decoder on the GPU (CPU_AND_GPU) for the C engine's --gpu fast path.
 *
 * Model I/O (from coreml_export/export_decoder*.py). Hidden H is read from the
 * loaded model at init (0.6B=1024, 1.7B=2048); head-dim 128 and cache 512 are fixed:
 *   inputs : x[B,S,H] cos[B,S,128] sin[B,S,128] wmask[B,512,S] amask[B,1,S,512]
 *   states : kc_0..kc_27, vc_0..vc_27  (on-device KV cache)
 *   output : hidden[B,S,H]  (or tok[B,S] for the GPU-argmax build)
 */
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_decoder.h"
#include <string.h>

#define HID 1024
#define HDIM 128
#define MAXLEN 512

static MLModel       *g_model = nil;
static MLState       *g_state = nil;          /* on-device KV cache (macOS 15) */
static int            g_HID = HID;            /* hidden read from the loaded model (0.6B=1024, 1.7B=2048) */

/* Reused S=1 inputs for the hot per-token decode path (avoids per-call alloc). */
static MLMultiArray  *g_x1 = nil, *g_cos1 = nil, *g_sin1 = nil, *g_wmask1 = nil, *g_amask1 = nil;
static MLDictionaryFeatureProvider *g_input1 = nil;

static MLMultiArray *mk(NSArray<NSNumber*> *shape) {
    NSError *e = nil;
    MLMultiArray *a = [[MLMultiArray alloc] initWithShape:shape
                                                 dataType:MLMultiArrayDataTypeFloat32
                                                    error:&e];
    if (e) { NSLog(@"[gpu_dec] MLMultiArray alloc failed: %@", e); return nil; }
    return a;
}

int gpu_dec_init(const char *mlpackage_path) {
    @autoreleasepool {
        NSError *err = nil;
        NSURL *pkg = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage_path]];

        /* Compile the .mlpackage to a temporary .mlmodelc */
        NSURL *compiled = [MLModel compileModelAtURL:pkg error:&err];
        if (err || !compiled) { NSLog(@"[gpu_dec] compile failed: %@", err); return 1; }

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndGPU;   /* GPU fast path, leave ANE free */
        g_model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        if (err || !g_model) { NSLog(@"[gpu_dec] load failed: %@", err); return 2; }

        g_state = [g_model newState];   /* macOS 15+ stateful KV cache */
        if (!g_state) { NSLog(@"[gpu_dec] newState failed (needs macOS 15+)"); return 3; }
        return 0;
    }
}

int gpu_dec_reset(void) {
    @autoreleasepool {
        if (!g_model) return 1;
        g_state = [g_model newState];   /* fresh zeroed KV cache for a new chunk */
        return 0;
    }
}

int gpu_dec_chunk(const float *emb, const float *cosv, const float *sinv,
                  const int *positions, int S, float *out) {
    @autoreleasepool {
        if (!g_model || S < 1 || S > MAXLEN) return 1;
        NSError *err = nil;
        MLMultiArray *x, *cosa, *sina, *wmask, *amask;
        MLDictionaryFeatureProvider *input;

        if (S == 1) {
            /* Hot decode path: reuse persistent S=1 arrays + provider. */
            if (!g_x1) {
                g_x1 = mk(@[@1, @1, @HID]); g_cos1 = mk(@[@1, @1, @HDIM]);
                g_sin1 = mk(@[@1, @1, @HDIM]); g_wmask1 = mk(@[@MAXLEN, @1]);
                g_amask1 = mk(@[@1, @1, @1, @MAXLEN]);
                if (!g_x1 || !g_cos1 || !g_sin1 || !g_wmask1 || !g_amask1) return 2;
                g_input1 = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                    @"x":     [MLFeatureValue featureValueWithMultiArray:g_x1],
                    @"cos":   [MLFeatureValue featureValueWithMultiArray:g_cos1],
                    @"sin":   [MLFeatureValue featureValueWithMultiArray:g_sin1],
                    @"wmask": [MLFeatureValue featureValueWithMultiArray:g_wmask1],
                    @"amask": [MLFeatureValue featureValueWithMultiArray:g_amask1],
                } error:&err];
                if (err) { NSLog(@"[gpu_dec] provider failed: %@", err); return 2; }
            }
            x = g_x1; cosa = g_cos1; sina = g_sin1; wmask = g_wmask1; amask = g_amask1;
            input = g_input1;
        } else {
            /* Prefill (once per chunk): variable S, allocate per call. */
            NSNumber *nS = @(S);
            x     = mk(@[@1, nS, @HID]);
            cosa  = mk(@[@1, nS, @HDIM]);
            sina  = mk(@[@1, nS, @HDIM]);
            wmask = mk(@[@MAXLEN, nS]);
            amask = mk(@[@1, @1, nS, @MAXLEN]);
            if (!x || !cosa || !sina || !wmask || !amask) return 2;
            input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"x":     [MLFeatureValue featureValueWithMultiArray:x],
                @"cos":   [MLFeatureValue featureValueWithMultiArray:cosa],
                @"sin":   [MLFeatureValue featureValueWithMultiArray:sina],
                @"wmask": [MLFeatureValue featureValueWithMultiArray:wmask],
                @"amask": [MLFeatureValue featureValueWithMultiArray:amask],
            } error:&err];
            if (err) { NSLog(@"[gpu_dec] provider failed: %@", err); return 4; }
        }

        memcpy((float *)x.dataPointer,    emb,  (size_t)S * HID  * sizeof(float));
        memcpy((float *)cosa.dataPointer, cosv, (size_t)S * HDIM * sizeof(float));
        memcpy((float *)sina.dataPointer, sinv, (size_t)S * HDIM * sizeof(float));

        float *pw = (float *)wmask.dataPointer;   /* [MAXLEN, S] row-major */
        float *pa = (float *)amask.dataPointer;   /* [1,1,S,MAXLEN] row-major */
        memset(pw, 0, (size_t)MAXLEN * S * sizeof(float));
        for (int s = 0; s < S; s++) {
            int pos = positions[s];
            if (pos < 0 || pos >= MAXLEN) return 3;
            pw[(size_t)pos * S + s] = 1.0f;                       /* scatter token s -> cache pos */
            float *row = pa + (size_t)s * MAXLEN;
            for (int j = 0; j < MAXLEN; j++) row[j] = (j <= pos) ? 0.0f : -1e9f;  /* causal */
        }

        id<MLFeatureProvider> result = [g_model predictionFromFeatures:input
                                                            usingState:g_state error:&err];
        if (err || !result) { NSLog(@"[gpu_dec] predict failed: %@", err); return 5; }

        if (out) {
            MLMultiArray *h = [result featureValueForName:@"hidden"].multiArrayValue;
            if (!h) { NSLog(@"[gpu_dec] no 'hidden' output"); return 6; }
            memcpy(out, (float *)h.dataPointer, (size_t)S * HID * sizeof(float));
        }
        return 0;
    }
}

int gpu_dec_hidden(void)   { return g_HID; }
int gpu_dec_head_dim(void) { return HDIM; }
int gpu_dec_max(void)      { return MAXLEN; }

void gpu_dec_free(void) {
    g_model = nil; g_state = nil;
    g_x1 = g_cos1 = g_sin1 = g_wmask1 = g_amask1 = nil; g_input1 = nil;
}

/* ========================================================================
 * Lever 3: batched decoder (B lanes) with GPU lm_head + argmax.
 * ======================================================================== */

static MLModel  *g_modelb = nil;
static MLState  *g_stateb = nil;
static int       g_B = 0;
static int       g_decb_hidden = 0;   /* 1 = model outputs "hidden" (CPU argmax path) */
/* Reused S=1 batched decode inputs (hot path). */
static MLMultiArray *g_bx = nil, *g_bcos = nil, *g_bsin = nil, *g_bwm = nil, *g_bam = nil;
static MLDictionaryFeatureProvider *g_binput = nil;

int gpu_decb_init(const char *mlpackage_path) {
    @autoreleasepool {
        NSError *err = nil;
        NSURL *pkg = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage_path]];
        NSURL *compiled = [MLModel compileModelAtURL:pkg error:&err];
        if (err || !compiled) { NSLog(@"[gpu_decb] compile failed: %@", err); return 1; }
        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsCPUAndGPU;
        g_modelb = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
        if (err || !g_modelb) { NSLog(@"[gpu_decb] load failed: %@", err); return 2; }
        MLFeatureDescription *xd = g_modelb.modelDescription.inputDescriptionsByName[@"x"];
        g_B = (int)[xd.multiArrayConstraint.shape[0] integerValue];   /* batch from input shape */
        if (g_B < 1) { NSLog(@"[gpu_decb] bad batch %d", g_B); return 3; }
        g_HID = (int)[xd.multiArrayConstraint.shape[2] integerValue]; /* hidden from x[B,S,H] */
        if (g_HID < 1) { NSLog(@"[gpu_decb] bad hidden %d", g_HID); return 3; }
        /* "hidden" output => lm_head/argmax happens on the CPU (throughput); else "tok". */
        g_decb_hidden = (g_modelb.modelDescription.outputDescriptionsByName[@"hidden"] != nil);
        g_stateb = [g_modelb newState];
        if (!g_stateb) { NSLog(@"[gpu_decb] newState failed (needs macOS 15+)"); return 4; }
        return 0;
    }
}

int gpu_decb_ready(void) { return g_modelb != nil; }
int gpu_decb_batch(void) { return g_B; }
int gpu_decb_is_hidden(void) { return g_decb_hidden; }

int gpu_decb_reset(void) {
    @autoreleasepool {
        if (!g_modelb) return 1;
        g_stateb = [g_modelb newState];
        return 0;
    }
}

/* Read greedy token ids from the model's int32 "tok" output (defensive about dtype). */
static int read_tok(id<MLFeatureProvider> result, int *dst, int b, int S, int col) {
    MLMultiArray *tok = [result featureValueForName:@"tok"].multiArrayValue;
    if (!tok) { NSLog(@"[gpu_decb] no 'tok' output"); return 1; }
    size_t idx = (size_t)b * S + col;
    if (tok.dataType == MLMultiArrayDataTypeInt32) {
        *dst = ((int32_t *)tok.dataPointer)[idx];
    } else if (tok.dataType == MLMultiArrayDataTypeFloat32) {
        *dst = (int)lrintf(((float *)tok.dataPointer)[idx]);
    } else {
        *dst = [[tok objectForKeyedSubscript:@[@(b), @(col)]] intValue];
    }
    return 0;
}

/* Build prefill inputs (S=Lmax, per-lane lens + write_lane preserve-mask) and run.
 * Returns the result provider (nil on error). Shared by the tok and hidden variants. */
static id<MLFeatureProvider> decb_prefill_run(const float *emb, const int *lens,
        const int *write_lane, int Lmax, const float *cosv, const float *sinv, NSError **err) {
    int B = g_B;
    NSNumber *nS = @(Lmax);
    MLMultiArray *x     = mk(@[@(B), nS, @(g_HID)]);
    MLMultiArray *cosa  = mk(@[@(B), nS, @HDIM]);
    MLMultiArray *sina  = mk(@[@(B), nS, @HDIM]);
    MLMultiArray *wmask = mk(@[@(B), @MAXLEN, nS]);
    MLMultiArray *amask = mk(@[@(B), @1, nS, @MAXLEN]);
    if (!x || !cosa || !sina || !wmask || !amask) return nil;

    memcpy((float *)x.dataPointer, emb, (size_t)B * Lmax * g_HID * sizeof(float));
    float *pc = (float *)cosa.dataPointer, *ps = (float *)sina.dataPointer;
    for (int b = 0; b < B; b++) {                 /* positions 0..Lmax-1 shared across lanes */
        memcpy(pc + (size_t)b * Lmax * HDIM, cosv, (size_t)Lmax * HDIM * sizeof(float));
        memcpy(ps + (size_t)b * Lmax * HDIM, sinv, (size_t)Lmax * HDIM * sizeof(float));
    }
    float *pw = (float *)wmask.dataPointer;       /* [B,MAXLEN,Lmax] */
    float *pa = (float *)amask.dataPointer;       /* [B,1,Lmax,MAXLEN] */
    memset(pw, 0, (size_t)B * MAXLEN * Lmax * sizeof(float));
    for (int b = 0; b < B; b++) {
        int wr = write_lane ? write_lane[b] : 1;              /* 0 => preserve this lane's KV */
        int L = lens[b]; if (L < 1) L = 1; if (L > Lmax) L = Lmax;
        if (wr) {
            for (int p = 0; p < L; p++)
                pw[((size_t)b * MAXLEN + p) * Lmax + p] = 1.0f;
        }
        for (int s = 0; s < Lmax; s++) {
            float *row = pa + ((size_t)b * Lmax + s) * MAXLEN;
            int lim = wr ? ((s < L) ? s : (L - 1)) : 0;
            for (int j = 0; j < MAXLEN; j++) row[j] = (j <= lim) ? 0.0f : -1e9f;
        }
    }
    MLDictionaryFeatureProvider *input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"x": [MLFeatureValue featureValueWithMultiArray:x],
        @"cos": [MLFeatureValue featureValueWithMultiArray:cosa],
        @"sin": [MLFeatureValue featureValueWithMultiArray:sina],
        @"wmask": [MLFeatureValue featureValueWithMultiArray:wmask],
        @"amask": [MLFeatureValue featureValueWithMultiArray:amask],
    } error:err];
    if (*err) return nil;
    return [g_modelb predictionFromFeatures:input usingState:g_stateb error:err];
}

int gpu_decb_prefill(const float *emb, const int *lens, const int *write_lane, int Lmax,
                     const float *cosv, const float *sinv, int *out_tok) {
    @autoreleasepool {
        if (!g_modelb || Lmax < 1 || Lmax > MAXLEN) return 1;
        NSError *err = nil;
        id<MLFeatureProvider> r = decb_prefill_run(emb, lens, write_lane, Lmax, cosv, sinv, &err);
        if (err || !r) { NSLog(@"[gpu_decb] prefill failed: %@", err); return 4; }
        for (int b = 0; b < g_B; b++) {
            int L = lens[b]; if (L < 1) L = 1; if (L > Lmax) L = Lmax;
            if (read_tok(r, &out_tok[b], b, Lmax, L - 1) != 0) return 5;
        }
        return 0;
    }
}

int gpu_decb_prefill_h(const float *emb, const int *lens, const int *write_lane, int Lmax,
                       const float *cosv, const float *sinv, float *out_hidden) {
    @autoreleasepool {
        if (!g_modelb || Lmax < 1 || Lmax > MAXLEN) return 1;
        NSError *err = nil;
        id<MLFeatureProvider> r = decb_prefill_run(emb, lens, write_lane, Lmax, cosv, sinv, &err);
        if (err || !r) { NSLog(@"[gpu_decb] prefill_h failed: %@", err); return 4; }
        MLMultiArray *h = [r featureValueForName:@"hidden"].multiArrayValue;  /* [B,Lmax,H] */
        if (!h) { NSLog(@"[gpu_decb] no 'hidden' output"); return 6; }
        const float *ph = (const float *)h.dataPointer;
        for (int b = 0; b < g_B; b++) {
            int L = lens[b]; if (L < 1) L = 1; if (L > Lmax) L = Lmax;
            memcpy(out_hidden + (size_t)b * g_HID,
                   ph + ((size_t)b * Lmax + (L - 1)) * g_HID, g_HID * sizeof(float));
        }
        return 0;
    }
}

/* Build S=1 decode inputs (reused persistent arrays) and run. nil on error. */
static id<MLFeatureProvider> decb_step_run(const float *emb, const int *positions,
        const float *cosv, const float *sinv, NSError **err) {
    int B = g_B;
    if (!g_bx) {
        g_bx  = mk(@[@(B), @1, @(g_HID)]);  g_bcos = mk(@[@(B), @1, @HDIM]);
        g_bsin = mk(@[@(B), @1, @HDIM]); g_bwm = mk(@[@(B), @MAXLEN, @1]);
        g_bam = mk(@[@(B), @1, @1, @MAXLEN]);
        if (!g_bx || !g_bcos || !g_bsin || !g_bwm || !g_bam) return nil;
        g_binput = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"x": [MLFeatureValue featureValueWithMultiArray:g_bx],
            @"cos": [MLFeatureValue featureValueWithMultiArray:g_bcos],
            @"sin": [MLFeatureValue featureValueWithMultiArray:g_bsin],
            @"wmask": [MLFeatureValue featureValueWithMultiArray:g_bwm],
            @"amask": [MLFeatureValue featureValueWithMultiArray:g_bam],
        } error:err];
        if (*err) return nil;
    }
    memcpy((float *)g_bx.dataPointer,   emb,  (size_t)B * g_HID  * sizeof(float));
    memcpy((float *)g_bcos.dataPointer, cosv, (size_t)B * HDIM * sizeof(float));
    memcpy((float *)g_bsin.dataPointer, sinv, (size_t)B * HDIM * sizeof(float));
    float *pw = (float *)g_bwm.dataPointer;    /* [B,MAXLEN,1] */
    float *pa = (float *)g_bam.dataPointer;    /* [B,1,1,MAXLEN] */
    memset(pw, 0, (size_t)B * MAXLEN * sizeof(float));
    for (int b = 0; b < B; b++) {
        int pos = positions[b]; if (pos < 0) pos = 0; if (pos >= MAXLEN) pos = MAXLEN - 1;
        pw[(size_t)b * MAXLEN + pos] = 1.0f;
        float *row = pa + (size_t)b * MAXLEN;
        for (int j = 0; j < MAXLEN; j++) row[j] = (j <= pos) ? 0.0f : -1e9f;
    }
    return [g_modelb predictionFromFeatures:g_binput usingState:g_stateb error:err];
}

int gpu_decb_step(const float *emb, const int *positions,
                  const float *cosv, const float *sinv, int *out_tok) {
    @autoreleasepool {
        if (!g_modelb) return 1;
        NSError *err = nil;
        id<MLFeatureProvider> r = decb_step_run(emb, positions, cosv, sinv, &err);
        if (err || !r) { NSLog(@"[gpu_decb] step failed: %@", err); return 5; }
        for (int b = 0; b < g_B; b++)
            if (read_tok(r, &out_tok[b], b, 1, 0) != 0) return 6;
        return 0;
    }
}

int gpu_decb_step_h(const float *emb, const int *positions,
                    const float *cosv, const float *sinv, float *out_hidden) {
    @autoreleasepool {
        if (!g_modelb) return 1;
        NSError *err = nil;
        id<MLFeatureProvider> r = decb_step_run(emb, positions, cosv, sinv, &err);
        if (err || !r) { NSLog(@"[gpu_decb] step_h failed: %@", err); return 5; }
        MLMultiArray *h = [r featureValueForName:@"hidden"].multiArrayValue;  /* [B,1,H] */
        if (!h) { NSLog(@"[gpu_decb] no 'hidden' output"); return 6; }
        memcpy(out_hidden, (const float *)h.dataPointer, (size_t)g_B * g_HID * sizeof(float));
        return 0;
    }
}

void gpu_decb_free(void) {
    g_modelb = nil; g_stateb = nil;
    g_bx = g_bcos = g_bsin = g_bwm = g_bam = nil; g_binput = nil;
}
