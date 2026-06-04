/*
 * coreml_decoder.mm - Objective-C++ bridge: drives the exported stateful
 * Qwen3-ASR decoder on the GPU (CPU_AND_GPU) for the C engine's --gpu fast path.
 *
 * Model I/O (from ane_proto/export_decoder.py):
 *   inputs : x[1,1,1024] cos[1,1,1,128] sin[1,1,1,128] wmask[1,1,512,1] amask[1,1,1,512]
 *   states : kc_0..kc_27, vc_0..vc_27  (on-device KV cache)
 *   output : hidden[1,1024]
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
        NSNumber *nS = @(S);
        MLMultiArray *x     = mk(@[@1, nS, @HID]);
        MLMultiArray *cosa  = mk(@[@1, nS, @HDIM]);
        MLMultiArray *sina  = mk(@[@1, nS, @HDIM]);
        MLMultiArray *wmask = mk(@[@MAXLEN, nS]);
        MLMultiArray *amask = mk(@[@1, @1, nS, @MAXLEN]);
        if (!x || !cosa || !sina || !wmask || !amask) return 2;

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

        MLDictionaryFeatureProvider *input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"x":     [MLFeatureValue featureValueWithMultiArray:x],
                @"cos":   [MLFeatureValue featureValueWithMultiArray:cosa],
                @"sin":   [MLFeatureValue featureValueWithMultiArray:sina],
                @"wmask": [MLFeatureValue featureValueWithMultiArray:wmask],
                @"amask": [MLFeatureValue featureValueWithMultiArray:amask],
            } error:&err];
        if (err) { NSLog(@"[gpu_dec] provider failed: %@", err); return 4; }

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

int gpu_dec_hidden(void)   { return HID; }
int gpu_dec_head_dim(void) { return HDIM; }
int gpu_dec_max(void)      { return MAXLEN; }

void gpu_dec_free(void) {
    g_model = nil; g_state = nil;
}
