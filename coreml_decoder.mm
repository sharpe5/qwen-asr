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
static MLMultiArray  *g_x = nil, *g_cos = nil, *g_sin = nil, *g_wmask = nil, *g_amask = nil;
static MLDictionaryFeatureProvider *g_input = nil;

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

        g_x     = mk(@[@1, @1, @HID]);
        g_cos   = mk(@[@1, @1, @1, @HDIM]);
        g_sin   = mk(@[@1, @1, @1, @HDIM]);
        g_wmask = mk(@[@1, @1, @MAXLEN, @1]);
        g_amask = mk(@[@1, @1, @1, @MAXLEN]);
        if (!g_x || !g_cos || !g_sin || !g_wmask || !g_amask) return 4;

        g_input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
            @"x":     [MLFeatureValue featureValueWithMultiArray:g_x],
            @"cos":   [MLFeatureValue featureValueWithMultiArray:g_cos],
            @"sin":   [MLFeatureValue featureValueWithMultiArray:g_sin],
            @"wmask": [MLFeatureValue featureValueWithMultiArray:g_wmask],
            @"amask": [MLFeatureValue featureValueWithMultiArray:g_amask],
        } error:&err];
        if (err) { NSLog(@"[gpu_dec] provider failed: %@", err); return 5; }
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

int gpu_dec_step(const float *x, const float *cosv, const float *sinv, int pos, float *out) {
    @autoreleasepool {
        if (!g_model || pos < 0 || pos >= MAXLEN) return 1;
        float *px = (float *)g_x.dataPointer;
        float *pc = (float *)g_cos.dataPointer;
        float *ps = (float *)g_sin.dataPointer;
        float *pw = (float *)g_wmask.dataPointer;
        float *pa = (float *)g_amask.dataPointer;
        memcpy(px, x, HID * sizeof(float));
        memcpy(pc, cosv, HDIM * sizeof(float));
        memcpy(ps, sinv, HDIM * sizeof(float));
        for (int j = 0; j < MAXLEN; j++) {
            pw[j] = (j == pos) ? 1.0f : 0.0f;           /* one-hot write position */
            pa[j] = (j <= pos) ? 0.0f : -1e9f;          /* causal / validity mask */
        }

        NSError *err = nil;
        id<MLFeatureProvider> result = [g_model predictionFromFeatures:g_input
                                                            usingState:g_state
                                                                 error:&err];
        if (err || !result) { NSLog(@"[gpu_dec] predict failed: %@", err); return 3; }

        MLFeatureValue *fv = [result featureValueForName:@"hidden"];
        MLMultiArray *h = fv.multiArrayValue;
        if (!h) { NSLog(@"[gpu_dec] no 'hidden' output"); return 4; }
        memcpy(out, (float *)h.dataPointer, HID * sizeof(float));
        return 0;
    }
}

int gpu_dec_hidden(void)   { return HID; }
int gpu_dec_head_dim(void) { return HDIM; }
int gpu_dec_max(void)      { return MAXLEN; }

void gpu_dec_free(void) {
    g_model = nil; g_state = nil; g_x = g_cos = g_sin = g_wmask = g_amask = nil; g_input = nil;
}
