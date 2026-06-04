# qwen_asr — Qwen3-ASR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_asr

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas gpu gpu_link test test-stream-cache

# Default: show available targets
all: help

help:
	@echo "qwen_asr — Qwen3-ASR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make gpu      - BLAS + CoreML GPU decoder fast path (--gpu, macOS)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make test     - Run regression suite (requires ./qwen_asr and model files)"
	@echo "  make test-stream-cache - Run stream cache on/off equivalence check"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./qwen_asr -d model_dir -i audio.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: gpu (BLAS + CoreML GPU decoder fast path; macOS only)
# Adds --gpu flag, routing the decoder through coreml_decoder.mm.
# Needs qwen_decoder_gpu.mlpackage (see ane_proto/export_decoder.py).
# =============================================================================
GPU_OBJ = coreml_decoder.o
ifeq ($(UNAME_S),Darwin)
gpu: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DQWEN_GPU
gpu: LDFLAGS = -lm -lpthread -framework Accelerate -framework CoreML -framework Foundation -lobjc -lc++
gpu:
	@$(MAKE) clean
	@$(MAKE) gpu_link CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with GPU (CoreML) decoder — run with --gpu"

gpu_link: $(OBJS) main.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) main.o $(GPU_OBJ) $(LDFLAGS)

$(GPU_OBJ): coreml_decoder.mm coreml_decoder.h
	clang -fobjc-arc -O3 -c -o $@ coreml_decoder.mm
else
gpu:
	@echo "gpu target is macOS-only (CoreML)"; exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_asr.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) main.o $(GPU_OBJ) $(TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

test:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-1.7b

# =============================================================================
# Dependencies
# =============================================================================
qwen_asr.o: qwen_asr.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h qwen_asr_audio.h qwen_asr_tokenizer.h
qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
qwen_asr_audio.o: qwen_asr_audio.c qwen_asr_audio.h
qwen_asr_encoder.o: qwen_asr_encoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_decoder.o: qwen_asr_decoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h
qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
main.o: main.c qwen_asr.h qwen_asr_kernels.h
