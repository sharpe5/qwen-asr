# qwen_asr — Qwen3-ASR Pure C Inference Engine
# Makefile

# Toolchain: Homebrew LLVM clang (one toolchain for all targets).
# Chosen over Apple clang (no libFuzzer runtime) and `zig cc` (ships neither the
# ASan nor libFuzzer runtime) so blas/gpu/debug/sanitize/fuzz all share one compiler.
LLVM_PREFIX := $(shell brew --prefix llvm 2>/dev/null)
CC = $(LLVM_PREFIX)/bin/clang

# libsndfile (optional): enables native -i decode of opus/ogg/flac/mp3.
# Auto-detected via Homebrew; if absent, the build still works (WAV/stdin only).
SNDFILE_PREFIX := $(shell brew --prefix libsndfile 2>/dev/null)
ifneq ($(wildcard $(SNDFILE_PREFIX)/include/sndfile.h),)
  SNDFILE_CFLAGS := -I$(SNDFILE_PREFIX)/include -DHAVE_SNDFILE
  SNDFILE_LIBS := -L$(SNDFILE_PREFIX)/lib -lsndfile
endif

CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math -Wno-date-time $(SNDFILE_CFLAGS)
LDFLAGS = -lm -lpthread $(SNDFILE_LIBS)

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_asr

# Model dir for the regression suite. Auto-detects whichever is present
# (prefers 1.7B), override on the CLI: make test MODEL_DIR=path/to/model
MODEL_DIR ?= $(firstword $(wildcard qwen3-asr-1.7b qwen3-asr-0.6b))

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

# Sanitizer / fuzzing flags (ASan + UBSan; no -ffast-math so UBSan stays honest).
SAN_FLAGS    = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
SAN_CFLAGS   = -Wall -Wextra -g -O1 -march=native -Wno-date-time $(SNDFILE_CFLAGS) \
               -DUSE_BLAS -DACCELERATE_NEW_LAPACK $(SAN_FLAGS)
# Fuzz: shared objects get coverage without a main; harnesses get the libFuzzer main.
FUZZ_BASE    = -g -O1 -Wno-date-time $(SNDFILE_CFLAGS) -DUSE_BLAS -DACCELERATE_NEW_LAPACK $(SAN_FLAGS)
FUZZ_LDLIBS  = -lm -lpthread -framework Accelerate $(SNDFILE_LIBS)
FUZZ_OBJS    = $(SRCS:.c=.fuzz.o)
FUZZ_DIR     = fuzz
FUZZ_BINS    = $(FUZZ_DIR)/fuzz_wav $(FUZZ_DIR)/fuzz_safetensors $(FUZZ_DIR)/fuzz_tokenizer $(FUZZ_DIR)/fuzz_mel
# Per-target wall-clock budget for an overnight run (seconds); override on the CLI.
FUZZ_SECONDS ?= 14400        # 4h each => ~8h to fuzz both targets
FUZZ_RSS_MB  ?= 4096
# Fork mode: the parent keeps fuzzing after a child crashes, so one bug does not
# end the night — every distinct crash is saved and we keep finding more.
FUZZ_FORKS   ?= 2

# Bare `make` should print the target list. Set this explicitly: GNU Make otherwise
# picks the first non-dot target (check-llvm) as the default goal, which runs silently.
.DEFAULT_GOAL := help

.PHONY: all clean debug info help blas gpu gpu_link test test-gpu test-stream-cache \
        check-llvm check-model sanitize sanitize-test sanitize-gpu tsan fuzz fuzz-wav fuzz-safetensors fuzz-tokenizer fuzz-mel overnight gpu-model

# Fail early with a clear message if the Homebrew LLVM toolchain is missing.
check-llvm:
	@test -n "$(LLVM_PREFIX)" && test -x "$(CC)" || { \
	  echo "ERROR: Homebrew LLVM clang not found (LLVM_PREFIX='$(LLVM_PREFIX)', CC='$(CC)')."; \
	  echo "Install it with:  brew install llvm"; exit 1; }

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
	@echo "Correctness / overnight:"
	@echo "  make sanitize       - ASan+UBSan build (then: make sanitize-test)"
	@echo "  make sanitize-gpu   - ASan+UBSan GPU build (CoreML decoder; run with --gpu)"
	@echo "  make fuzz           - Build the parser fuzzers"
	@echo "  make fuzz-wav       - Fuzz the WAV parser (FUZZ_SECONDS=$(FUZZ_SECONDS))"
	@echo "  make fuzz-safetensors - Fuzz the safetensors reader"
	@echo "  make overnight      - Build + fuzz both targets unattended; logs/crashes in fuzz/findings/"
	@echo ""
	@echo "Example: make blas && ./qwen_asr -d model_dir -i audio.wav"
	@echo "Overnight: make overnight FUZZ_SECONDS=28800   # 8h per target"

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
blas: check-llvm
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: gpu (BLAS + CoreML GPU decoder fast path; macOS only)
# Adds --gpu flag, routing the decoder through coreml_decoder.mm.
# `make gpu` compiles the binary AND generates the CoreML .mlpackage(s) the
# --gpu path loads, for each downloaded model (see coreml_export/).
# =============================================================================
GPU_OBJ = coreml_decoder.o

# CoreML export tooling: `uv run` converts the HF decoder weights into the
# .mlpackage the binary loads. One package per model size. The export scripts
# carry their own deps (PEP 723 inline metadata: coremltools/torch + a
# Python <3.13 pin, since coremltools 9.0's BlobWriter has no wheel for >=3.13);
# uv provisions a matching interpreter + deps on first run and caches them.
EXPORT_DIR       = coreml_export
GPU_MODELS      ?= qwen3-asr-0.6b qwen3-asr-1.7b
GPU_BATCH       ?= 4
GPU_FORCE_EXPORT ?=

ifeq ($(UNAME_S),Darwin)
gpu: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK -DQWEN_GPU
gpu: LDFLAGS = -lm -lpthread -framework Accelerate -framework CoreML -framework Foundation -lobjc -lc++ $(SNDFILE_LIBS)
gpu: check-llvm
	@$(MAKE) clean
	@$(MAKE) gpu_link CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with GPU (CoreML) decoder — run with --gpu"
	@$(MAKE) gpu-model

# Generate the CoreML .mlpackage(s) the --gpu path loads, from downloaded HF
# weights, via `uv run` (interpreter + deps come from each export script's inline
# PEP 723 metadata; cached after first run). Exports one batched-hidden package
# per present model size, skipping any that already exist (GPU_FORCE_EXPORT=1 to
# regenerate). Safe to run standalone. Requires uv: https://docs.astral.sh/uv/.
gpu-model:
	@command -v uv >/dev/null || { echo "uv required for GPU model export: brew install uv  (https://docs.astral.sh/uv/)"; exit 1; }
	@found=0; \
	for m in $(GPU_MODELS); do \
	  [ -f "$$m/config.json" ] || continue; found=1; \
	  tag=$$(echo "$$m" | grep -oE '0\.6b|1\.7b'); \
	  pkg="qwen_decoder_gpu_$${tag}_b$(GPU_BATCH)_hidden.mlpackage"; \
	  if [ -d "$$pkg" ] && [ -z "$(GPU_FORCE_EXPORT)" ]; then \
	    echo "==> $$pkg exists (skip; GPU_FORCE_EXPORT=1 to regenerate)"; \
	  else \
	    echo "==> exporting $$pkg from $$m via uv run (first run downloads ~GB; 1.7B takes minutes) ..."; \
	    ( cd "$(EXPORT_DIR)" && QWEN_MODEL_DIR="$(CURDIR)/$$m" uv run export_decoder_batched.py $(GPU_BATCH) hidden ) || exit 1; \
	  fi; \
	done; \
	if [ $$found -eq 0 ]; then \
	  echo "==> no model present ($(GPU_MODELS)); run ./download_model.sh first."; \
	  echo "    Binary is built; --gpu needs a generated package."; \
	fi

gpu_link: $(OBJS) main.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) main.o $(GPU_OBJ) $(LDFLAGS)

# GPU_MM_EXTRA lets the sanitize-gpu target instrument the Objective-C++ decoder
# too (default empty = normal build). Without it, OOB writes inside the .mm would
# not be caught, since ASan only checks instrumented accesses.
GPU_MM_EXTRA ?=
$(GPU_OBJ): coreml_decoder.mm coreml_decoder.h
	$(CC) -fobjc-arc -O3 $(GPU_MM_EXTRA) -c -o $@ coreml_decoder.mm

# Sanitized GPU build: ASan+UBSan over the CPU pipeline AND the CoreML decoder
# marshalling in coreml_decoder.mm. Run the result manually with --gpu (0.6B,
# segmented, e.g. -S 28) under ASAN_OPTIONS to exercise the GPU path. CoreML
# itself is not instrumented, but the C/ObjC++ buffer handling around it is.
sanitize-gpu: check-llvm
	@$(MAKE) clean
	@$(MAKE) gpu_link CC="$(CC)" \
	  CFLAGS="$(SAN_CFLAGS) -DQWEN_GPU" \
	  GPU_MM_EXTRA="$(SAN_FLAGS)" \
	  LDFLAGS="-lm -lpthread -framework Accelerate -framework CoreML -framework Foundation -lobjc -lc++ $(SNDFILE_LIBS) $(SAN_FLAGS)"
	@echo ""
	@echo "Built sanitized GPU (ASan+UBSan, --gpu via CoreML). Run e.g.:"
	@echo "  ASAN_OPTIONS=abort_on_error=1:detect_leaks=0 ./qwen_asr -d qwen3-asr-0.6b -i samples/jfk.wav --gpu -S 28"
else
gpu:
	@echo "gpu target is macOS-only (CoreML)"; exit 1
sanitize-gpu:
	@echo "sanitize-gpu is macOS-only (CoreML)"; exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_asr.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build (AddressSanitizer). Uses the shared Homebrew LLVM clang, which ships
# the ASan runtime; CC is forwarded into the recursive make so it isn't lost.
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: check-llvm
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CC="$(CC)" CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Sanitized build (ASan + UBSan) — for running the regression suite under
# sanitizers on realistic inputs. Note: LeakSanitizer is unreliable on Apple
# Silicon, so detect_leaks is left off here; run the Linux build for leak checks.
# =============================================================================
sanitize: check-llvm
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CC="$(CC)" CFLAGS="$(SAN_CFLAGS)" \
	         LDFLAGS="-lm -lpthread -framework Accelerate $(SNDFILE_LIBS) $(SAN_FLAGS)"
	@echo ""
	@echo "Built sanitized (ASan+UBSan). Run: make sanitize-test"

# Run the regression suite under the sanitized binary, logging to fuzz/findings/.
sanitize-test: check-model sanitize
	@mkdir -p $(FUZZ_DIR)/findings
	ASAN_OPTIONS=abort_on_error=1:detect_leaks=0 \
	UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
	  ./asr_regression.py --binary ./$(TARGET) --model-dir $(MODEL_DIR) 2>&1 \
	  | tee $(FUZZ_DIR)/findings/sanitize-$$(date +%Y%m%d-%H%M%S).log

# ThreadSanitizer build — detects data races in the pthread thread pool
# (qwen_asr_kernels.c). Separate from ASan (the two cannot be combined). No
# -ffast-math. Run the regression / a few transcriptions to exercise the pool.
TSAN_CFLAGS = -Wall -Wextra -g -O1 -march=native -Wno-date-time $(SNDFILE_CFLAGS) \
              -DUSE_BLAS -DACCELERATE_NEW_LAPACK -fsanitize=thread
tsan: check-llvm
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CC="$(CC)" CFLAGS="$(TSAN_CFLAGS)" \
	         LDFLAGS="-lm -lpthread -framework Accelerate $(SNDFILE_LIBS) -fsanitize=thread"
	@echo ""
	@echo "Built ThreadSanitizer build — run a transcription to check for data races"

# =============================================================================
# Fuzzing (libFuzzer + ASan + UBSan). Shared objects are built with
# -fsanitize=fuzzer-no-link (coverage, no main); each harness links the libFuzzer
# main via -fsanitize=fuzzer. Corpus persists under fuzz/corpus/, crashes land in
# fuzz/findings/ as <target>-crash-<hash> files (re-run a fuzzer on that file to
# reproduce).
# =============================================================================
%.fuzz.o: %.c
	$(CC) $(FUZZ_BASE) -fsanitize=fuzzer-no-link -c -o $@ $<

$(FUZZ_DIR)/fuzz_%: $(FUZZ_DIR)/fuzz_%.c $(FUZZ_OBJS)
	$(CC) $(FUZZ_BASE) -fsanitize=fuzzer -I. -o $@ $< $(FUZZ_OBJS) $(FUZZ_LDLIBS)

fuzz: check-llvm $(FUZZ_BINS)
	@echo ""
	@echo "Built fuzzers: $(FUZZ_BINS)"
	@echo "Run one:  make fuzz-wav   |   make fuzz-safetensors   (FUZZ_SECONDS=$(FUZZ_SECONDS))"
	@echo "Run both overnight:  make overnight"

# Common libFuzzer invocation: persistent corpus + seeds, crashes -> findings/, tee'd log.
# Fork mode (-fork) + -ignore_* means a crash/timeout/OOM is logged and fuzzing
# CONTINUES, so an overnight run collects every distinct failure instead of
# stopping at the first. Each unique crash is saved as a separate reproducer file.
# $(1)=target name  $(2)=corpus subdir
define RUN_FUZZER
	@mkdir -p $(FUZZ_DIR)/corpus/$(2) $(FUZZ_DIR)/findings
	ASAN_OPTIONS=abort_on_error=1:detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
	  $(FUZZ_DIR)/fuzz_$(1) -fork=$(FUZZ_FORKS) \
	  -ignore_crashes=1 -ignore_timeouts=1 -ignore_ooms=1 \
	  -max_total_time=$(FUZZ_SECONDS) -rss_limit_mb=$(FUZZ_RSS_MB) $(FUZZ_EXTRA) \
	  -print_final_stats=1 -artifact_prefix=$(FUZZ_DIR)/findings/$(1)-crash- \
	  $(FUZZ_DIR)/corpus/$(2) 2>&1 \
	  | tee $(FUZZ_DIR)/findings/$(1)-$$(date +%Y%m%d-%H%M%S).log
endef

fuzz-wav: $(FUZZ_DIR)/fuzz_wav
	@mkdir -p $(FUZZ_DIR)/corpus/wav
	@cp -n samples/*.wav $(FUZZ_DIR)/corpus/wav/ 2>/dev/null || true
	$(call RUN_FUZZER,wav,wav)

fuzz-safetensors: $(FUZZ_DIR)/fuzz_safetensors
	@mkdir -p $(FUZZ_DIR)/corpus/safetensors
	@# Seed with a minimal valid (empty) safetensors file: u64 header_len=2, then "{}".
	@test -s $(FUZZ_DIR)/corpus/safetensors/seed-empty || \
	  printf '\002\000\000\000\000\000\000\000{}' > $(FUZZ_DIR)/corpus/safetensors/seed-empty
	$(call RUN_FUZZER,safetensors,safetensors)

fuzz-tokenizer: $(FUZZ_DIR)/fuzz_tokenizer
	@mkdir -p $(FUZZ_DIR)/corpus/tokenizer
	@test -s $(FUZZ_DIR)/corpus/tokenizer/seed || \
	  printf 'Hello, world! CPU CUDA PostgreSQL Redis 123' > $(FUZZ_DIR)/corpus/tokenizer/seed
	$(call RUN_FUZZER,tokenizer,tokenizer)

fuzz-mel: $(FUZZ_DIR)/fuzz_mel
	@mkdir -p $(FUZZ_DIR)/corpus/mel
	$(call RUN_FUZZER,mel,mel)

# Unattended overnight run: build everything, then fuzz both targets in sequence.
# The leading '-' lets the second fuzzer run even if the first finds a crash
# (libFuzzer exits non-zero on a finding). Logs + crashes are under fuzz/findings/.
overnight: fuzz
	@mkdir -p $(FUZZ_DIR)/findings
	@echo "=== overnight fuzz run started $$(date) (FUZZ_SECONDS=$(FUZZ_SECONDS) per target) ==="
	-$(MAKE) fuzz-wav
	-$(MAKE) fuzz-safetensors
	@echo "=== overnight fuzz run finished $$(date) ==="
	@echo "Crash artifacts (if any):"
	@ls -1 $(FUZZ_DIR)/findings/*-crash-* 2>/dev/null || echo "  none — no crashes found"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) main.o $(GPU_OBJ) $(TARGET) $(FUZZ_OBJS) $(FUZZ_BINS)
	@# Note: fuzz/corpus and fuzz/findings are intentionally NOT removed.

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

# Guard: a clear, actionable message when no model is present. (The fuzzers and
# `make overnight` need NO model — only the quality regression below does.)
check-model:
	@test -n "$(MODEL_DIR)" || { \
	  echo "No model dir found (looked for qwen3-asr-1.7b / qwen3-asr-0.6b)."; \
	  echo "Download one:   ./download_model.sh --model small   # Qwen3-ASR-0.6B"; \
	  echo "          or:   ./download_model.sh --model large   # Qwen3-ASR-1.7B"; \
	  echo "Use existing:   make test MODEL_DIR=/path/to/model"; \
	  exit 1; }

# `make test` builds a fresh binary, then exercises BOTH decode paths:
#   1. CPU  — the quality regression (the binary decodes on CPU without --gpu)
#   2. GPU  — a functional pass with --gpu (CoreML), via test-gpu
# On macOS it builds `gpu` (the CPU+GPU superset) so both paths are present; on
# other platforms it builds `blas` and runs the CPU regression only.
ifeq ($(UNAME_S),Darwin)
test: check-model gpu
	@echo ""
	@echo "==> CPU decode path: quality regression on $(MODEL_DIR)"
	./asr_regression.py --binary ./$(TARGET) --model-dir $(MODEL_DIR)
	@$(MAKE) test-gpu
else
test: check-model blas
	@echo "==> CPU decode path: quality regression on $(MODEL_DIR)"
	./asr_regression.py --binary ./$(TARGET) --model-dir $(MODEL_DIR)
	@echo "(GPU path is macOS-only; skipping --gpu exercise)"
endif

# GPU functional check: run the --gpu (CoreML) path over the sample WAVs. GPU
# mode is 0.6B-only and segmented, so it uses qwen3-asr-0.6b with -S 28. Requires
# the model dir + qwen_decoder_gpu.mlpackage; if either is absent, SKIP (don't
# fail). Each sample must exit 0 with non-empty transcription on stdout.
test-gpu:
	@any=0; fail=0; \
	for m in $(GPU_MODELS); do \
	  tag=$$(echo "$$m" | grep -oE '0\.6b|1\.7b'); \
	  pkg="qwen_decoder_gpu_$${tag}_b$(GPU_BATCH)_hidden.mlpackage"; \
	  if [ ! -d "$$m" ] || [ ! -d "$$pkg" ]; then \
	    echo "==> GPU decode ($$m): SKIP (need $$m + $$pkg)"; continue; fi; \
	  any=1; n=0; \
	  echo "==> GPU decode path ($$m, --gpu -S 28) over samples"; \
	  for w in $$(find samples -type f -name '*.wav' | sort); do \
	    n=$$((n+1)); \
	    out=$$(./$(TARGET) -d "$$m" -i "$$w" --gpu -S 28 --silent 2>/dev/null); \
	    if [ -z "$$out" ]; then echo "  GPU FAIL (empty output): $$w"; fail=1; \
	    else echo "  GPU ok: $$w"; fi; \
	  done; \
	  echo "  ($$m: $$n samples)"; \
	done; \
	if [ $$any -eq 0 ]; then echo "==> GPU decode path: SKIP (no model+package present)"; exit 0; fi; \
	if [ $$fail -ne 0 ]; then echo "GPU path check FAILED"; exit 1; fi; \
	echo "GPU path check PASSED"

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
