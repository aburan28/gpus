# gpuckpt — incremental, deduplicated GPU checkpoint storage on top of the
# CUDA checkpoint custom-storage interface.
#
# Build targets
#   make            build/gpuckpt (CLI) + build/libgpuckpt.a (+ mock driver, signer test)
#   make test       CPU-only test suite (host-image backend + mock libcuda)
#   make bench      synthetic CPU-only benchmark of the storage layer
#   make clean
#
# The CUDA adapter needs <cuda.h> for the checkpoint struct layouts. If
# $(CUDA_HOME)/include/cuda.h exists it is used and the binary reports
# "cuda-header: real". Otherwise the documentation-derived header in
# tests/mock/cuda.h is used and the binary reports "cuda-header: mock
# (UNVERIFIED)". A mock-header build must never be used against a real
# driver; see docs/design.md "Build modes".

CC      ?= cc
CFLAGS  ?= -O2 -g
override CFLAGS += -std=gnu11 -Wall -Wextra -Wshadow -Wpointer-arith -pthread -Iinclude -Isrc
LDFLAGS += -pthread
LDLIBS  += -ldl

CUDA_HOME ?= /usr/local/cuda
ifneq ($(wildcard $(CUDA_HOME)/include/cuda.h),)
  CUDA_INC   := -I$(CUDA_HOME)/include -DGPUCKPT_CUDA_HEADER_REAL=1
  CUDA_HDR   := real
else
  CUDA_INC   := -Itests/mock -DGPUCKPT_CUDA_HEADER_REAL=0
  CUDA_HDR   := mock
endif
# GPUDirect Storage header: the toolkit's when present (CUDA >= 11.4), else the
# documentation-derived stand-in (binary reports "cufile-header: mock").
ifneq ($(and $(filter real,$(CUDA_HDR)),$(wildcard $(CUDA_HOME)/include/cufile.h)),)
  CUDA_INC   += -DGPUCKPT_CUFILE_HEADER_REAL=1
else
  CUDA_INC   += -Itests/mock/cufile -DGPUCKPT_CUFILE_HEADER_REAL=0
endif

# SHA-256: OpenSSL EVP when available (SHA-NI, ~10x faster), builtin otherwise.
PKG_CONFIG ?= pkg-config
ifeq ($(shell $(PKG_CONFIG) --exists libcrypto && echo yes),yes)
  override CFLAGS += -DGC_USE_OPENSSL $(shell $(PKG_CONFIG) --cflags libcrypto)
  LDLIBS += $(shell $(PKG_CONFIG) --libs libcrypto)
endif

# S3 backend: needs libcurl. Without it the binary builds and refuses s3:// repos.
ifeq ($(shell $(PKG_CONFIG) --exists libcurl && echo yes),yes)
  override CFLAGS += -DGC_USE_CURL $(shell $(PKG_CONFIG) --cflags libcurl)
  LDLIBS += $(shell $(PKG_CONFIG) --libs libcurl)
endif

B := build
LIB_SRCS := src/sha256.c src/store.c src/manifest.c src/image_file.c src/snapshot.c src/util.c src/objstore.c src/s3.c src/hostmem.c src/cuda_backend.c
LIB_OBJS := $(patsubst src/%.c,$(B)/%.o,$(LIB_SRCS)) $(B)/kernel_src.o
KERNEL   := src/kernels/sha256_chunks.cu

MOCKS := $(B)/libcuda_mock.so $(B)/libnvrtc_mock.so $(B)/libcufile_mock.so

all: $(B)/gpuckpt $(B)/libgpuckpt.a $(MOCKS) $(B)/sigv4_test $(B)/kernel_test
	@echo "built: cuda-header=$(CUDA_HDR)"

$(B):
	mkdir -p $(B)

$(B)/%.o: src/%.c include/gpuckpt.h src/*.h | $(B)
	$(CC) $(CFLAGS) -c -o $@ $<

$(B)/cuda_backend.o: src/cuda_backend.c include/gpuckpt.h src/*.h | $(B)
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

# The CUDA kernel source is embedded verbatim for NVRTC.
$(B)/embed: tools/embed.c | $(B)
	$(CC) -O1 -o $@ $<

$(B)/kernel_src.c: $(KERNEL) $(B)/embed
	$(B)/embed gc_kernel_sha256_src $(KERNEL) > $@

$(B)/kernel_src.o: $(B)/kernel_src.c
	$(CC) $(CFLAGS) -c -o $@ $<

$(B)/libgpuckpt.a: $(LIB_OBJS)
	ar rcs $@ $^

$(B)/gpuckpt: src/cli.c $(B)/libgpuckpt.a
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ src/cli.c $(B)/libgpuckpt.a $(LDFLAGS) $(LDLIBS)

# Mock driver: implements the checkpoint state machine over host buffers so the
# full lock -> checkpoint -> copy -> complete -> restore -> complete -> unlock
# path runs without a GPU. Always built against the mock header.
$(B)/libcuda_mock.so: tests/mock/libcuda_mock.c tests/mock/cuda.h $(KERNEL) | $(B)
	$(CC) $(CFLAGS) -Wno-missing-prototypes -Itests/mock -Isrc -shared -fPIC -o $@ $<

$(B)/libnvrtc_mock.so: tests/mock/libnvrtc_mock.c | $(B)
	$(CC) $(CFLAGS) -Wno-missing-prototypes -shared -fPIC -o $@ $<

$(B)/libcufile_mock.so: tests/mock/libcufile_mock.c tests/mock/cufile/cufile.h | $(B)
	$(CC) $(CFLAGS) -Wno-missing-prototypes -Itests/mock -Itests/mock/cufile -shared -fPIC -o $@ $<

$(B)/kernel_test: tests/kernel_test.c $(KERNEL) $(B)/libgpuckpt.a
	$(CC) $(CFLAGS) -o $@ tests/kernel_test.c $(B)/libgpuckpt.a $(LDFLAGS) $(LDLIBS)

# Prebuilt PTX for hosts without NVRTC at runtime (GPUCKPT_KERNEL_PTX=build/sha256_chunks.ptx).
ptx: $(B)/sha256_chunks.ptx
$(B)/sha256_chunks.ptx: $(KERNEL) | $(B)
	$(CUDA_HOME)/bin/nvcc -ptx -arch=$${GPUCKPT_PTX_ARCH:-compute_70} -o $@ $(KERNEL)

$(B)/sigv4_test: tests/sigv4_test.c $(B)/libgpuckpt.a
	$(CC) $(CFLAGS) -o $@ tests/sigv4_test.c $(B)/libgpuckpt.a $(LDFLAGS) $(LDLIBS)

test: all
	bash tests/run.sh

bench: all
	bash bench/bench.sh

clean:
	rm -rf $(B)

.PHONY: all test bench clean ptx
