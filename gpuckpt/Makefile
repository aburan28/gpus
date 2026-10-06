# gpuckpt — incremental, deduplicated GPU checkpoint storage on top of the
# CUDA checkpoint custom-storage interface.
#
# Build targets
#   make            build/gpuckpt (CLI) + build/libgpuckpt.a (+ mock driver, signer test)
#   make test       CPU-only test suite (host-image backend + mock libcuda)
#   make bench      synthetic CPU-only benchmark of the storage layer
#   make graph-workload  real-header graph workload (C + embedded PTX; no nvcc)
#   make test-graphs GRAPH_OUTPUT=/new/evidence/dir  real-driver replay validation
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
LIB_SRCS := src/sha256.c src/store.c src/manifest.c src/image_file.c src/snapshot.c src/util.c src/objstore.c src/s3.c src/cuda_backend.c
LIB_OBJS := $(patsubst src/%.c,$(B)/%.o,$(LIB_SRCS))

all: $(B)/gpuckpt $(B)/libgpuckpt.a $(B)/libcuda_mock.so $(B)/sigv4_test
	@echo "built: cuda-header=$(CUDA_HDR)"

$(B):
	mkdir -p $(B)

$(B)/%.o: src/%.c include/gpuckpt.h src/*.h | $(B)
	$(CC) $(CFLAGS) -c -o $@ $<

$(B)/cuda_backend.o: src/cuda_backend.c include/gpuckpt.h src/*.h | $(B)
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

$(B)/libgpuckpt.a: $(LIB_OBJS)
	ar rcs $@ $^

$(B)/gpuckpt: src/cli.c $(B)/libgpuckpt.a
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ src/cli.c $(B)/libgpuckpt.a $(LDFLAGS) $(LDLIBS)

# Mock driver: implements the checkpoint state machine over host buffers so the
# full lock -> checkpoint -> copy -> complete -> restore -> complete -> unlock
# path runs without a GPU. Always built against the mock header.
$(B)/libcuda_mock.so: tests/mock/libcuda_mock.c tests/mock/cuda.h | $(B)
	$(CC) $(CFLAGS) -Wno-missing-prototypes -Itests/mock -shared -fPIC -o $@ $<

$(B)/sigv4_test: tests/sigv4_test.c $(B)/libgpuckpt.a
	$(CC) $(CFLAGS) -o $@ tests/sigv4_test.c $(B)/libgpuckpt.a $(LDFLAGS) $(LDLIBS)

test: all $(B)/libcuda_mock.so $(B)/sigv4_test
	bash tests/run.sh
	python3 -m unittest discover -s tests -p 'test_graph_replay.py' -v

# A graph fixture must never build against tests/mock/cuda.h. The CPU test
# suite tests the Python orchestration, not CUDA graph execution.
$(B)/graph_replay: tests/cuda/graph_replay.c | $(B)
	@test "$(CUDA_HDR)" = real || { echo "graph-workload requires real CUDA headers (set CUDA_HOME)" >&2; exit 1; }
	$(CC) -O2 -g -std=gnu11 -Wall -Wextra -Werror -I$(CUDA_HOME)/include -o $@ $< -ldl

graph-workload: $(B)/graph_replay

test-graphs: all graph-workload
	@test -n "$(GRAPH_OUTPUT)" || { echo "set GRAPH_OUTPUT to a new evidence directory" >&2; exit 1; }
	python3 scripts/graph-replay.py --output "$(GRAPH_OUTPUT)"

bench: all
	bash bench/bench.sh

clean:
	rm -rf $(B)

.PHONY: all test bench clean graph-workload test-graphs
