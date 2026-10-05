# gpuckpt

Incremental, deduplicated storage for GPU process checkpoints, built on the
CUDA driver's checkpoint custom-storage interface (CUDA 13.x,
`cuCheckpointProcessCheckpoint` with `customStorageInfo_out`).

A helper process locks a running CUDA process, has the driver map its GPU
memory into the helper, hashes the image in fixed-size chunks, writes only
chunks the store does not already hold, records a complete per-snapshot
manifest, and completes the operation. Restore refills a checkpointed
process's GPU memory from any snapshot, re-verifying every chunk's hash on
the way in.

**Status: storage layer and control flow implemented and tested on CPU.
No GPU run has been made.** The CUDA adapter compiles against the real
`cuda.h` when a toolkit is present and against a documentation-derived
stand-in otherwise; the test suite exercises the full lock → checkpoint →
copy → complete → restore → complete → unlock sequence through a mock
driver. The open hardware questions are tracked in [docs/gates.md](docs/gates.md).

## Build and test

```sh
make                 # build/gpuckpt, build/libgpuckpt.a, build/libcuda_mock.so
make test            # 77 CPU-only checks (host files + mock driver)
make bench           # synthetic storage-layer benchmark
CUDA_HOME=/usr/local/cuda make   # on a CUDA host: real header, binary reports cuda-header: real
```

Dependencies: a C11 compiler, make, python3 and bash for tests. OpenSSL's
libcrypto is used for SHA-256 when `pkg-config` finds it (hardware SHA
extensions), otherwise a builtin implementation is compiled in.

`gpuckpt version` prints which hash implementation and which CUDA header
the binary was built with. A binary showing `cuda-header: mock (UNVERIFIED)`
must not be used against a real driver.

## Usage

```sh
gpuckpt init --repo /ckpt/store --chunk-size 1048576

# Checkpoint a running CUDA process and let it continue (round trip in place)
gpuckpt snapshot --repo /ckpt/store --pid 12345 --resume --note "epoch 12"

# Checkpoint and leave it CHECKPOINTED (GPU released), e.g. before a CRIU dump
gpuckpt snapshot --repo /ckpt/store --pid 12345 --parent 20261005T221530Z-3fa9c1d2

# Refill a CHECKPOINTED process from any snapshot and unlock it
gpuckpt restore --repo /ckpt/store --pid 12345 --snapshot 20261005T221530Z-3fa9c1d2

gpuckpt list   --repo /ckpt/store
gpuckpt show   --repo /ckpt/store --snapshot ID [--chunks]
gpuckpt verify --repo /ckpt/store            # re-hash every chunk of every snapshot
gpuckpt forget --repo /ckpt/store --snapshot ID
gpuckpt gc     --repo /ckpt/store            # delete chunks no manifest references
gpuckpt state  --pid 12345                   # RUNNING / LOCKED / CHECKPOINTING / ...
```

The same store works on plain files, which is how the tests and benchmark
run and how an image already dumped to disk can be deduplicated:

```sh
gpuckpt snapshot-file --repo R --input dev0.bin,dev1.bin
gpuckpt restore-file  --repo R --snapshot ID --output out0.bin,out1.bin
```

Every command prints `key=value` statistics: bytes and chunks seen, chunks
and bytes newly stored, chunks identical to the parent at the same offset,
and per-phase timings (lock, map, copy with read/hash/write split, complete,
restore). Those are the plan's step-5 measurements, reported separately.

Exit codes: 0 ok, 1 usage, 2 the operation failed cleanly, 3 the target
was left in an unexpected driver state (details on stderr).

## Repository layout

```
include/gpuckpt.h        public API: repo, image, snapshot, manifest, cuda adapter
src/store.c              content-addressed chunk store (write-once, link()-published)
src/manifest.c           self-checked text manifests
src/snapshot.c           chunk-parallel create / restore / verify / gc, repo locking
src/image_file.c         host-file image backend
src/cuda_backend.c       CUDA checkpoint adapter (dlopen'ed libcuda, custom storage)
src/cli.c                command-line front end
tests/run.sh             test suite
tests/mock/              documentation-derived cuda.h and a mock libcuda
bench/bench.sh           storage-layer benchmark
scripts/criu-*.sh        EXPERIMENTAL CRIU orchestration (gate 3, unverified)
docs/design.md           data model, control flow, failure policy, build modes
docs/gates.md            hardware questions still open, and what the code assumes
```

## Design in brief

- **Deduplicated, not chained.** A manifest lists every chunk of the image;
  only chunks absent from the store are written. Restoring needs that
  manifest and the chunks it names. Forgetting a snapshot never breaks
  another; `gc` reclaims exactly the unreferenced chunks.
- **Content is identity.** Chunk files are named by SHA-256 of their bytes
  and published with `link(2)`, so N concurrent writers of the same content
  produce one file and one "new" count. Readers re-hash on restore and
  verify; manifests carry a trailer hash over their own text.
- **Immutable records.** Chunk files and manifests are never rewritten.
  `init` refuses an existing repo; `snapshot --id` refuses an existing id.
- **Never complete a failed copy.** If the copy phase fails, the driver
  operation is left open and the CLI exits 3 with the target state, because
  completing a checkpoint releases GPU memory. `--complete-on-error` opts
  into the alternative explicitly.
- **Runtime-bound driver.** `libcuda.so.1` is `dlopen`ed and every symbol
  resolved by name (override with `GPUCKPT_LIBCUDA`), so a driver without
  the checkpoint API fails with the missing symbol's name, and the mock can
  stand in for tests.

See [docs/design.md](docs/design.md).

## Benchmark (CPU-only, synthetic)

`make bench` on this development container, 4 vCPUs, 256 MiB random image,
1 MiB chunks, OpenSSL SHA-256, container disk. These numbers cover hashing,
dedup lookup and store writes only. They do not include GPU-to-host copy,
driver mapping time, or the driver's refill on restore, which the CUDA path
adds on top and reports per run.

```
change_pct  bytes_new    wall_ms    read_ms    hash_ms   write_ms      MiB_s
100         268435456      332.5       36.5      207.0     1077.0      770.0
0                   0       58.2       33.2      194.3        1.4     4400.4
1             2097152       58.8       35.9      183.1        5.3     4350.4
5            12582912       63.2       33.4      185.8       30.1     4049.9
10           26214400       77.2       37.3      185.4       80.4     3314.0
25           67108864      218.9       35.2      199.5      632.6     1169.3
50          134217728      307.2       34.6      193.6      992.3      833.2

restore: wall_ms=952.6  (store read + verify + file write, byte-identical)
```

Read/hash/write columns are summed over the 4 worker threads. The hash
cost is flat (the full image is always read and hashed, as the plan
states); the write cost scales with the changed fraction. At low change
rates the storage layer runs at about 4 GiB/s here, so on the GPU path the
copy bandwidth, not the hashing, is expected to bound capture time. That
expectation is untested.

## What is not done

- No run against a real driver, GPU, or CRIU install. Gates 1 to 6 in
  `docs/gates.md` are open.
- GPUDirect Storage, remote replication, application dirty tracking,
  retention policies beyond `forget` + `gc` (plan step 6).
- Multi-GPU behaviour of the custom-storage mapping is modelled only by the
  mock (two devices).
