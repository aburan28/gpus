# Data path and performance options

Every option here changes how bytes move, never what is stored: all modes
produce the same chunk names (plain SHA-256 of the chunk bytes) and the same
manifests, so snapshots taken one way restore any other way. Each option
reports what it actually obtained (`io.*=` lines), because several of them
are requests the OS or driver can refuse.

## Where the bytes go

For one checkpoint of an image of `S` bytes in `N` chunks, of which a
fraction `f` is not already in the store:

| Mode | Device → host | Hashed on the CPU | Written to the store |
|------|---------------|-------------------|----------------------|
| default | `S` | `S` | `f·S`, from host memory |
| `--gpu-hash on` | `32·N` digests + `f·S` + the cross-check sample | the sample only (8 chunks by default) | `f·S`, from host memory |
| `--gpu-hash on --gds on` | `32·N` digests (+ the sample) | the sample only | `f·S`, device → file via cuFileWrite |

For one restore, the full image always goes back into GPU memory, because
the checkpoint released it (the plan's performance constraint):

| Mode | Host → device | Verification |
|------|---------------|--------------|
| default | `S` | every chunk re-hashed on the CPU before it is copied |
| `--gpu-hash on` | `S` | the restored image is hashed on the device; any chunk it flags is copied back and re-hashed on the CPU before the restore is failed |
| `--gpu-hash on --gds on` | 0 | file → device via cuFileRead, then device verification as above |

The test suite proves these rows against the mock driver's own copy
counters, for example that a direct checkpoint moves exactly `32·N` bytes
device→host and a direct restore moves zero bytes host→device.

## Options

### Staging memory: `--hostmem malloc|thp|hugetlb|hugetlb1g` (default `thp`)

One arena of `threads × pipeline × chunk_size` bytes holds every staging
buffer. It is pre-faulted when created, so first-touch page faults happen
before the checkpoint rather than inside it.

- `thp`: anonymous mmap aligned to 2 MiB with `madvise(MADV_HUGEPAGE)`.
  Works when `/sys/kernel/mm/transparent_hugepage/enabled` is `always` or
  `madvise`. Coverage is measured from `/proc/self/smaps` and reported as
  `N of M kB in huge pages`; THP is a request, not a guarantee.
- `hugetlb`: `MAP_HUGETLB` with 2 MiB pages. Needs pages reserved with
  `sysctl vm.nr_hugepages=N`. Falls back to `thp` and says why.
- `hugetlb1g`: 1 GiB pages. Needs `hugepagesz=1G hugepages=N` on the kernel
  command line. Falls back to 2 MiB, then `thp`.
- `malloc`: page-aligned heap memory with no huge-page request. Under
  THP `always` the kernel may still back it with huge pages; the report
  shows what was measured.

Huge pages cut the number of page-table and IOMMU entries the DMA engine
walks, and make pinning cheaper.

### Locking: `--mlock`

`mlock(2)` on the arena: no swap-out, no faults mid-copy. It needs
`CAP_IPC_LOCK` or a large enough `ulimit -l`; failure is reported. On the
CUDA path, `--pin register` already page-locks the arena, so `--mlock` adds
nothing there. It matters for the file backend and when pinning fails.

### CUDA pinning: `--pin register|alloc|none` (default `register`)

- `register`: `cuMemHostRegister(PORTABLE)` on our arena. DMA engines copy
  straight into it, and it keeps the huge-page backing chosen above.
- `alloc`: the driver allocates the arena with `cuMemHostAlloc`.
  `--hostmem` is ignored, because the driver picks the pages.
- `none`: pageable memory. The driver bounces each copy through its own
  pinned buffer.

Registration cost grows with size. The CLI therefore allocates, pre-faults
and pins the arena **before it locks the target**, reports the time as
`ns_staging_setup`, and reuses the same arena for the `--resume` refill. The
`io.hostmem` line ends in `[prepared before the pause]` when that happened.

### Copy overlap: `--pipeline 1-4` (default 2)

Each worker owns `pipeline` slots, each with its own CUDA stream. While one
slot's chunk is hashed and stored, the next chunk's `cuMemcpyDtoHAsync` is
already running into the other slot; restore overlaps `cuMemcpyHtoDAsync`
with fetching the next chunk the same way. With pipelining, `ns_read_sum`
counts only the time workers waited on copies, so copy time hidden behind
hashing does not appear in it. Backends with synchronous copies, such as
the file backend, clamp to 1 and say so.

### Device hashing: `--gpu-hash on` (default off)

A SHA-256 kernel (`src/kernels/sha256_chunks.cu`, one thread per chunk)
hashes the custom-storage mapping in place. Only the 32-byte digests cross
PCIe. Chunks the store already has are then never copied at all, and only
new chunks are staged. The kernel is compiled at runtime by NVRTC for the
oldest compute capability present (`GPUCKPT_LIBNVRTC`, else
`libnvrtc.so.13`, `.12`, `libnvrtc.so`). Alternatively, `make ptx` builds
PTX with nvcc, loaded with `GPUCKPT_KERNEL_PTX=build/sha256_chunks.ptx`.

The same source compiles as C. `tests/kernel_test.c` checks it against
OpenSSL on every padding boundary, and the mock driver runs it for every
launch, so the arithmetic the GPU runs is tested here. GPU and CPU digests
are bit-identical, so the store stays CPU-verifiable.

Safeguards:

- `--gpu-hash-check N` (default 8): before trusting the digests, N
  evenly spaced chunks are copied back and re-hashed on the CPU. A
  mismatch turns device hashing off and the snapshot is re-hashed on the
  CPU. With `0`, a wrong kernel would store chunks under wrong names. The
  test suite shows `gpuckpt verify` catching exactly that.
- A kernel fault falls back to CPU hashing, and the adapter never launches
  the kernel again in that process.

Why it is off by default: whether a kernel may read the custom-storage
mapping is gate 7 (`docs/gates.md`). If it cannot, the fault is sticky for
the helper's CUDA context and may prevent completing the checkpoint. Try it
first on a process you can afford to lose.

### GPUDirect Storage: `--gds on` (default off)

New chunks go from the mapping to their chunk file with `cuFileWrite`.
Restore goes back with `cuFileRead`. No staging buffer is allocated. Files
are opened `O_DIRECT` when the filesystem allows it, and the report says
which. The mapping is registered with `cuFileBufRegister` and deregistered
before `cuCheckpointOperationComplete` unmaps it. If registration fails,
cuFile uses its own GPU-side bounce buffers.

Requirements, each reported when unmet:

- `--gpu-hash on`, because a chunk must be hashed before it can be named,
  and without a host copy only the GPU can hash it.
- A local repository. For S3, the network stack is in host memory, so run
  the checkpoint against a local repo and sync to S3 after the target has
  resumed (`docs/s3.md`).
- libcufile and a working `cuFileDriverOpen` (`GPUCKPT_LIBCUFILE` to
  override).

cuFile silently uses "compatibility mode" (POSIX I/O through host bounce
buffers) when nvidia-fs or the filesystem does not support the direct path.
This build cannot detect that. Check `cufile.json` and the cufile log
before attributing a speedup to GDS. Each chunk is one file with one
registered handle, so per-chunk overhead is significant at 1 MiB. Use
`init --chunk-size` of 16 to 64 MiB for GDS repositories.

## What has been measured, and what has not

Measured on this container's CPU with the file backend: the storage
layer's hash, dedup and write costs (README, "Benchmark"), and THP backing
of the staging arena (4096 of 4096 kB in the test run).

Not measured anywhere, because no GPU has been available: copy bandwidth
with and without pinning, overlap gained by `--pipeline`, kernel hashing
throughput, cuFile throughput, and pause-time reduction. The CLI prints
every timing needed to measure them: `ns_staging_setup`, `ns_lock`,
`ns_map`, `ns_copy`, `ns_gpu_hash`, `ns_complete`, and the restore
equivalents. Gate 5 in `docs/gates.md` is where those measurements belong.
