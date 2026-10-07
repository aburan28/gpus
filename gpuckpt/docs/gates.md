# Implementation gates

The plan lists questions that must be answered on real hardware before the
design is trusted. This table tracks each one against the code in this
repository. "CPU-tested" means the test suite exercises the code path with
the mock driver; it never stands in for the GPU evidence the gate asks for.

| # | Question | Code status | Evidence status |
|---|----------|-------------|-----------------|
| 1 | Is there an exposed GPU dirty-page bitmap? | None used. Change detection is whole-image hashing (`src/snapshot.c`). The manifest records `chunks_same_as_parent` so a future tracking feature can be checked against ground truth. | Open. Inspect the pinned `cuda.h` on the target host. |
| 2 | Are image offsets stable across separate checkpoints? | Not relied on. Every manifest is positionally complete and dedup is by content hash, so a layout change costs bytes, never correctness. Positional inheritance is not implemented. | Open. Allocation-churn test: snapshot, allocate/free in the target, snapshot, compare `chunks_same_as_parent` to the expected unchanged fraction. |
| 3 | Does the installed CRIU plugin handle our custom backend? | `scripts/criu-dump.sh` / `criu-restore.sh` encode the intended sequence with both plugin modes selectable. | Open and UNVERIFIED. Requires a host with CRIU, the NVIDIA CRIU plugin, and a driver exposing the custom-storage API. |
| 4 | Can returned mappings be used directly with cuFile/GDS? | Implemented behind `--gds on` (`src/cuda_backend.c`): `cuFileBufRegister` on each mapping (deregistered before complete), `cuFileWrite`/`cuFileRead` per chunk file, O_DIRECT when available. Falls back to staging when cuFile is unavailable. | Open. Mock-tested only. Must also confirm the direct path rather than cuFile compatibility mode (cufile log), since this build cannot tell them apart. |
| 5 | Does a low change rate reduce pause time materially? | Timings are split: `ns_staging_setup` (outside the pause), `checkpoint.ns_lock`, `ns_map`, `ns_copy`, `ns_gpu_hash`, `ns_complete`, per-phase `ns_read/ns_hash/ns_write`, plus `restore.*`, and `bytes_staged`/`bytes_direct`. With `--gpu-hash on` only changed chunks are copied, so low change rates now shrink device-to-host traffic, not only store writes. | Open. Needs the same image checkpointed at several change rates in each mode on real hardware. |
| 6 | Can the application continue during export? | Yes, by construction: `--resume` restores and unlocks only after `cuCheckpointOperationComplete` has succeeded on a fully stored snapshot. No access to a mapping after complete (`cu_read` refuses). Export to remote storage is not implemented; it would operate on the committed store, never on a mapping. | Lifetime handling CPU-tested (`tests/run.sh`, fault-injection section). Concurrency on a real driver untested. |

| 7 | Can a kernel read the custom-storage mapping in place? | `--gpu-hash on` launches `gc_sha256_chunks` on each `devPtr`. The CPU cross-check (`--gpu-hash-check`) catches wrong digests, a launch or sync failure falls back to CPU hashing, and the kernel is not launched again in that process. Default off. | Open. The reference calls the pointer "zero-copy mapped device memory ... to copy to/from"; kernel access is not stated. A fault would be sticky for the helper's context and might block `cuCheckpointOperationComplete`, so the first trial must use a disposable target. |
| 8 | Do async copies from the mapping overlap with host work? | `--pipeline 2` (default) issues `cuMemcpyDtoHAsync` on per-worker streams into a `cuMemHostRegister`ed arena and waits per slot. | Open. Compare `ns_copy` with `--pipeline 1` and `2` on the same image. |
| 9 | Does pinning huge-page memory work and help? | `--hostmem thp` + `--pin register` (defaults) register a 2 MiB-aligned, THP-backed, pre-faulted arena before the pause. Coverage is reported from smaps. | THP backing confirmed in this container (4096 of 4096 kB). Registration and DMA speed on huge pages are unmeasured. |

## Additional facts established from the CUDA 13.4 reference

- `customStorageInfo_out` is a pointer-to-pointer. Whether the driver fills
  a caller-provided struct or replaces the pointer is not stated. The adapter
  passes a zeroed caller struct and reads back through the pointer, so both
  behaviours work; the mock tests both (`GPUCKPT_MOCK_ALLOC_INFO`).
- Per-device entries carry `devPtr`, `size`, `stream` and no device identity.
  The adapter asks `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)`
  and records the ordinal and UUID when that succeeds, `-1`/`-` otherwise.
  Whether the attribute query works on these mappings is itself untested.
- Custom storage requires `cuInit` in the helper and retained primary
  contexts; the adapter retains the primary context of every device.
- Checkpointing the helper's own process is unsupported; the CLI refuses
  `--pid $$`.
- `cuCheckpointOperationComplete` after a checkpoint releases GPU memory.
  The CLI therefore never completes a failed copy unless told to with
  `--complete-on-error`, and says why.

## What the mock cannot tell you

The mock headers' enum values and struct layouts (`tests/mock/cuda.h`,
`tests/mock/cufile/cufile.h`) are documentation-derived.
A binary built against them reports `cuda-header: mock (UNVERIFIED)` or
`cufile-header: mock (UNVERIFIED)`, and must not be run against a real
driver. Nothing in the test suite measures GPU bandwidth, mapping latency,
driver refill time, kernel throughput, cuFile throughput, or multi-GPU
behaviour. The mock runs asynchronous copies synchronously, so it cannot
catch a missing wait before a staging buffer is reused. The code waits on
every slot before reuse and on all slots before a worker exits, and that
ordering is reviewed, not tested.
