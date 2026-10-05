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
| 4 | Can returned mappings be used directly with cuFile/GDS? | Not implemented. Reads go through `cuMemcpyDtoH` into pinned staging buffers. | Open. Plan step 6. |
| 5 | Does a low change rate reduce pause time materially? | Timings are split: `checkpoint.ns_lock`, `ns_map`, `ns_copy`, `ns_complete`, and per-phase `ns_read/ns_hash/ns_write`, plus `restore.*`. The CPU benchmark shows the storage-layer share only. | Open. The hash still reads the full image; whether that dominates the pause is a measurement on the real copy path. |
| 6 | Can the application continue during export? | Yes, by construction: `--resume` restores and unlocks only after `cuCheckpointOperationComplete` has succeeded on a fully stored snapshot. No access to a mapping after complete (`cu_read` refuses). Export to remote storage is not implemented; it would operate on the committed store, never on a mapping. | Lifetime handling CPU-tested (`tests/run.sh`, fault-injection section). Concurrency on a real driver untested. |

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

The mock header's enum values and struct layouts are documentation-derived.
A binary built against it reports `cuda-header: mock (UNVERIFIED)` and must
not be run against a real driver. Nothing in the test suite measures GPU
bandwidth, mapping latency, driver refill time, or multi-GPU behaviour.
