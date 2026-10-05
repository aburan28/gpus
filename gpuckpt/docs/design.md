# Design

## Data model

```
<repo>/
  config                  format version, chunk size, hash algorithm
  lock                    flock: shared for snapshot/restore/verify, exclusive for gc
  chunks/<hh>/<sha256>    one file per distinct chunk content, write-once
  snapshots/<id>.manifest one file per snapshot, write-once, self-checked
  tmp/                    staging for atomic writes
```

A manifest lists, per device, the image size and the ordered SHA-256 of
every fixed-size chunk. The final chunk may be short. The manifest ends
with a SHA-256 trailer over its own text, so truncation or bit-rot is
detected before any chunk is read.

### Incremental vs. differential vs. deduplicated

| Mode | Newly stored data | Restoration depends on |
|------|-------------------|------------------------|
| Full | every chunk | this manifest only |
| Incremental (classic) | chunks changed since the previous snapshot | this manifest and the whole chain back to the last full |
| Differential (classic) | chunks changed since the last full | this manifest and that full |
| **Deduplicated (this implementation)** | chunks whose content is absent from the store | this manifest and the chunks it names, nothing else |

The deduplicated form gets the storage and transfer savings of an
incremental scheme without the chain dependency: forgetting any snapshot
never invalidates another, and `gc` reclaims exactly the chunks no
remaining manifest names. The parent id in a manifest is for reporting
(`chunks_same_as_parent`), never for reconstruction.

### Why fixed-size chunks

The per-device records the driver returns do not identify application
allocations, so there is no stable structure to align chunks to. Fixed
chunks make the chunk list positionally complete and the hash work
embarrassingly parallel. Content-defined chunking would help only if data
shifts within the image, which gate 2 in `docs/gates.md` has not shown.
Chunk size is set at `init` and fixed for the repository's life; 1 MiB is
the default.

## Control flow (CUDA)

```
gpuckpt snapshot --pid P [--resume]
  state(P)                       must be RUNNING or LOCKED
  lock(P, timeout)               if RUNNING
  checkpoint(P, custom)          LOCKED -> CHECKPOINTING; GPU memory mapped into us
  for each chunk, N threads:     cuMemcpyDtoH -> sha256 -> store_put if absent
  manifest_save                  only after every chunk is in the store
  complete(handle)               CHECKPOINTING -> CHECKPOINTED; GPU memory released
  [--resume]
    restore(P, custom)           CHECKPOINTED -> RESTORING; fresh GPU memory mapped
    for each chunk, N threads:   store_get -> sha256 verify -> cuMemcpyHtoD
    complete(handle)             RESTORING -> LOCKED
    unlock(P)                    LOCKED -> RUNNING
```

`gpuckpt restore --pid P --snapshot ID` is the second half on its own.

### Failure policy

A failure between `checkpoint` and `complete` leaves the target
CHECKPOINTING with its memory still mapped. Completing at that point would
release the memory without a stored copy, so the CLI does not; it reports
the state and exits 3. `--complete-on-error` overrides this for operators
who prefer a dead-but-unstuck target. The same holds for a failure during
restore (target left RESTORING). The store is never left inconsistent:
chunks are written before the manifest, and an orphaned chunk is reclaimed
by `gc`.

### Timings recorded

Checkpoint: `ns_lock`, `ns_map` (the checkpoint call itself), `ns_copy`
(the whole copy/hash/store phase), `ns_complete`; inside the copy phase,
`ns_read`, `ns_hash`, `ns_write` summed over threads, and `ns_wall`.
Restore: `ns_copy`, `ns_complete`, and the same inner split. These answer
the plan's step 5 separately: storage savings (`bytes_new`), capture time,
full pause time (lock to unlock with `--resume`), and recovery time.

## Build modes

The adapter needs `<cuda.h>` only for struct layouts and enum values. With
`CUDA_HOME` pointing at a toolkit, the real header is used and
`gpuckpt version` prints `cuda-header: real`. Without one, the
documentation-derived `tests/mock/cuda.h` is used so the code compiles and
the mock-driver tests run; that binary prints `cuda-header: mock
(UNVERIFIED)` and must never be pointed at a real `libcuda.so.1`.

The driver library itself is always loaded at runtime with `dlopen`, every
symbol resolved by name. A driver without the checkpoint symbols fails
with a message naming the symbol, which is the "symbol resolution must pass
on the intended host" check the plan asks for.

## What the benchmark does not measure

`make bench` runs the storage layer over a host file. It shows how the
hash/dedup/write cost scales with change rate on this CPU. It does not
include GPU-to-host copy bandwidth, the driver's mapping time, pinned
memory effects, or the driver's own work in `complete`. On the CUDA path
these dominate for large images and are reported by the CLI per run.

## Not implemented (plan step 6)

GPUDirect Storage, remote replication, application-assisted dirty
tracking, and multi-snapshot retention policies beyond `forget` + `gc`.
