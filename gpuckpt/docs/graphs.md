# CUDA graph replay after checkpoint/restore

An application instantiates a graph once, checkpoints/restores through
`gpuckpt`, then replays the same executable with correct results.

| Requirement | Implementation | Evidence status |
|---|---|---|
| Stream-captured and explicit graphs | Kernel plus pinned-host copy in both modes | Hardware unverified |
| Same executable and allocation, no rebuilding | One instantiation; identities and outputs checked | Hardware unverified |
| Repeated `snapshot --resume` and separate `restore` | Alternating sequences; stored chunks verified | Hardware unverified |
| Output matches uninterrupted execution | Per-value oracle plus checksums and separate control | Hardware unverified |
| Full-process restart, including CPU graph state | Requires matching CRIU image | Existing gate 3 remains open |

The CUDA checkpoint driver owns CUDA object restoration. `gpuckpt` stores
and refills its mapped device-memory image. This change adds a real-driver
workload and correctness runner on the existing checkpoint/restore path;
it does not serialize graph handles into a manifest or rebuild graphs.
CPU-side handles, module and argument metadata must survive too. A GPU
snapshot alone is not a portable graph or a full process checkpoint. Older
GPU images must not be injected into incompatible CPU/graph state.

The fixture completes capture, uploads and warms the graph, then retains
it across each checkpoint. At each barrier the stream is synchronized.
A PTX kernel increments 4,096 device values and a graph node copies them to
pinned host memory. Every replay checks every value against a CPU oracle
before an independent device read. The Python runner also checks checksums,
object identities and one instantiation, and compares an uninterrupted
control. It restores only the snapshot just committed for that process.

The helper never captures its custom-storage copy streams. The CUDA
reference forbids capture on these streams when completing an operation;
that restriction does not establish target-graph compatibility.

On a Linux GPU host with a driver exposing custom-storage checkpoint APIs,
use real CUDA headers and the privileges required by the checkpoint driver:

```sh
cd gpuckpt
make clean
CUDA_HOME=/usr/local/cuda make all graph-workload
./build/gpuckpt version       # must report cuda-header: real
make test-graphs CUDA_HOME=/usr/local/cuda GRAPH_OUTPUT=/new/path/graph-evidence
```

No nvcc or CUDA Runtime library is needed: the fixture is C plus embedded
PTX, resolving the Driver API from `libcuda.so.1`. The output directory must
be new. To customize the run:

```sh
python3 scripts/graph-replay.py --output /new/path/graph-evidence \
  --cycles 10 --replays 100 --timeout 120 --modes both
```

Evidence includes raw stdout/stderr, input commands, exit codes, wall
times, graph events, source/binary hashes, platform and Git revision.
Events include device UUID and driver version. `report.json` is `passed`,
`failed` or `blocked`; exit 77 is unavailable prerequisites, never a pass.
Mock-header binaries and `GPUCKPT_LIBCUDA` overrides are rejected.
On failure the runner stops its own fixture and retains evidence/store.
It never requests `--complete-on-error` or unlocks an incomplete checkpoint.
Timeouts are per operation. Cycle wall time includes checkpoint, store
verification, protocol and replay; it is not graph throughput.

CPU tests exercise orchestration, and real-header CI compiles the workload.
Neither proves hardware replay. Device launches, graph updates, allocation
nodes, active capture, multi-GPU, migration and CRIU need further cases.

Primary API references:

- [CUDA 13.4 checkpointing](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__CHECKPOINT.html)
- [CUDA 13.4 graph management](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)
- [NVIDIA cuda-checkpoint lifecycle and CRIU](https://github.com/NVIDIA/cuda-checkpoint)
