# Local validation, 2026-10-06

| Check | Observed result | Raw evidence |
|---|---|---|
| Storage and mock-driver regression | 77 passed, 0 failed; S3 unavailable locally | `cpu-tests.log` |
| Graph-runner CPU protocol checks | 12 passed | `cpu-tests.log` |
| Real-header adapter and graph workload build | Exit 0, NVIDIA CUDA 13.4.92 package headers, graph build uses `-Werror` | `real-header-build.log` |
| AddressSanitizer and UndefinedBehaviorSanitizer | 77 passed, 0 failed, leak detection disabled | `asan-ubsan.log` |
| LeakSanitizer | Blocked: cannot inspect threads in this process-traced environment | `leak-sanitizer-blocked.log` |
| Real-driver graph replay | Exit 77, blocked: `libcuda.so.1` absent; no CUDA graphs executed | `hardware-attempt/report.json`, `hardware-attempt/capture-control/workload.stderr` |

The original regression attempt failed in the existing mock checkpoint
sequence (`initial-regression-failure.log`). The same suite passed in a
fresh directory. Tests now create a unique state directory under `build/`
to isolate repeated runs; the final regular and sanitizer runs both passed.
The earlier failure is retained, not counted as a passing run. Its exact
underlying stale-state cause was not established.

Reproduction from `gpuckpt/` (replace the real-header path for your host):

```sh
make clean
PKG_CONFIG=/bin/false make test
make clean
PKG_CONFIG=/bin/false make CFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer'
ASAN_OPTIONS=detect_leaks=0 bash tests/run.sh
make clean
PKG_CONFIG=/bin/false make all graph-workload CUDA_HOME=/path/to/nvidia/cu13
python3 scripts/graph-replay.py --output /new/evidence/directory
```

Hardware replay, CRIU restart, migration and multi-GPU graph behavior remain
unverified. No performance improvement is claimed. The attempted run records
the base revision, dirty worktree and SHA-256 hashes of the exact new sources
and compiled binaries; compilation is not hardware execution evidence.
