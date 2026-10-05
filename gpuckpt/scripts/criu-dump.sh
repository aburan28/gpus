#!/usr/bin/env bash
# EXPERIMENTAL, UNVERIFIED: full-process checkpoint = gpuckpt (GPU memory,
# custom storage) + CRIU (CPU state). Gate 3 in docs/gates.md. This script
# encodes the intended sequence; it has not been run against a real driver
# or a real CRIU + cuda plugin install, and the interaction between a
# custom-storage CHECKPOINTED state and the CRIU CUDA plugin's own
# cuda-checkpoint handling is exactly what the gate must establish.
#
# usage: criu-dump.sh <pid> <repo> <criu-images-dir> [snapshot-id]
set -euo pipefail
PID=$1; REPO=$2; IMG=$3; ID=${4:-}
GPUCKPT=${GPUCKPT:-$(dirname "$0")/../build/gpuckpt}
mkdir -p "$IMG"

# 1. GPU side: lock, checkpoint to the store, complete (target left CHECKPOINTED,
#    GPU resources released). No --resume: the process is about to be dumped.
args=(snapshot --repo "$REPO" --pid "$PID" --timeout-ms "${LOCK_TIMEOUT_MS:-30000}")
[ -n "$ID" ] && args+=(--id "$ID")
"$GPUCKPT" "${args[@]}" | tee "$IMG/gpuckpt-snapshot.txt"
SNAP=$(grep '^snapshot=' "$IMG/gpuckpt-snapshot.txt" | cut -d= -f2)
echo "$SNAP" > "$IMG/gpuckpt-snapshot-id"

# 2. CPU side. The CRIU CUDA plugin normally drives cuda-checkpoint itself;
#    with the GPU state already CHECKPOINTED by us, the plugin must either
#    detect that and skip its own toggle, or be disabled. Which of these holds
#    for the installed plugin version is gate 3. Both variants are kept here
#    behind CRIU_CUDA_MODE so the test can try each explicitly.
case "${CRIU_CUDA_MODE:-plugin}" in
  plugin)  criu dump -t "$PID" -D "$IMG" --shell-job --leave-running=false "${CRIU_EXTRA[@]:-}" ;;
  noplugin) criu dump -t "$PID" -D "$IMG" --shell-job --no-plugins "${CRIU_EXTRA[@]:-}" ;;
  *) echo "CRIU_CUDA_MODE must be plugin|noplugin" >&2; exit 1 ;;
esac
echo "dumped pid $PID: CRIU images in $IMG, GPU snapshot $SNAP in $REPO"
