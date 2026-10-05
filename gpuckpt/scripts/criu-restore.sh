#!/usr/bin/env bash
# EXPERIMENTAL, UNVERIFIED counterpart of criu-dump.sh. See that file.
# usage: criu-restore.sh <repo> <criu-images-dir>
set -euo pipefail
REPO=$1; IMG=$2
GPUCKPT=${GPUCKPT:-$(dirname "$0")/../build/gpuckpt}
SNAP=$(cat "$IMG/gpuckpt-snapshot-id")

# 1. CPU side: restore the process. It comes back with the driver in the
#    CHECKPOINTED state (persistence mode or an earlier cuInit in the helper
#    is required by the restore API). The new pid is written by --pidfile.
criu restore -D "$IMG" --shell-job --restore-detached --pidfile "$IMG/restored.pid" "${CRIU_EXTRA[@]:-}"
NEWPID=$(cat "$IMG/restored.pid")

# 2. GPU side: refill GPU memory from the store, complete, unlock.
"$GPUCKPT" state --pid "$NEWPID"
"$GPUCKPT" restore --repo "$REPO" --pid "$NEWPID" --snapshot "$SNAP"
echo "restored pid $NEWPID from GPU snapshot $SNAP"
