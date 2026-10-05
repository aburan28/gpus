#!/usr/bin/env bash
# CPU-only benchmark of the storage layer through the host-file backend.
#
# What it measures: hash + dedup + store-write cost of snapshotting an image
# of SIZE_MIB at several change rates, and restore throughput. Every number
# is for THIS machine's CPU and disk and says nothing about GPU copy
# bandwidth, cuCheckpointProcessCheckpoint mapping time, or the driver's
# refill on restore. The CUDA path adds those on top (see docs/design.md
# "What the benchmark does not measure").
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/gpuckpt
SIZE_MIB=${SIZE_MIB:-256}
CHUNK=${CHUNK:-1048576}
THREADS=${THREADS:-0}
OUT=bench/out; rm -rf "$OUT"; mkdir -p "$OUT"
img="$OUT/img.bin"
python3 - "$img" "$SIZE_MIB" <<'PY'
import sys, random
p, mib = sys.argv[1], int(sys.argv[2])
r = random.Random(1)
with open(p, "wb") as f:
    for _ in range(mib): f.write(r.randbytes(1 << 20))
PY
$BIN init --repo "$OUT/repo" --chunk-size "$CHUNK" >/dev/null
ns() { grep "^$2=" "$1" | cut -d= -f2; }
ms() { awk -v v="$(ns "$1" "$2")" 'BEGIN{printf "%.1f", v/1e6}'; }
printf "%-10s %10s %10s %10s %10s %10s %10s %10s\n" change_pct bytes_new wall_ms read_ms hash_ms write_ms MiB_s threads
prev=""
for pct in 100 0 1 5 10 25 50; do
    if [ "$pct" != 100 ]; then
        python3 - "$img" "$CHUNK" "$pct" "$SIZE_MIB" <<'PY'
import sys, random
p, cs, pct, mib = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
n = (mib << 20) // cs
k = n * pct // 100
r = random.Random(pct)
idx = r.sample(range(n), k)
with open(p, "r+b") as f:
    for i in idx:
        f.seek(i * cs); f.write(r.randbytes(cs))
PY
    fi
    o="$OUT/snap_$pct.out"
    args=(--repo "$OUT/repo" --input "$img" --id "c$pct" --threads "$THREADS")
    [ -n "$prev" ] && args+=(--parent "$prev")
    $BIN snapshot-file "${args[@]}" > "$o"
    prev="c$pct"
    printf "%-10s %10s %10.1f %10.1f %10.1f %10.1f %10s %10s\n" "$pct" "$(ns $o bytes_new)" \
        "$(ms $o ns_wall)" "$(ms $o ns_read_sum)" "$(ms $o ns_hash_sum)" "$(ms $o ns_write_sum)" \
        "$(ns $o throughput_MiB_s)" "$(ns $o threads)"
done
echo
echo "restore (full image refill from store):"
o="$OUT/restore.out"
$BIN restore-file --repo "$OUT/repo" --snapshot c50 --output "$OUT/restored.bin" --threads "$THREADS" > "$o"
printf "  wall_ms=%.1f read_ms=%.1f hash_ms=%.1f write_ms=%.1f MiB_s=%s threads=%s\n" \
    "$(ms $o ns_wall)" "$(ms $o ns_read_sum)" "$(ms $o ns_hash_sum)" "$(ms $o ns_write_sum)" \
    "$(ns $o throughput_MiB_s)" "$(ns $o threads)"
cmp "$OUT/restored.bin" "$img" && echo "  restored image byte-identical: yes"
echo
echo "store: $(find "$OUT/repo/chunks" -type f | wc -l) chunk files, $(du -sh "$OUT/repo/chunks" | cut -f1) for $(ls "$OUT/repo/snapshots" | wc -l) snapshots of a ${SIZE_MIB} MiB image"
echo "(image regenerated per row; *_ms columns for read/hash/write are summed over threads)"
