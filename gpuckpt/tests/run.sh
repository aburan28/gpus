#!/usr/bin/env bash
# CPU-only test suite. Exercises the storage layer through the host-file
# backend and the CUDA adapter through the mock driver. Needs: bash, python3,
# cmp. Run via `make test`.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/gpuckpt
MOCK=./build/libcuda_mock.so
# Isolate repeated runs (including mock process state and fault injection).
# Keep evidence until make clean, rather than reusing one mutable directory.
T=$(mktemp -d ./build/tmp-test.XXXXXX)
echo "test_directory=$T"
CS=$((1<<20))
pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  ok   $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL $1"; }
check(){ if "$@" >/dev/null 2>&1; then ok "$*"; else bad "$*"; fi; }
expect_fail(){ if "$@" >/dev/null 2>&1; then bad "(expected failure) $*"; else ok "(fails as expected) $*"; fi; }
stat_of(){ grep "^$2=" "$1" | cut -d= -f2; }

echo "== build info"; $BIN version

echo "== sha256 known vector"
python3 - "$T" <<'PY'
import sys, hashlib, os
t=sys.argv[1]
open(f"{t}/abc.bin","wb").write(b"abc")
PY
$BIN init --repo "$T/r0" --chunk-size 4096 >/dev/null
$BIN snapshot-file --repo "$T/r0" --input "$T/abc.bin" --id abc >/dev/null
h=$($BIN show --repo "$T/r0" --snapshot abc --chunks | awk '/^c 0 0 /{print $4}')
[ "$h" = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ] && ok "sha256(abc) matches FIPS vector" || bad "sha256(abc)=$h"

echo "== generate images (dev0: 40 MiB structured, dev1: 5 MiB + 123 B tail)"
python3 - "$T" $CS <<'PY'
import sys, os, random, hashlib
t, cs = sys.argv[1], int(sys.argv[2])
rnd = random.Random(7)
# dev0: 40 chunks; chunks 3,4,5 are identical (zeros) and 10,11 are identical random -> 36 unique
blocks = []
z = bytes(cs); dup = rnd.randbytes(cs)
for i in range(40):
    if i in (3,4,5): blocks.append(z)
    elif i in (10,11): blocks.append(dup)
    else: blocks.append(rnd.randbytes(cs))
open(f"{t}/dev0.img","wb").write(b"".join(blocks))
# dev1: 5 full chunks + 123-byte tail, all unique, last chunk random
d1 = rnd.randbytes(5*cs + 123)
open(f"{t}/dev1.img","wb").write(d1)
uniq = set(hashlib.sha256(b).digest() for b in blocks)
for i in range(0, len(d1), cs): uniq.add(hashlib.sha256(d1[i:i+cs]).digest())
open(f"{t}/expect_unique","w").write(str(len(uniq)))
PY
cp "$T/dev0.img" "$T/dev0.orig"; cp "$T/dev1.img" "$T/dev1.orig"
EXP_UNIQ=$(cat "$T/expect_unique")

echo "== repo init / immutability"
check $BIN init --repo "$T/repo" --chunk-size $CS
expect_fail $BIN init --repo "$T/repo" --chunk-size $CS
expect_fail $BIN init --repo "$T/badcs" --chunk-size 1000

echo "== first snapshot: dedup within image"
$BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img,$T/dev1.img" --id s1 --note "first" > "$T/s1.out"
[ "$(stat_of $T/s1.out chunks_total)" = "46" ] && ok "chunks_total=46" || bad "chunks_total=$(stat_of $T/s1.out chunks_total)"
[ "$(stat_of $T/s1.out chunks_new)" = "$EXP_UNIQ" ] && ok "chunks_new=$EXP_UNIQ (unique content only)" || bad "chunks_new=$(stat_of $T/s1.out chunks_new) expected $EXP_UNIQ"
[ "$(stat_of $T/s1.out bytes_total)" = "$((40*CS + 5*CS + 123))" ] && ok "bytes_total exact" || bad "bytes_total"
[ "$(find $T/repo/chunks -type f | wc -l)" = "$EXP_UNIQ" ] && ok "store holds exactly $EXP_UNIQ chunk files" || bad "store file count"
expect_fail $BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img" --id s1

echo "== second snapshot after modifying 3 chunks of dev0: only 3 new"
python3 - "$T" $CS <<'PY'
import sys, random
t, cs = sys.argv[1], int(sys.argv[2])
rnd = random.Random(99)
with open(f"{t}/dev0.img","r+b") as f:
    for i in (0, 7, 39):
        f.seek(i*cs); f.write(rnd.randbytes(cs))
PY
$BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img,$T/dev1.img" --id s2 --parent s1 > "$T/s2.out"
[ "$(stat_of $T/s2.out chunks_new)" = "3" ] && ok "chunks_new=3" || bad "chunks_new=$(stat_of $T/s2.out chunks_new)"
[ "$(stat_of $T/s2.out bytes_new)" = "$((3*CS))" ] && ok "bytes_new=3 chunks" || bad "bytes_new"
[ "$(stat_of $T/s2.out chunks_same_as_parent)" = "43" ] && ok "chunks_same_as_parent=43" || bad "same_as_parent=$(stat_of $T/s2.out chunks_same_as_parent)"

echo "== third snapshot, no changes: zero new bytes"
$BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img,$T/dev1.img" --id s3 --parent s2 > "$T/s3.out"
[ "$(stat_of $T/s3.out chunks_new)" = "0" ] && ok "chunks_new=0" || bad "chunks_new"

echo "== restore to files and compare bytes"
$BIN restore-file --repo "$T/repo" --snapshot s1 --output "$T/r1_0,$T/r1_1" >/dev/null
check cmp "$T/r1_0" "$T/dev0.orig"
check cmp "$T/r1_1" "$T/dev1.orig"
$BIN restore-file --repo "$T/repo" --snapshot s2 --output "$T/r2_0,$T/r2_1" >/dev/null
check cmp "$T/r2_0" "$T/dev0.img"
check cmp "$T/r2_1" "$T/dev1.img"
expect_fail $BIN restore-file --repo "$T/repo" --snapshot s2 --output "$T/only_one"
expect_fail $BIN restore-file --repo "$T/repo" --snapshot nope --output "$T/x,$T/y"

echo "== list / show / verify"
[ "$($BIN list --repo "$T/repo" | tr '\n' ' ')" = "s1 s2 s3 " ] && ok "list sorted" || bad "list: $($BIN list --repo $T/repo | tr '\n' ' ')"
check $BIN show --repo "$T/repo" --snapshot s1
$BIN show --repo "$T/repo" --snapshot s1 | grep -q "note            first" && ok "note recorded" || bad "note"
check $BIN verify --repo "$T/repo"

echo "== manifest tamper: trailer catches it"
cp "$T/repo/snapshots/s3.manifest" "$T/s3.bak"
sed -i 's/^c 1 /c 1 /; 0,/^c 0 /s/^c 0 \(.\)/c 0 0/' "$T/repo/snapshots/s3.manifest"
expect_fail $BIN show --repo "$T/repo" --snapshot s3
head -c 2000 "$T/s3.bak" > "$T/repo/snapshots/s3.manifest"
expect_fail $BIN show --repo "$T/repo" --snapshot s3
cp "$T/s3.bak" "$T/repo/snapshots/s3.manifest"
check $BIN show --repo "$T/repo" --snapshot s3

echo "== forget + gc reclaims exactly the 3 chunks only s1 referenced"
check $BIN forget --repo "$T/repo" --snapshot s1
expect_fail $BIN forget --repo "$T/repo" --snapshot s1
$BIN gc --repo "$T/repo" > "$T/gc.out"
[ "$(stat_of $T/gc.out chunks_deleted)" = "3" ] && ok "gc deleted 3" || bad "gc deleted $(stat_of $T/gc.out chunks_deleted)"
[ "$(stat_of $T/gc.out bytes_freed)" = "$((3*CS))" ] && ok "gc freed 3 chunks of bytes" || bad "bytes_freed"
check $BIN verify --repo "$T/repo" --snapshot s2
expect_fail $BIN verify --repo "$T/repo" --snapshot s1

echo "== gc refuses to run with an unreadable manifest (unknown live set != empty)"
echo garbage > "$T/repo/snapshots/broken.manifest"
expect_fail $BIN gc --repo "$T/repo"
rm "$T/repo/snapshots/broken.manifest"
check $BIN gc --repo "$T/repo"

echo "== chunk corruption: verify reports, restore refuses"
victim=$(ls "$T"/repo/chunks/*/* | head -1)
cp "$victim" "$T/victim.bak"
printf 'X' | dd of="$victim" bs=1 seek=100 conv=notrunc status=none
$BIN verify --repo "$T/repo" --snapshot s2 > "$T/v.out" 2>&1 || true
grep -q "corrupt=1" "$T/v.out" && ok "verify counts 1 corrupt chunk" || bad "verify output: $(cat $T/v.out)"
expect_fail $BIN restore-file --repo "$T/repo" --snapshot s2 --output "$T/c0,$T/c1"
rm "$victim"
$BIN verify --repo "$T/repo" --snapshot s2 > "$T/v2.out" 2>&1 || true
grep -q "missing=1" "$T/v2.out" && ok "verify counts 1 missing chunk" || bad "verify output: $(cat $T/v2.out)"
cp "$T/victim.bak" "$victim"
check $BIN verify --repo "$T/repo" --snapshot s2

echo "== concurrent snapshots into one repo"
python3 -c "
import random; r=random.Random(3)
open('$T/cA.img','wb').write(r.randbytes(8<<20)); open('$T/cB.img','wb').write(r.randbytes(8<<20))"
$BIN snapshot-file --repo "$T/repo" --input "$T/cA.img" --id cA --threads 4 >/dev/null &
$BIN snapshot-file --repo "$T/repo" --input "$T/cB.img" --id cB --threads 4 >/dev/null &
wait
check $BIN verify --repo "$T/repo" --snapshot cA
check $BIN verify --repo "$T/repo" --snapshot cB

echo "== dedup race: 64 identical chunks, 8 threads -> exactly 1 new chunk, 10 rounds"
python3 -c "open('$T/same.img','wb').write(bytes([0xAB])*(64<<20))"
race_ok=1
for i in 1 2 3 4 5 6 7 8 9 10; do
    $BIN init --repo "$T/race$i" --chunk-size $CS >/dev/null
    n=$($BIN snapshot-file --repo "$T/race$i" --input "$T/same.img" --threads 8 | grep '^chunks_new=' | cut -d= -f2)
    [ "$n" = "1" ] || race_ok=0
done
[ $race_ok = 1 ] && ok "chunks_new=1 in all 10 rounds" || bad "dedup over-count under contention"

echo "== 1-thread and 8-thread snapshots produce identical manifests (minus id/time/stats)"
$BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img" --id t1 --threads 1 >/dev/null
$BIN snapshot-file --repo "$T/repo" --input "$T/dev0.img" --id t8 --threads 8 >/dev/null
d1=$($BIN show --repo "$T/repo" --snapshot t1 --chunks | grep '^c ' | sha256sum)
d8=$($BIN show --repo "$T/repo" --snapshot t8 --chunks | grep '^c ' | sha256sum)
[ "$d1" = "$d8" ] && ok "chunk lists identical" || bad "chunk lists differ"

# ---------------------------------------------------------------- mock CUDA
echo "== mock CUDA driver: full custom-storage lifecycle"
export GPUCKPT_LIBCUDA="$PWD/$MOCK"
export GPUCKPT_MOCK_STATE_DIR="$T/mock"
export GPUCKPT_MOCK_IMAGES="$T/dev0.orig,$T/dev1.orig"
PID=4242
$BIN init --repo "$T/crepo" --chunk-size $CS >/dev/null
$BIN state --pid $PID | grep -q "state=RUNNING" && ok "initial state RUNNING" || bad "initial state"
expect_fail $BIN restore --repo "$T/crepo" --pid $PID --snapshot nothing      # RUNNING: wrong state
expect_fail $BIN unlock --pid $PID                                            # RUNNING: illegal
check $BIN restore-tid --pid $PID

echo "-- snapshot --resume: checkpoint, restore in place, unlock"
$BIN snapshot --repo "$T/crepo" --pid $PID --id g1 --resume > "$T/g1.out" 2>&1 || { bad "snapshot --resume exit $?"; cat "$T/g1.out"; }
grep -q "state=RUNNING" "$T/g1.out" && ok "target RUNNING after resume" || bad "state after resume: $(tail -3 $T/g1.out)"
[ "$(stat_of $T/g1.out checkpoint.chunks_total)" = "46" ] && ok "46 chunks captured through the adapter" || bad "chunks: $(stat_of $T/g1.out checkpoint.chunks_total)"
grep -q "uuid=GPU-a0000000-0000-0000-0000-000000000000" "$T/g1.out" && ok "device 0 UUID recorded via pointer->ordinal lookup" || bad "uuid: $(grep uuid $T/g1.out)"
grep -q "uuid=GPU-a0000000-0000-0000-0000-000000000001" "$T/g1.out" && ok "device 1 UUID recorded" || bad "uuid dev1"
check cmp "$T/mock/$PID.mem.0" "$T/dev0.orig"
check cmp "$T/mock/$PID.mem.1" "$T/dev1.orig"
$BIN restore-file --repo "$T/crepo" --snapshot g1 --output "$T/g1_0,$T/g1_1" >/dev/null
check cmp "$T/g1_0" "$T/dev0.orig"
check cmp "$T/g1_1" "$T/dev1.orig"

echo "-- target keeps running and changes memory; next snapshot stores only the delta"
python3 - "$T/mock/$PID.mem.0" $CS <<'PY'
import sys, random
p, cs = sys.argv[1], int(sys.argv[2])
with open(p, "r+b") as f:
    f.seek(2*cs); f.write(random.Random(5).randbytes(cs))
PY
cp "$T/mock/$PID.mem.0" "$T/dev0.v2"
$BIN snapshot --repo "$T/crepo" --pid $PID --id g2 --parent g1 > "$T/g2.out" 2>&1 || bad "snapshot g2 exit $?"
[ "$(stat_of $T/g2.out checkpoint.chunks_new)" = "1" ] && ok "1 new chunk after 1 chunk changed" || bad "g2 chunks_new=$(stat_of $T/g2.out checkpoint.chunks_new)"
grep -q "state=CHECKPOINTED" "$T/g2.out" && ok "target CHECKPOINTED (no --resume)" || bad "state: $(tail -1 $T/g2.out)"
[ ! -e "$T/mock/$PID.mem.0" ] && ok "mock released GPU memory on complete" || bad "mem file still present"
expect_fail $BIN snapshot --repo "$T/crepo" --pid $PID            # CHECKPOINTED: cannot snapshot again
expect_fail $BIN lock --pid $PID

echo "-- restore the earlier snapshot g1 into the checkpointed target"
$BIN restore --repo "$T/crepo" --pid $PID --snapshot g1 > "$T/rg1.out" 2>&1 || bad "restore exit $?"
grep -q "state=RUNNING" "$T/rg1.out" && ok "RUNNING after restore+unlock" || bad "state: $(tail -1 $T/rg1.out)"
check cmp "$T/mock/$PID.mem.0" "$T/dev0.orig"
check cmp "$T/mock/$PID.mem.1" "$T/dev1.orig"

echo "-- restore g2 with --no-unlock leaves target LOCKED; unlock separately"
$BIN snapshot --repo "$T/crepo" --pid $PID --id g3 >/dev/null 2>&1
$BIN restore --repo "$T/crepo" --pid $PID --snapshot g2 --no-unlock > "$T/rg2.out" 2>&1 || bad "restore g2 exit $?"
grep -q "state=LOCKED" "$T/rg2.out" && ok "LOCKED with --no-unlock" || bad "state: $(tail -1 $T/rg2.out)"
check cmp "$T/mock/$PID.mem.0" "$T/dev0.v2"
check $BIN unlock --pid $PID
$BIN state --pid $PID | grep -q RUNNING && ok "RUNNING after unlock" || bad "unlock"

echo "-- lock on an already LOCKED target is accepted by snapshot (no double lock)"
check $BIN lock --pid $PID --timeout-ms 100
$BIN snapshot --repo "$T/crepo" --pid $PID --id g4 --resume >/dev/null 2>&1 && ok "snapshot from LOCKED" || bad "snapshot from LOCKED"

echo "-- driver allocates the info struct (alternate reading of the reference)"
GPUCKPT_MOCK_ALLOC_INFO=1 $BIN snapshot --repo "$T/crepo" --pid $PID --id g5 --resume > "$T/g5.out" 2>&1 && ok "adapter handles driver-allocated info" || { bad "alloc-info variant"; tail -3 "$T/g5.out"; }

echo "-- fault injection"
GPUCKPT_MOCK_LOCK_TIMEOUT=1 $BIN snapshot --repo "$T/crepo" --pid $PID --id g6 > "$T/g6.out" 2>&1 && bad "lock timeout not reported" || ok "lock timeout -> exit $?"
grep -q "not locked within" "$T/g6.out" && ok "timeout message names the timeout" || bad "msg: $(cat $T/g6.out)"
$BIN state --pid $PID | grep -q RUNNING && ok "still RUNNING after lock timeout" || bad "state after timeout"
rc=0; GPUCKPT_MOCK_FAIL_COMPLETE=1 $BIN snapshot --repo "$T/crepo" --pid $PID --id g7 > "$T/g7.out" 2>&1 || rc=$?
[ $rc = 3 ] && ok "complete failure -> exit 3 (state unknown)" || bad "exit $rc"
grep -q "snapshot g7 is stored and valid" "$T/g7.out" && ok "snapshot reported as stored despite complete failure" || bad "msg"
check $BIN verify --repo "$T/crepo" --snapshot g7
# the mock left the op live in that (now dead) process; state file says CHECKPOINTING
$BIN state --pid $PID | grep -q CHECKPOINTING && ok "target reports CHECKPOINTING (stuck, as a real driver would)" || bad "state: $($BIN state --pid $PID)"

echo "-- restore refuses a snapshot whose shape does not match the target"
rm -rf "$T/mock"; GPUCKPT_MOCK_IMAGES="$T/dev0.orig" $BIN snapshot --repo "$T/crepo" --pid 7 --id one >/dev/null 2>&1
rc=0; GPUCKPT_MOCK_IMAGES="$T/dev0.orig" $BIN restore --repo "$T/crepo" --pid 7 --snapshot g1 > "$T/mm.out" 2>&1 || rc=$?
[ $rc = 3 ] && grep -q "devices" "$T/mm.out" && ok "shape mismatch refused before any copy (exit 3, target left RESTORING)" || { bad "mismatch rc=$rc"; cat "$T/mm.out"; }

source tests/s3_tests.sh

echo
echo "passed=$pass failed=$fail"
[ $fail = 0 ]
