#!/usr/bin/env bash
# Data-path tests: staging memory (malloc / THP / hugetlb / mlock), CUDA
# pinning, double-buffered async copies, device hashing and GPUDirect
# Storage. Sourced by tests/run.sh after the mock-driver section; expects
# BIN, T, CS, EXP_UNIQ, ok/bad/check/expect_fail/stat_of and the mock
# driver environment.

IMG_BYTES=$((40*CS + 5*CS + 123))
NCH=46
DIGEST_BYTES=$((NCH*32))
export GPUCKPT_MOCK_COUNTERS="$T/ctr"; mkdir -p "$GPUCKPT_MOCK_COUNTERS"
ctr() { awk -v k="$2" '$1==k{print $2}' "$GPUCKPT_MOCK_COUNTERS/$1.txt" 2>/dev/null; }
rep() { grep "^$2io\.$3=" "$1" | head -1 | cut -d= -f2-; }
fresh_ctr() { rm -f "$GPUCKPT_MOCK_COUNTERS"/*.txt; }
NVRTC_MOCK="$PWD/build/libnvrtc_mock.so"
CUFILE_MOCK="$PWD/build/libcufile_mock.so"

echo "== staging memory: every mode snapshots and restores byte-identically"
$BIN init --repo "$T/hm" --chunk-size $CS >/dev/null
thp_mode=$(sed -n 's/.*\[\(.*\)\].*/\1/p' /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)
for mode in malloc thp hugetlb hugetlb1g; do
    o="$T/hm_$mode.out"
    $BIN snapshot-file --repo "$T/hm" --input "$T/dev0.orig,$T/dev1.orig" --id "hm-$mode" --hostmem $mode > "$o" 2>&1 || { bad "snapshot --hostmem $mode"; cat "$o"; continue; }
    $BIN restore-file --repo "$T/hm" --snapshot "hm-$mode" --output "$T/hm0,$T/hm1" --hostmem $mode > "$T/hm_r.out" 2>&1 || bad "restore --hostmem $mode"
    if cmp -s "$T/hm0" "$T/dev0.orig" && cmp -s "$T/hm1" "$T/dev1.orig"; then ok "--hostmem $mode: $(rep $o '' hostmem)"; else bad "--hostmem $mode restore differs"; fi
done
case "$(rep $T/hm_hugetlb.out '' hostmem)" in
  hugetlb:*) ok "hugetlb pages were available and used" ;;
  "thp (fallback)"*MAP_HUGETLB*) ok "hugetlb unavailable here: fell back to THP and said why" ;;
  *) bad "hugetlb report: $(rep $T/hm_hugetlb.out '' hostmem)" ;;
esac
rep "$T/hm_hugetlb1g.out" '' hostmem | grep -q "1 GiB\|hugetlb1g:" && ok "1 GiB request reports its own outcome" || bad "hugetlb1g report: $(rep $T/hm_hugetlb1g.out '' hostmem)"
huge=$(stat_of "$T/hm_thp.out" hostmem_huge_kb)
if [ "$thp_mode" = never ]; then
    rep "$T/hm_thp.out" '' hostmem | grep -q "THP disabled" && ok "THP disabled system-wide is reported" || bad "THP=never not reported"
elif [ "${huge:-0}" -gt 0 ]; then ok "THP backing obtained: ${huge} kB huge pages (THP mode: $thp_mode)"
else echo "  note THP requested but 0 kB obtained (mode: $thp_mode); reported, not an error"; fi
[ "$(stat_of $T/hm_malloc.out hostmem_huge_kb)" = "0" ] && ok "malloc arena reports 0 kB huge pages" || bad "malloc huge kb"
$BIN snapshot-file --repo "$T/hm" --input "$T/dev1.orig" --id hm-mlock --mlock > "$T/hm_ml.out" 2>&1 || bad "snapshot --mlock"
case "$(rep $T/hm_ml.out '' mlock)" in locked*|failed:*RLIMIT_MEMLOCK*) ok "--mlock: $(rep $T/hm_ml.out '' mlock)";; *) bad "mlock report: $(rep $T/hm_ml.out '' mlock)";; esac
$BIN snapshot-file --repo "$T/hm" --input "$T/dev1.orig" --id hm-pin --pin alloc > "$T/hm_pin.out" 2>&1 || bad "snapshot --pin alloc"
rep "$T/hm_pin.out" '' pin | grep -q "alloc failed: backend cannot allocate pinned memory" && ok "--pin alloc on the file backend falls back and says why" || bad "pin report: $(rep $T/hm_pin.out '' pin)"
rep "$T/hm_thp.out" '' pipeline | grep -q "file backend copies are synchronous" && ok "pipeline clamps to 1 for synchronous backends" || bad "pipeline report: $(rep $T/hm_thp.out '' pipeline)"
expect_fail $BIN snapshot-file --repo "$T/hm" --input "$T/dev1.orig" --hostmem bogus
expect_fail $BIN snapshot-file --repo "$T/hm" --input "$T/dev1.orig" --pipeline 9

echo "== CUDA staging: pinned arena + double-buffered async copies (defaults)"
$BIN init --repo "$T/pr" --chunk-size $CS >/dev/null
fresh_ctr
$BIN snapshot --repo "$T/pr" --pid 6001 --id p1 --resume > "$T/p1.out" 2>&1 || { bad "default cuda snapshot"; cat "$T/p1.out"; }
rep "$T/p1.out" checkpoint. pin | grep -q "^register: .* page-locked for DMA" && ok "staging arena registered with cuMemHostRegister" || bad "pin: $(rep $T/p1.out checkpoint. pin)"
rep "$T/p1.out" checkpoint. pipeline | grep -q "^2 slots" && ok "two slots per worker" || bad "pipeline: $(rep $T/p1.out checkpoint. pipeline)"
[ "$(ctr cuda host_register)" = "1" ] && ok "arena registered once and reused for checkpoint and resume" || bad "host_register=$(ctr cuda host_register)"
rep "$T/p1.out" checkpoint. hostmem | grep -q "\[prepared before the pause\]" && rep "$T/p1.out" restore. hostmem | grep -q "\[prepared before the pause\]" && ok "staging allocated, pre-faulted and pinned before the target is locked" || bad "hostmem: $(rep $T/p1.out checkpoint. hostmem)"
lockline=$(grep -n "^checkpoint.ns_staging_setup=" "$T/p1.out" | cut -d: -f1); mapline=$(grep -n "^checkpoint: mapped" "$T/p1.out" | cut -d: -f1)
[ -n "$lockline" ] && [ "$lockline" -lt "$mapline" ] && ok "staging setup timed and reported before the checkpoint begins" || bad "staging setup ordering"
[ "$(ctr cuda async_copies)" -ge $((2*NCH)) ] && ok "every chunk copied asynchronously ($(ctr cuda async_copies) async copies)" || bad "async_copies=$(ctr cuda async_copies)"
[ "$(ctr cuda dtoh_bytes)" = "$IMG_BYTES" ] && ok "CPU-hash path moves the whole image to the host ($IMG_BYTES bytes)" || bad "dtoh=$(ctr cuda dtoh_bytes)"
[ "$(stat_of $T/p1.out checkpoint.bytes_staged)" = "$IMG_BYTES" ] && ok "bytes_staged agrees with the driver's count" || bad "bytes_staged=$(stat_of $T/p1.out checkpoint.bytes_staged)"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6001.mem.0" "$T/dev0.orig"
fresh_ctr
$BIN snapshot --repo "$T/pr" --pid 6002 --id p2 --pin alloc --pipeline 3 --resume > "$T/p2.out" 2>&1 || bad "--pin alloc snapshot"
[ "$(ctr cuda host_alloc_bytes)" -gt 0 ] && rep "$T/p2.out" checkpoint. pin | grep -q "^alloc:" && ok "--pin alloc uses cuMemHostAlloc" || bad "pin alloc: $(rep $T/p2.out checkpoint. pin)"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6002.mem.1" "$T/dev1.orig"
GPUCKPT_MOCK_REGISTER_FAIL=1 $BIN snapshot --repo "$T/pr" --pid 6003 --id p3 --resume > "$T/p3.out" 2>&1 || bad "register-fail snapshot"
rep "$T/p3.out" checkpoint. pin | grep -q "register failed: cuMemHostRegister: CUresult 801" && ok "registration failure falls back to pageable memory and says why" || bad "pin: $(rep $T/p3.out checkpoint. pin)"
[ "$($BIN show --repo $T/pr --snapshot p3 --chunks | grep '^c ' | sha256sum)" = "$($BIN show --repo $T/pr --snapshot p1 --chunks | grep '^c ' | sha256sum)" ] && ok "pinned, alloc'd and pageable runs store identical chunk lists" || bad "chunk lists differ across pin modes"

echo "== device hashing (NVRTC-compiled kernel on the mock GPU)"
export GPUCKPT_LIBNVRTC="$NVRTC_MOCK"
$BIN init --repo "$T/gh" --chunk-size $CS >/dev/null
fresh_ctr
$BIN snapshot --repo "$T/gh" --pid 6101 --id h1 --gpu-hash on --gpu-hash-check 0 > "$T/h1.out" 2>&1 || { bad "gpu-hash snapshot"; cat "$T/h1.out"; }
rep "$T/h1.out" checkpoint. gpu_hash | grep -q "^on: $NCH chunks hashed on the device" && ok "$(rep $T/h1.out checkpoint. gpu_hash)" || bad "gpu_hash: $(rep $T/h1.out checkpoint. gpu_hash)"
[ "$(ctr cuda kernel_launches)" = "2" ] && ok "one kernel launch per device" || bad "launches=$(ctr cuda kernel_launches)"
[ "$($BIN show --repo $T/gh --snapshot h1 --chunks | grep '^c ' | sha256sum)" = "$($BIN show --repo $T/pr --snapshot p1 --chunks | grep '^c ' | sha256sum)" ] && ok "GPU digests identical to CPU digests for every chunk" || bad "GPU and CPU chunk lists differ"
staged=$(stat_of $T/h1.out checkpoint.bytes_staged)
[ "$(ctr cuda dtoh_bytes)" = "$((DIGEST_BYTES + staged))" ] && ok "host received digests + staged new chunks only ($DIGEST_BYTES + $staged bytes)" || bad "dtoh=$(ctr cuda dtoh_bytes) staged=$staged"
check $BIN verify --repo "$T/gh" --snapshot h1
echo "-- restore with device-side verification"
fresh_ctr
$BIN restore --repo "$T/gh" --pid 6101 --snapshot h1 --gpu-hash on > "$T/h1r.out" 2>&1 || { bad "gpu-verified restore"; cat "$T/h1r.out"; }
rep "$T/h1r.out" restore. gpu_hash | grep -q "^on: $NCH restored chunks verified on the device" && ok "$(rep $T/h1r.out restore. gpu_hash)" || bad "restore gpu_hash: $(rep $T/h1r.out restore. gpu_hash)"
[ "$(stat_of $T/h1r.out restore.ns_hash_sum)" = "0" ] && ok "no CPU hashing during a GPU-verified restore" || bad "ns_hash_sum=$(stat_of $T/h1r.out restore.ns_hash_sum)"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6101.mem.0" "$T/dev0.orig"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6101.mem.1" "$T/dev1.orig"
echo "-- incremental: one changed chunk, only that chunk crosses to the host"
python3 - "$GPUCKPT_MOCK_STATE_DIR/6101.mem.0" $CS <<'PY'
import sys, random
p, cs = sys.argv[1], int(sys.argv[2])
with open(p, "r+b") as f:
    f.seek(17*cs); f.write(random.Random(17).randbytes(cs))
PY
fresh_ctr
$BIN snapshot --repo "$T/gh" --pid 6101 --id h2 --parent h1 --gpu-hash on --gpu-hash-check 0 > "$T/h2.out" 2>&1 || bad "incremental gpu snapshot"
[ "$(ctr cuda dtoh_bytes)" = "$((DIGEST_BYTES + CS))" ] && ok "dtoh = $DIGEST_BYTES digest bytes + 1 changed chunk, of $IMG_BYTES image bytes" || bad "dtoh=$(ctr cuda dtoh_bytes) want $((DIGEST_BYTES + CS))"
[ "$(stat_of $T/h2.out checkpoint.chunks_new)" = "1" ] && [ "$(stat_of $T/h2.out checkpoint.chunks_same_as_parent)" = "45" ] && ok "chunks_new=1, same_as_parent=45" || bad "new=$(stat_of $T/h2.out checkpoint.chunks_new) same=$(stat_of $T/h2.out checkpoint.chunks_same_as_parent)"
echo "-- CPU cross-check of GPU digests"
GPUCKPT_MOCK_STATE_DIR="$T/mock" $BIN restore --repo "$T/gh" --pid 6101 --snapshot h2 >/dev/null 2>&1
fresh_ctr
$BIN snapshot --repo "$T/gh" --pid 6101 --id h3 --gpu-hash on --resume > "$T/h3.out" 2>&1 || bad "checked gpu snapshot"
[ "$(stat_of $T/h3.out checkpoint.chunks_cpu_checked)" = "8" ] && rep "$T/h3.out" checkpoint. gpu_hash | grep -q "8 cross-checked on the CPU" && ok "default cross-check re-hashes 8 sampled chunks on the CPU" || bad "cpu_checked=$(stat_of $T/h3.out checkpoint.chunks_cpu_checked)"
GPUCKPT_MOCK_KERNEL_CORRUPT=1 $BIN snapshot --repo "$T/gh" --pid 6101 --id h4 --gpu-hash on --resume > "$T/h4.out" 2>&1 || { bad "corrupt-kernel snapshot+resume should succeed"; tail -3 "$T/h4.out"; }
rep "$T/h4.out" checkpoint. gpu_hash | grep -q "^off (GPU digest of chunk 0 differed" && ok "a wrong GPU digest is caught at snapshot; snapshot re-hashed on the CPU" || bad "gpu_hash: $(rep $T/h4.out checkpoint. gpu_hash)"
rep "$T/h4.out" restore. gpu_hash | grep -q "^on, unreliable: the device flagged 2 chunk(s) that re-hashed correctly on the CPU" && ok "at restore, the 2 chunks the device flagged (chunk 0 of each GPU) are re-hashed on the CPU and cleared" || bad "restore gpu_hash: $(rep $T/h4.out restore. gpu_hash)"
check $BIN verify --repo "$T/gh" --snapshot h4
$BIN init --repo "$T/gx" --chunk-size $CS >/dev/null
GPUCKPT_MOCK_KERNEL_CORRUPT=1 $BIN snapshot --repo "$T/gx" --pid 6101 --id x1 --gpu-hash on --gpu-hash-check 0 --resume >/dev/null 2>&1 || true
$BIN verify --repo "$T/gx" --snapshot x1 > "$T/x1v.out" 2>&1 && bad "unchecked corrupt digest went unnoticed" || { grep -q "corrupt=2" "$T/x1v.out" && ok "with --gpu-hash-check 0 wrong digests store 2 misnamed chunks, and CPU verify finds both" || bad "verify: $(cat $T/x1v.out)"; }
echo "-- device hashing unavailable or failing: reported, CPU path used"
GPUCKPT_MOCK_KERNEL_FAULT=1 $BIN snapshot --repo "$T/gh" --pid 6101 --id h5 --gpu-hash on --resume > "$T/h5.out" 2>&1 || bad "kernel-fault snapshot"
rep "$T/h5.out" checkpoint. gpu_hash | grep -q "device hashing failed: gc_sha256_chunks kernel: CUresult 700" && ok "kernel fault falls back to CPU hashing" || bad "gpu_hash: $(rep $T/h5.out checkpoint. gpu_hash)"
rep "$T/h5.out" restore. gpu_hash | grep -q "^off (requested" && ok "after a fault the kernel is not offered again in the same process; restore verifies on the CPU" || bad "restore after fault: $(rep $T/h5.out restore. gpu_hash)"
$BIN state --pid 6101 | grep -q RUNNING && ok "target RUNNING after the faulted run" || bad "state: $($BIN state --pid 6101)"
GPUCKPT_MOCK_NVRTC_FAIL=1 $BIN snapshot --repo "$T/gh" --pid 6101 --id h6 --gpu-hash on --resume > "$T/h6.out" 2>&1 || bad "nvrtc-fail snapshot"
grep -q "warning: --gpu-hash unavailable: nvrtcCompileProgram(--gpu-architecture=compute_90): NVRTC_ERROR_COMPILATION: mock: injected compile failure" "$T/h6.out" && ok "NVRTC compile failure is reported with its log" || bad "nvrtc msg: $(grep warning $T/h6.out)"
GPUCKPT_LIBNVRTC=/nonexistent/libnvrtc.so $BIN snapshot --repo "$T/gh" --pid 6101 --id h7 --gpu-hash on --resume > "$T/h7.out" 2>&1 || bad "no-nvrtc snapshot"
grep -q "NVRTC not found" "$T/h7.out" && ok "missing NVRTC is reported" || bad "no-nvrtc msg: $(grep warning $T/h7.out)"
printf '// MOCKPTX-FILE\n' > "$T/k.ptx"
GPUCKPT_KERNEL_PTX="$T/k.ptx" GPUCKPT_LIBNVRTC=/nonexistent $BIN snapshot --repo "$T/gh" --pid 6101 --id h8 --gpu-hash on --resume > "$T/h8.out" 2>&1 || bad "prebuilt-ptx snapshot"
rep "$T/h8.out" checkpoint. gpu_hash | grep -q "^on:" && ok "GPUCKPT_KERNEL_PTX loads a prebuilt kernel without NVRTC" || bad "ptx: $(rep $T/h8.out checkpoint. gpu_hash)"
vbad=0; for s in h5 h6 h7 h8; do $BIN verify --repo "$T/gh" --snapshot $s >/dev/null 2>&1 || { vbad=1; bad "verify $s"; }; done
[ $vbad = 0 ] && ok "fallback snapshots all verify"

echo "== GPUDirect Storage: chunk bytes never enter host memory"
export GPUCKPT_LIBCUFILE="$CUFILE_MOCK"
$BIN init --repo "$T/gd" --chunk-size $CS >/dev/null
fresh_ctr
$BIN snapshot --repo "$T/gd" --pid 6201 --id d1 --gpu-hash on --gpu-hash-check 0 --gds on > "$T/d1.out" 2>&1 || { bad "gds snapshot"; cat "$T/d1.out"; }
rep "$T/d1.out" checkpoint. gds | grep -q "^on: [0-9]* bytes written device->file" && ok "$(rep $T/d1.out checkpoint. gds)" || bad "gds: $(rep $T/d1.out checkpoint. gds)"
[ "$(ctr cuda dtoh_bytes)" = "$DIGEST_BYTES" ] && ok "device->host traffic is the $DIGEST_BYTES digest bytes and nothing else" || bad "dtoh=$(ctr cuda dtoh_bytes)"
[ "$(stat_of $T/d1.out checkpoint.bytes_staged)" = "0" ] && ok "bytes_staged=0" || bad "bytes_staged=$(stat_of $T/d1.out checkpoint.bytes_staged)"
[ "$(ctr cufile write_bytes)" = "$(stat_of $T/d1.out checkpoint.bytes_direct)" ] && [ "$(ctr cufile buf_registered)" = "2" ] && ok "cuFileWrite moved $(ctr cufile write_bytes) bytes; both mappings registered" || bad "cufile write=$(ctr cufile write_bytes) direct=$(stat_of $T/d1.out checkpoint.bytes_direct) reg=$(ctr cufile buf_registered)"
[ "$(stat_of $T/d1.out checkpoint.chunks_new)" = "$EXP_UNIQ" ] && ok "chunks_new=$EXP_UNIQ (write-once publish dedups racing writers)" || bad "chunks_new=$(stat_of $T/d1.out checkpoint.chunks_new)"
check $BIN verify --repo "$T/gd" --snapshot d1
fresh_ctr
$BIN restore --repo "$T/gd" --pid 6201 --snapshot d1 --gpu-hash on --gds on > "$T/d1r.out" 2>&1 || { bad "gds restore"; cat "$T/d1r.out"; }
[ "$(ctr cuda htod_bytes)" = "0" ] && [ "$(ctr cufile read_bytes)" = "$IMG_BYTES" ] && ok "restore: 0 host->device bytes, $IMG_BYTES bytes via cuFileRead" || bad "htod=$(ctr cuda htod_bytes) cufile read=$(ctr cufile read_bytes)"
rep "$T/d1r.out" restore. hostmem | grep -q "not used (direct I/O)" && ok "no staging arena allocated for a direct restore" || bad "hostmem: $(rep $T/d1r.out restore. hostmem)"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6201.mem.0" "$T/dev0.orig"
check cmp "$GPUCKPT_MOCK_STATE_DIR/6201.mem.1" "$T/dev1.orig"
echo "-- corrupted chunk under a direct restore is caught by device verification"
$BIN snapshot --repo "$T/gd" --pid 6201 --id d2 --gpu-hash on --gds on >/dev/null 2>&1
victim=$(find "$T/gd/chunks" -type f | sort | head -1); cp "$victim" "$T/gdvictim"
printf 'Z' | dd of="$victim" bs=1 seek=7 conv=notrunc status=none
rc=0; $BIN restore --repo "$T/gd" --pid 6201 --snapshot d2 --gpu-hash on --gds on > "$T/d2r.out" 2>&1 || rc=$?
[ $rc = 3 ] && grep -q "restored bytes do not match the manifest" "$T/d2r.out" && ok "device verification rejects it; operation not completed (exit 3)" || { bad "corrupt direct restore rc=$rc"; tail -3 "$T/d2r.out"; }
cp "$T/gdvictim" "$victim"
echo "-- GDS preconditions are enforced and reported"
$BIN init --repo "$T/gd2" --chunk-size $CS >/dev/null
$BIN snapshot --repo "$T/gd2" --pid 6202 --id e1 --gds on --resume > "$T/e1.out" 2>&1 || bad "gds without gpu-hash"
rep "$T/e1.out" checkpoint. gds | grep -q "^off (needs device hashing" && ok "--gds without --gpu-hash: off, with the reason" || bad "gds: $(rep $T/e1.out checkpoint. gds)"
GPUCKPT_MOCK_CUFILE_FAIL=1 $BIN snapshot --repo "$T/gd2" --pid 6202 --id e2 --gpu-hash on --gds on --resume > "$T/e2.out" 2>&1 || bad "cufile-fail snapshot"
grep -q "warning: --gds unavailable: cuFileDriverOpen: cuFile error 5001" "$T/e2.out" && rep "$T/e2.out" checkpoint. gds | grep -q "did not enable direct I/O" && ok "cuFile driver failure falls back to staging" || bad "cufile fail: $(grep -h 'warning\|io.gds' $T/e2.out)"
GPUCKPT_LIBCUFILE=/nonexistent $BIN snapshot --repo "$T/gd2" --pid 6202 --id e3 --gpu-hash on --gds on --resume > "$T/e3.out" 2>&1 || bad "no-cufile snapshot"
grep -q "libcufile not found" "$T/e3.out" && ok "missing libcufile is reported" || bad "no-cufile msg: $(grep warning $T/e3.out)"
check $BIN verify --repo "$T/gd2"
if ./build/gpuckpt version | grep -q "s3: libcurl"; then
    export AWS_ACCESS_KEY_ID=AKIDTEST AWS_SECRET_ACCESS_KEY=sekrit AWS_REGION=us-east-1
    python3 tests/mock/s3_server.py > "$T/gdsport" 2>/dev/null &
    GDS_S3PID=$!
    for i in $(seq 1 50); do [ -s "$T/gdsport" ] && break; sleep 0.1; done
    export GPUCKPT_S3_ENDPOINT="http://127.0.0.1:$(cat $T/gdsport)"
    $BIN init --repo s3://gds/r --chunk-size $CS >/dev/null
    $BIN snapshot --repo s3://gds/r --pid 6203 --id s1 --gpu-hash on --gds on --resume > "$T/e4.out" 2>&1 || bad "gds to s3"
    rep "$T/e4.out" checkpoint. gds | grep -q "^off (repo is s3" && rep "$T/e4.out" checkpoint. gpu_hash | grep -q "^on:" && ok "S3 repo: device hashing on, direct I/O off with the reason" || bad "s3 gds: $(rep $T/e4.out checkpoint. gds)"
    kill $GDS_S3PID; wait $GDS_S3PID 2>/dev/null || true
    unset GPUCKPT_S3_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION
fi
unset GPUCKPT_LIBNVRTC GPUCKPT_LIBCUFILE GPUCKPT_MOCK_COUNTERS
