#!/usr/bin/env bash
# S3 backend tests. Sourced by tests/run.sh after the local-store sections;
# expects $BIN, $T, $CS, ok/bad/check/expect_fail/stat_of to be defined.

if ! ./build/gpuckpt version | grep -q "s3: libcurl"; then echo "== S3 backend not built (no libcurl): section skipped"; return 0 2>/dev/null || exit 0; fi

echo "== SigV4: C signer vs spec-derived Python reference vs botocore"
SIG_ARGS=(GET examplebucket.s3.amazonaws.com /test.txt "" e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 20130524T000000Z us-east-1 AKIAIOSFODNN7EXAMPLE wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY "")
c_sig=$(./build/sigv4_test "${SIG_ARGS[@]}")
py_sig=$(python3 -c "
import sys; sys.path.insert(0,'tests/mock'); import sigv4_ref
a,_,_=sigv4_ref.sign('GET','examplebucket.s3.amazonaws.com','/test.txt','','e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','20130524T000000Z','us-east-1','AKIAIOSFODNN7EXAMPLE','wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
print(a)")
[ "$c_sig" = "$py_sig" ] && ok "GET signature matches Python reference" || { bad "GET signature"; echo "  C:  $c_sig"; echo "  py: $py_sig"; }
# with session token and a query string
SIG_ARGS2=(GET 127.0.0.1:9000 /bkt/ "list-type=2&max-keys=1000&prefix=chunks%2F" e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 20261006T010203Z eu-west-1 AKID secret TOKEN123)
c_sig2=$(./build/sigv4_test "${SIG_ARGS2[@]}")
py_sig2=$(python3 -c "
import sys; sys.path.insert(0,'tests/mock'); import sigv4_ref
a,_,_=sigv4_ref.sign('GET','127.0.0.1:9000','/bkt/','list-type=2&max-keys=1000&prefix=chunks%2F','e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855','20261006T010203Z','eu-west-1','AKID','secret','TOKEN123')
print(a)")
[ "$c_sig2" = "$py_sig2" ] && ok "LIST+token signature matches Python reference" || { bad "LIST+token signature"; echo "  C:  $c_sig2"; echo "  py: $py_sig2"; }
bt=$(python3 - <<'PY' 2>/dev/null
import sys
try:
    import botocore.auth, botocore.credentials, botocore.awsrequest, datetime
except ImportError:
    print("SKIP"); sys.exit(0)
class FixedDT(datetime.datetime):
    @classmethod
    def utcnow(cls): return cls(2013,5,24,0,0,0)
    @classmethod
    def now(cls, tz=None): return cls(2013,5,24,0,0,0, tzinfo=tz)
botocore.auth.datetime.datetime = FixedDT
creds = botocore.credentials.Credentials('AKIAIOSFODNN7EXAMPLE','wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY')
req = botocore.awsrequest.AWSRequest(method='GET', url='https://examplebucket.s3.amazonaws.com/test.txt',
      headers={'host':'examplebucket.s3.amazonaws.com','x-amz-content-sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'})
botocore.auth.SigV4Auth(creds,'s3','us-east-1').add_auth(req)
print(req.headers['Authorization'])
PY
)
if [ "$bt" = "SKIP" ] || [ -z "$bt" ]; then echo "  skip botocore cross-check (not importable)"; else
  [ "$c_sig" = "$bt" ] && ok "GET signature matches botocore SigV4Auth" || { bad "botocore mismatch"; echo "  C:  $c_sig"; echo "  bc: $bt"; }
fi

echo "== mock S3 endpoint: repo lifecycle over HTTP"
export AWS_ACCESS_KEY_ID=AKIDTEST AWS_SECRET_ACCESS_KEY=sekrit AWS_REGION=us-east-1
MOCK_S3_PAGE=7 python3 tests/mock/s3_server.py > "$T/s3port" 2> "$T/s3server.err" &
S3PID=$!
for i in $(seq 1 50); do [ -s "$T/s3port" ] && break; sleep 0.1; done
PORT=$(cat "$T/s3port")
export GPUCKPT_S3_ENDPOINT="http://127.0.0.1:$PORT"
S3REPO="s3://testbucket/ckpt/repo"
check $BIN init --repo "$S3REPO" --chunk-size $CS
expect_fail $BIN init --repo "$S3REPO" --chunk-size $CS
expect_fail $BIN list --repo "s3://testbucket/nothing-here"
$BIN snapshot-file --repo "$S3REPO" --input "$T/dev0.orig,$T/dev1.orig" --id s3a > "$T/s3a.out" || bad "s3 snapshot exit $?"
[ "$(stat_of $T/s3a.out chunks_total)" = "46" ] && ok "46 chunks to S3" || bad "chunks_total=$(stat_of $T/s3a.out chunks_total)"
[ "$(stat_of $T/s3a.out chunks_new)" = "$EXP_UNIQ" ] && ok "chunks_new=$EXP_UNIQ on S3 (dedup via If-None-Match)" || bad "chunks_new=$(stat_of $T/s3a.out chunks_new)"
[ "$(stat_of $T/s3a.out threads)" = "16" ] && ok "S3 default 16 threads" || bad "threads=$(stat_of $T/s3a.out threads)"
$BIN snapshot-file --repo "$S3REPO" --input "$T/dev0.img,$T/dev1.img" --id s3b --parent s3a > "$T/s3b.out" || bad "s3 snapshot b"
[ "$(stat_of $T/s3b.out chunks_new)" = "3" ] && ok "3 new chunks on second S3 snapshot" || bad "chunks_new=$(stat_of $T/s3b.out chunks_new)"
expect_fail $BIN snapshot-file --repo "$S3REPO" --input "$T/dev0.img" --id s3a      # manifest immutability over HTTP
[ "$($BIN list --repo "$S3REPO" | tr '\n' ' ')" = "s3a s3b " ] && ok "list over paginated ListObjectsV2 (page=7)" || bad "list: $($BIN list --repo $S3REPO | tr '\n' ' ')"
check $BIN show --repo "$S3REPO" --snapshot s3a
check $BIN verify --repo "$S3REPO"
$BIN restore-file --repo "$S3REPO" --snapshot s3a --output "$T/s3r0,$T/s3r1" >/dev/null || bad "s3 restore"
check cmp "$T/s3r0" "$T/dev0.orig"
check cmp "$T/s3r1" "$T/dev1.orig"
$BIN restore-file --repo "$S3REPO" --snapshot s3b --output "$T/s3q0,$T/s3q1" >/dev/null || bad "s3 restore b"
check cmp "$T/s3q0" "$T/dev0.img"

echo "-- forget + gc with grace: young chunks are kept, then reclaimed with --grace-seconds 0"
check $BIN forget --repo "$S3REPO" --snapshot s3a
$BIN gc --repo "$S3REPO" > "$T/s3gc1.out"
[ "$(stat_of $T/s3gc1.out chunks_deleted)" = "0" ] && ok "default grace keeps 3 fresh unreferenced chunks" || bad "deleted $(stat_of $T/s3gc1.out chunks_deleted) under grace"
$BIN gc --repo "$S3REPO" --grace-seconds 0 > "$T/s3gc2.out"
[ "$(stat_of $T/s3gc2.out chunks_deleted)" = "3" ] && ok "grace 0 reclaims exactly 3" || bad "deleted $(stat_of $T/s3gc2.out chunks_deleted)"
check $BIN verify --repo "$S3REPO" --snapshot s3b
expect_fail $BIN verify --repo "$S3REPO" --snapshot s3a

echo "-- local repo synced to S3 key-for-key is a valid S3 repo"
python3 - "$T/repo" "$PORT" <<'PY'
import sys, os, urllib.request, hashlib
sys.path.insert(0, 'tests/mock'); import sigv4_ref, datetime
root, port = sys.argv[1], sys.argv[2]
ak, sk = os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY']
n = 0
for dp, _, fs in os.walk(root):
    for f in fs:
        if f.startswith('.tmp.') or f == 'lock': continue
        p = os.path.join(dp, f); key = os.path.relpath(p, root)
        body = open(p, 'rb').read()
        ph = hashlib.sha256(body).hexdigest()
        d = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        host = f'127.0.0.1:{port}'
        uri = '/testbucket/synced/' + key
        auth, _, _ = sigv4_ref.sign('PUT', host, uri, '', ph, d, 'us-east-1', ak, sk)
        req = urllib.request.Request(f'http://{host}{uri}', data=body, method='PUT',
              headers={'host': host, 'x-amz-content-sha256': ph, 'x-amz-date': d, 'Authorization': auth})
        urllib.request.urlopen(req).read(); n += 1
print(n)
PY
check $BIN verify --repo "s3://testbucket/synced"
[ "$($BIN list --repo s3://testbucket/synced | wc -l)" = "$($BIN list --repo $T/repo | wc -l)" ] && ok "same snapshot list local vs synced S3" || bad "synced list differs"

echo "-- bad credentials are rejected by the endpoint's signature check"
AWS_SECRET_ACCESS_KEY=wrong $BIN list --repo "$S3REPO" >/dev/null 2>"$T/s3bad.err" && bad "wrong secret accepted" || ok "wrong secret -> $(grep -o 'HTTP 403' $T/s3bad.err | head -1)"

echo "-- transient 503s are retried"
kill $S3PID; wait $S3PID 2>/dev/null || true
MOCK_S3_FAIL_EVERY=4 GPUCKPT_S3_RETRIES=3 python3 tests/mock/s3_server.py > "$T/s3port2" 2>>"$T/s3server.err" &
S3PID=$!
for i in $(seq 1 50); do [ -s "$T/s3port2" ] && break; sleep 0.1; done
export GPUCKPT_S3_ENDPOINT="http://127.0.0.1:$(cat $T/s3port2)"
check $BIN init --repo "s3://flaky/r" --chunk-size $CS
$BIN snapshot-file --repo "s3://flaky/r" --input "$T/dev1.orig" --id f1 >/dev/null 2>&1 && ok "snapshot completes through every-4th-request 503s" || bad "snapshot under 503 injection"
check $BIN verify --repo "s3://flaky/r"
GPUCKPT_S3_RETRIES=0 $BIN verify --repo "s3://flaky/r" >/dev/null 2>&1 && bad "retries=0 should fail under injection" || ok "retries=0 fails as expected"
kill $S3PID; wait $S3PID 2>/dev/null || true
unset GPUCKPT_S3_ENDPOINT AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_REGION
