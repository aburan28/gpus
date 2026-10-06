#!/usr/bin/env python3
"""Mock S3 endpoint for the test suite (path-style, single bucket).

Implements PUT (with If-None-Match: *), GET, HEAD, DELETE and ListObjectsV2
with continuation tokens, over an in-memory dict. Every request's AWS
Signature V4 is recomputed from the shared secret with an independent,
spec-derived implementation (sigv4_ref.verify) and rejected with 403 on
mismatch, so the C signer is checked on every call, not only in the unit
test.

Environment:
  MOCK_S3_PORT      listen port (default 0 = pick free; printed on stdout)
  MOCK_S3_PAGE      max keys per list page regardless of max-keys (pagination test)
  MOCK_S3_FAIL_EVERY  return 503 on every Nth request (retry test)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION  credentials to verify
"""
import os, sys, threading, time, hashlib, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigv4_ref

STORE = {}
LOCK = threading.Lock()
COUNT = [0]
PAGE = int(os.environ.get("MOCK_S3_PAGE", "0") or 0)
FAIL_EVERY = int(os.environ.get("MOCK_S3_FAIL_EVERY", "0") or 0)
AK = os.environ.get("AWS_ACCESS_KEY_ID", "")
SK = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
REGION = os.environ.get("AWS_REGION", "us-east-1")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _auth_ok(self, body):
        try:
            return sigv4_ref.verify(self.command, self.path, self.headers, body, AK, SK, REGION)
        except Exception as e:  # noqa
            sys.stderr.write(f"verify error: {e}\n")
            return False

    def _split(self):
        u = urllib.parse.urlsplit(self.path)
        parts = u.path.split("/", 2)  # '', bucket, key
        bucket = parts[1] if len(parts) > 1 else ""
        key = urllib.parse.unquote(parts[2]) if len(parts) > 2 else ""
        q = urllib.parse.parse_qs(u.query, keep_blank_values=True)
        return bucket, key, q

    def _pre(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        with LOCK:
            COUNT[0] += 1
            c = COUNT[0]
        if FAIL_EVERY and c % FAIL_EVERY == 0:
            self._send(503, b"<Error><Code>SlowDown</Code></Error>")
            return None
        if not self._auth_ok(body):
            self._send(403, b"<Error><Code>SignatureDoesNotMatch</Code></Error>")
            return None
        return body

    def do_PUT(self):
        body = self._pre()
        if body is None:
            return
        if hashlib.sha256(body).hexdigest() != self.headers.get("x-amz-content-sha256"):
            return self._send(400, b"<Error><Code>XAmzContentSHA256Mismatch</Code></Error>")
        _, key, _ = self._split()
        with LOCK:
            if self.headers.get("If-None-Match") == "*" and key in STORE:
                return self._send(412, b"<Error><Code>PreconditionFailed</Code></Error>")
            STORE[key] = (body, time.time())
        self._send(200, b"", {"ETag": '"%s"' % hashlib.md5(body).hexdigest()})

    def do_GET(self):
        if self._pre() is None:
            return
        _, key, q = self._split()
        if "list-type" in q:
            return self._list(q)
        with LOCK:
            v = STORE.get(key)
        if v is None:
            return self._send(404, b"<Error><Code>NoSuchKey</Code></Error>")
        self._send(200, v[0])

    def do_HEAD(self):
        if self._pre() is None:
            return
        _, key, _ = self._split()
        with LOCK:
            v = STORE.get(key)
        if v is None:
            return self._send(404)
        self._send(200, v[0])

    def do_DELETE(self):
        if self._pre() is None:
            return
        _, key, _ = self._split()
        with LOCK:
            STORE.pop(key, None)
        self._send(204)

    def _list(self, q):
        prefix = q.get("prefix", [""])[0]
        maxk = int(q.get("max-keys", ["1000"])[0])
        if PAGE:
            maxk = min(maxk, PAGE)
        start = q.get("continuation-token", [""])[0]
        with LOCK:
            keys = sorted(k for k in STORE if k.startswith(prefix) and k > start)
        page, rest = keys[:maxk], keys[maxk:]
        out = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">',
               f"<Prefix>{escape(prefix)}</Prefix><KeyCount>{len(page)}</KeyCount><MaxKeys>{maxk}</MaxKeys>",
               f"<IsTruncated>{'true' if rest else 'false'}</IsTruncated>"]
        if rest:
            out.append(f"<NextContinuationToken>{escape(page[-1])}</NextContinuationToken>")
        for k in page:
            body, mt = STORE[k]
            lm = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(mt))
            out.append(f"<Contents><Key>{escape(k)}</Key><LastModified>{lm}</LastModified>"
                       f"<Size>{len(body)}</Size><StorageClass>STANDARD</StorageClass></Contents>")
        out.append("</ListBucketResult>")
        self._send(200, "".join(out).encode(), {"Content-Type": "application/xml"})


def main():
    port = int(os.environ.get("MOCK_S3_PORT", "0") or 0)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(srv.server_address[1], flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
