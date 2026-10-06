"""Independent AWS Signature V4 reference, written from the signing spec
(docs.aws.amazon.com, "Create a signed AWS API request"). Used by the mock
S3 server to verify every request from the C client, and by tests/run.sh to
compare against the C signer and, when botocore is importable, against
botocore's own SigV4Auth."""
import hashlib, hmac, urllib.parse


def _enc(s, keep_slash):
    safe = "-_.~" + ("/" if keep_slash else "")
    return urllib.parse.quote(s, safe=safe)


def canonical_query(query):
    if not query:
        return ""
    pairs = []
    for part in query.split("&"):
        k, _, v = part.partition("=")
        pairs.append((_enc(urllib.parse.unquote(k), False), _enc(urllib.parse.unquote(v), False)))
    return "&".join(f"{k}={v}" for k, v in sorted(pairs))


def sign(method, host, canonical_uri, canonical_qs, payload_hash, amz_date, region, ak, sk, token=None,
         signed_headers=None, headers=None):
    """Return (authorization header value, canonical request, string to sign)."""
    headers = dict(headers or {})
    headers.setdefault("host", host)
    headers.setdefault("x-amz-content-sha256", payload_hash)
    headers.setdefault("x-amz-date", amz_date)
    if token:
        headers.setdefault("x-amz-security-token", token)
    if signed_headers is None:
        signed_headers = sorted(headers)
    canon_headers = "".join(f"{h}:{headers[h].strip()}\n" for h in signed_headers)
    sh = ";".join(signed_headers)
    creq = "\n".join([method, canonical_uri, canonical_qs, canon_headers, sh, payload_hash])
    date = amz_date[:8]
    scope = f"{date}/{region}/s3/aws4_request"
    sts = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(creq.encode()).hexdigest()])

    def h(k, m):
        return hmac.new(k, m.encode(), hashlib.sha256).digest()

    k = h(h(h(h(("AWS4" + sk).encode(), date), region), "s3"), "aws4_request")
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={sh}, Signature={sig}", creq, sts


def verify(method, raw_path, headers, body, ak, sk, region):
    auth = headers.get("Authorization", "")
    if not auth.startswith("AWS4-HMAC-SHA256 "):
        return False
    fields = dict(p.strip().split("=", 1) for p in auth[len("AWS4-HMAC-SHA256 "):].split(","))
    signed = fields["SignedHeaders"].split(";")
    u = urllib.parse.urlsplit(raw_path)
    # canonical URI: the path as sent, re-encoded once per SigV4 rules
    canonical_uri = _enc(urllib.parse.unquote(u.path), True)
    hdrs = {h: headers.get(h, "") for h in signed}
    payload_hash = headers.get("x-amz-content-sha256", "")
    if hashlib.sha256(body).hexdigest() != payload_hash:
        return False
    amz_date = headers.get("x-amz-date", "")
    expect, _, _ = sign(method, hdrs.get("host", ""), canonical_uri, canonical_query(u.query), payload_hash,
                        amz_date, region, ak, sk, headers.get("x-amz-security-token"),
                        signed_headers=signed, headers=hdrs)
    return hmac.compare_digest(expect.split("Signature=")[1], fields["Signature"])
