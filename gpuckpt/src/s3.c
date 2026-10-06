/* S3 object store client: libcurl transport, AWS Signature Version 4
 * signing implemented here (no AWS SDK). Supports AWS S3 and S3-compatible
 * endpoints (MinIO, Ceph RGW, mocks) through GPUCKPT_S3_ENDPOINT.
 *
 * Write-once semantics on S3 come from `If-None-Match: *` on PUT, which S3
 * answers with 412 Precondition Failed when the key exists. That gives the
 * same exactly-one-winner guarantee link(2) gives the local store.
 *
 * Transient failures (network errors, HTTP 5xx, 429) are retried with
 * exponential backoff; every request is idempotent, so a retry is safe.
 */
#include "objstore.h"
#include <pthread.h>
#include <unistd.h>
#include <time.h>

/* ------------------------------------------------------------ HMAC/SigV4 */

void gc_hmac_sha256(const void *key, size_t klen, const void *data, size_t dlen, uint8_t out[GC_HASH_LEN])
{
    uint8_t k[64] = {0};
    if (klen > 64) gc_sha256(key, klen, k);
    else memcpy(k, key, klen);
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    gc_sha256_ctx c;
    uint8_t inner[GC_HASH_LEN];
    gc_sha256_init(&c); gc_sha256_update(&c, ipad, 64); gc_sha256_update(&c, data, dlen); gc_sha256_final(&c, inner);
    gc_sha256_init(&c); gc_sha256_update(&c, opad, 64); gc_sha256_update(&c, inner, GC_HASH_LEN); gc_sha256_final(&c, out);
}

/* Build the Authorization header value. canonical_uri and canonical_query
 * must already be URI-encoded per the SigV4 rules. */
int gc_s3_sign(const char *method, const char *host, const char *canonical_uri,
               const char *canonical_query, const char *payload_hash_hex,
               const char *amz_date, const char *region, const char *access_key,
               const char *secret_key, const char *session_token,
               char *authorization, size_t authorization_len)
{
    char date[9];
    memcpy(date, amz_date, 8); date[8] = 0;
    const char *signed_headers = session_token && *session_token
        ? "host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
        : "host;x-amz-content-sha256;x-amz-date";
    char canonical[8192];
    int n = snprintf(canonical, sizeof canonical,
                     "%s\n%s\n%s\nhost:%s\nx-amz-content-sha256:%s\nx-amz-date:%s\n%s%s%s\n%s\n%s",
                     method, canonical_uri, canonical_query ? canonical_query : "", host,
                     payload_hash_hex, amz_date,
                     session_token && *session_token ? "x-amz-security-token:" : "",
                     session_token && *session_token ? session_token : "",
                     session_token && *session_token ? "\n" : "",
                     signed_headers, payload_hash_hex);
    if (n < 0 || (size_t)n >= sizeof canonical) return GC_EINVAL;
    uint8_t h[GC_HASH_LEN];
    char hhex[GC_HEX_LEN + 1];
    gc_sha256(canonical, (size_t)n, h);
    gc_hex(h, GC_HASH_LEN, hhex);
    char scope[256], sts[1024];
    snprintf(scope, sizeof scope, "%s/%s/s3/aws4_request", date, region);
    n = snprintf(sts, sizeof sts, "AWS4-HMAC-SHA256\n%s\n%s\n%s", amz_date, scope, hhex);
    char ksecret[300];
    int kn = snprintf(ksecret, sizeof ksecret, "AWS4%s", secret_key);
    if (kn < 0 || (size_t)kn >= sizeof ksecret) return GC_EINVAL;
    uint8_t k1[GC_HASH_LEN], k2[GC_HASH_LEN], k3[GC_HASH_LEN], k4[GC_HASH_LEN], sig[GC_HASH_LEN];
    gc_hmac_sha256(ksecret, (size_t)kn, date, 8, k1);
    gc_hmac_sha256(k1, GC_HASH_LEN, region, strlen(region), k2);
    gc_hmac_sha256(k2, GC_HASH_LEN, "s3", 2, k3);
    gc_hmac_sha256(k3, GC_HASH_LEN, "aws4_request", 12, k4);
    gc_hmac_sha256(k4, GC_HASH_LEN, sts, (size_t)n, sig);
    char sighex[GC_HEX_LEN + 1];
    gc_hex(sig, GC_HASH_LEN, sighex);
    n = snprintf(authorization, authorization_len,
                 "AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
                 access_key, scope, signed_headers, sighex);
    return (n < 0 || (size_t)n >= authorization_len) ? GC_EINVAL : GC_OK;
}

/* SigV4 URI encoding: unreserved characters pass, '/' passes when
 * keep_slash, everything else becomes %XX (uppercase). */
__attribute__((unused)) static int uri_encode(const char *s, int keep_slash, char *out, size_t n)
{
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        int ok = (*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') ||
                 *p == '-' || *p == '_' || *p == '.' || *p == '~' || (keep_slash && *p == '/');
        if (ok) { if (o + 2 > n) return GC_EINVAL; out[o++] = (char)*p; }
        else { if (o + 4 > n) return GC_EINVAL; snprintf(out + o, 4, "%%%02X", *p); o += 3; }
    }
    out[o] = 0;
    return GC_OK;
}

#ifndef GC_USE_CURL
/* Built without libcurl: s3:// repositories are refused with a clear reason. */
const char *gc_s3_impl(void) { return "unavailable"; }
int gc_s3_open(const char *url, gc_s3 **out)
{
    (void)out;
    gc_set_error("%s: this gpuckpt was built without libcurl; rebuild with libcurl-dev installed", url);
    return GC_EBACKEND;
}
void gc_s3_close(gc_s3 *s) { (void)s; }
int gc_s3_put(gc_s3 *s, const char *k, const void *d, size_t n) { (void)s; (void)k; (void)d; (void)n; return GC_EBACKEND; }
int gc_s3_get(gc_s3 *s, const char *k, void **d, size_t *n) { (void)s; (void)k; (void)d; (void)n; return GC_EBACKEND; }
int gc_s3_head(gc_s3 *s, const char *k, uint64_t *n) { (void)s; (void)k; (void)n; return GC_EBACKEND; }
int gc_s3_delete(gc_s3 *s, const char *k, uint64_t *n) { (void)s; (void)k; (void)n; return GC_EBACKEND; }
int gc_s3_list(gc_s3 *s, const char *p, gc_obj_list_cb cb, void *u) { (void)s; (void)p; (void)cb; (void)u; return GC_EBACKEND; }
#else
#include <curl/curl.h>

const char *gc_s3_impl(void) { return "libcurl"; }

struct gc_s3 {
    char bucket[256];
    char prefix[512];     /* "" or "a/b/" */
    char region[64];
    char scheme[8];       /* http or https */
    char host[512];       /* Host header value, host[:port] */
    char access_key[256];
    char secret_key[256];
    char token[2048];
    char ca_bundle[PATH_MAX];
    int  path_style;
    int  retries;
};

/* One easy handle per thread, reused across requests for connection reuse.
 * Worker threads release theirs through the key destructor; the main
 * thread's is released by the atexit hook along with curl's globals. */
static pthread_once_t curl_once = PTHREAD_ONCE_INIT;
static pthread_key_t curl_key;
static void curl_thread_fini(void *h) { if (h) curl_easy_cleanup(h); }
static void curl_atexit(void)
{
    CURL *h = pthread_getspecific(curl_key);
    if (h) { curl_easy_cleanup(h); pthread_setspecific(curl_key, NULL); }
    curl_global_cleanup();
}
static void curl_init_once(void)
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
    pthread_key_create(&curl_key, curl_thread_fini);
    atexit(curl_atexit);
}
static CURL *thread_curl(void)
{
    CURL *h = pthread_getspecific(curl_key);
    if (!h) { h = curl_easy_init(); pthread_setspecific(curl_key, h); }
    return h;
}

static const char *envor(const char *a, const char *b, const char *dflt)
{
    const char *v = getenv(a);
    if ((!v || !*v) && b) v = getenv(b);
    return v && *v ? v : dflt;
}

int gc_s3_open(const char *url, gc_s3 **out)
{
    if (strncmp(url, "s3://", 5) != 0) { gc_set_error("%s: not an s3:// url", url); return GC_EINVAL; }
    gc_s3 *s = calloc(1, sizeof *s);
    if (!s) return GC_ENOMEM;
    const char *p = url + 5;
    const char *slash = strchr(p, '/');
    size_t bl = slash ? (size_t)(slash - p) : strlen(p);
    if (bl == 0 || bl >= sizeof s->bucket) { free(s); gc_set_error("%s: missing bucket", url); return GC_EINVAL; }
    memcpy(s->bucket, p, bl); s->bucket[bl] = 0;
    if (slash && slash[1]) {
        snprintf(s->prefix, sizeof s->prefix, "%s", slash + 1);
        size_t n = strlen(s->prefix);
        if (s->prefix[n - 1] != '/' && n + 1 < sizeof s->prefix) { s->prefix[n] = '/'; s->prefix[n + 1] = 0; }
    }
    snprintf(s->region, sizeof s->region, "%s", envor("AWS_REGION", "AWS_DEFAULT_REGION", "us-east-1"));
    snprintf(s->access_key, sizeof s->access_key, "%s", envor("AWS_ACCESS_KEY_ID", NULL, ""));
    snprintf(s->secret_key, sizeof s->secret_key, "%s", envor("AWS_SECRET_ACCESS_KEY", NULL, ""));
    snprintf(s->token, sizeof s->token, "%s", envor("AWS_SESSION_TOKEN", NULL, ""));
    snprintf(s->ca_bundle, sizeof s->ca_bundle, "%s", envor("AWS_CA_BUNDLE", NULL, ""));
    if (!s->access_key[0] || !s->secret_key[0]) {
        free(s);
        gc_set_error("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set (instance profiles are not supported)");
        return GC_EINVAL;
    }
    const char *ep = envor("GPUCKPT_S3_ENDPOINT", NULL, NULL);
    if (ep) {
        const char *h = ep;
        if (!strncmp(ep, "https://", 8)) { strcpy(s->scheme, "https"); h = ep + 8; }
        else if (!strncmp(ep, "http://", 7)) { strcpy(s->scheme, "http"); h = ep + 7; }
        else { free(s); gc_set_error("GPUCKPT_S3_ENDPOINT must start with http:// or https://"); return GC_EINVAL; }
        snprintf(s->host, sizeof s->host, "%s", h);
        char *t = strchr(s->host, '/'); if (t) *t = 0;
        s->path_style = atoi(envor("GPUCKPT_S3_PATH_STYLE", NULL, "1"));
        if (!s->path_style) {
            char vh[512];
            snprintf(vh, sizeof vh, "%s.%s", s->bucket, s->host);
            snprintf(s->host, sizeof s->host, "%s", vh);
        }
    } else {
        strcpy(s->scheme, "https");
        s->path_style = atoi(envor("GPUCKPT_S3_PATH_STYLE", NULL, "0"));
        if (s->path_style) snprintf(s->host, sizeof s->host, "s3.%s.amazonaws.com", s->region);
        else snprintf(s->host, sizeof s->host, "%s.s3.%s.amazonaws.com", s->bucket, s->region);
    }
    s->retries = atoi(envor("GPUCKPT_S3_RETRIES", NULL, "4"));
    pthread_once(&curl_once, curl_init_once);
    *out = s;
    return GC_OK;
}

void gc_s3_close(gc_s3 *s) { free(s); }

/* ---------------------------------------------------------------- HTTP */

typedef struct {
    char  *buf; size_t len, cap;
    size_t rd_off; const char *rd_src; size_t rd_len;   /* upload source */
    long   status;
    curl_off_t content_length;
    char   err[CURL_ERROR_SIZE];
} xfer;

static size_t on_write(char *p, size_t sz, size_t nm, void *u)
{
    xfer *x = u;
    size_t n = sz * nm;
    if (x->len + n + 1 > x->cap) {
        size_t nc = x->cap ? x->cap * 2 : 65536;
        while (nc < x->len + n + 1) nc *= 2;
        char *nb = realloc(x->buf, nc);
        if (!nb) return 0;
        x->buf = nb; x->cap = nc;
    }
    memcpy(x->buf + x->len, p, n);
    x->len += n;
    x->buf[x->len] = 0;
    return n;
}

static size_t on_read(char *p, size_t sz, size_t nm, void *u)
{
    xfer *x = u;
    size_t n = sz * nm, left = x->rd_len - x->rd_off;
    if (n > left) n = left;
    memcpy(p, x->rd_src + x->rd_off, n);
    x->rd_off += n;
    return n;
}

static void amz_date_now(char out[17])
{
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(out, 17, "%Y%m%dT%H%M%SZ", &tm);
}

/* One signed request. key may be "" for bucket-level operations; query must
 * be in canonical (sorted, encoded) form or NULL. Body is sent for PUT. */
static int s3_request(gc_s3 *s, const char *method, const char *key, const char *query,
                      const void *body, size_t blen, const char *extra_header, xfer *x)
{
    char fullkey[1024], enc_key[3072], canonical_uri[3400], url[4096];
    if (snprintf(fullkey, sizeof fullkey, "%s%s", s->prefix, key) >= (int)sizeof fullkey) return GC_EINVAL;
    if (uri_encode(fullkey, 1, enc_key, sizeof enc_key)) return GC_EINVAL;
    if (s->path_style) snprintf(canonical_uri, sizeof canonical_uri, "/%s/%s", s->bucket, enc_key);
    else snprintf(canonical_uri, sizeof canonical_uri, "/%s", enc_key);
    if (s->path_style && !*fullkey) snprintf(canonical_uri, sizeof canonical_uri, "/%s/", s->bucket);
    snprintf(url, sizeof url, "%s://%s%s%s%s", s->scheme, s->host, canonical_uri, query ? "?" : "", query ? query : "");

    uint8_t ph[GC_HASH_LEN];
    char phex[GC_HEX_LEN + 1];
    gc_sha256(body ? body : "", body ? blen : 0, ph);
    gc_hex(ph, GC_HASH_LEN, phex);

    int attempt = 0;
    for (;;) {
        char amz_date[17], auth[1024];
        amz_date_now(amz_date);
        int rc = gc_s3_sign(method, s->host, canonical_uri, query, phex, amz_date, s->region,
                            s->access_key, s->secret_key, s->token, auth, sizeof auth);
        if (rc) return rc;

        CURL *c = thread_curl();
        if (!c) return GC_ENOMEM;
        curl_easy_reset(c);
        memset(x, 0, sizeof *x);
        x->rd_src = body; x->rd_len = blen;

        struct curl_slist *hdr = NULL;
        char line[4200];
        snprintf(line, sizeof line, "Host: %s", s->host); hdr = curl_slist_append(hdr, line);
        snprintf(line, sizeof line, "x-amz-content-sha256: %s", phex); hdr = curl_slist_append(hdr, line);
        snprintf(line, sizeof line, "x-amz-date: %s", amz_date); hdr = curl_slist_append(hdr, line);
        if (s->token[0]) { snprintf(line, sizeof line, "x-amz-security-token: %s", s->token); hdr = curl_slist_append(hdr, line); }
        snprintf(line, sizeof line, "Authorization: %s", auth); hdr = curl_slist_append(hdr, line);
        hdr = curl_slist_append(hdr, "Expect:");
        if (extra_header) hdr = curl_slist_append(hdr, extra_header);

        curl_easy_setopt(c, CURLOPT_URL, url);
        curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdr);
        curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, on_write);
        curl_easy_setopt(c, CURLOPT_WRITEDATA, x);
        curl_easy_setopt(c, CURLOPT_ERRORBUFFER, x->err);
        curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 20L);
        curl_easy_setopt(c, CURLOPT_TIMEOUT, 600L);
        curl_easy_setopt(c, CURLOPT_TCP_KEEPALIVE, 1L);
        if (s->ca_bundle[0]) curl_easy_setopt(c, CURLOPT_CAINFO, s->ca_bundle);
        if (!strcmp(method, "PUT")) {
            curl_easy_setopt(c, CURLOPT_UPLOAD, 1L);
            curl_easy_setopt(c, CURLOPT_READFUNCTION, on_read);
            curl_easy_setopt(c, CURLOPT_READDATA, x);
            curl_easy_setopt(c, CURLOPT_INFILESIZE_LARGE, (curl_off_t)blen);
        } else if (!strcmp(method, "HEAD")) {
            curl_easy_setopt(c, CURLOPT_NOBODY, 1L);
        } else if (!strcmp(method, "DELETE")) {
            curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, "DELETE");
        }
        CURLcode cc = curl_easy_perform(c);
        curl_slist_free_all(hdr);
        if (cc == CURLE_OK) {
            curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &x->status);
            curl_easy_getinfo(c, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &x->content_length);
            int transient = x->status == 429 || x->status == 500 || x->status == 502 || x->status == 503 || x->status == 504;
            if (!transient) return GC_OK;
            if (attempt >= s->retries) {
                gc_set_error("%s %s: HTTP %ld after %d attempts: %.200s", method, fullkey, x->status, attempt + 1, x->buf ? x->buf : "");
                return GC_EIO;
            }
        } else {
            if (attempt >= s->retries) {
                gc_set_error("%s %s: %s", method, url, x->err[0] ? x->err : curl_easy_strerror(cc));
                return GC_EIO;
            }
        }
        free(x->buf); x->buf = NULL;
        useconds_t backoff = (useconds_t)(200000u << attempt);
        usleep(backoff > 5000000u ? 5000000u : backoff);
        attempt++;
    }
}

static int fail_status(const char *what, const char *key, const xfer *x)
{
    gc_set_error("%s %s: HTTP %ld: %.200s", what, key, x->status, x->buf ? x->buf : "");
    return GC_EIO;
}

int gc_s3_put(gc_s3 *s, const char *key, const void *data, size_t len)
{
    xfer x;
    int rc = s3_request(s, "PUT", key, NULL, data, len, "If-None-Match: *", &x);
    if (rc) return rc;
    rc = x.status == 200 ? GC_OK : x.status == 412 ? GC_EEXIST : fail_status("PUT", key, &x);
    free(x.buf);
    return rc;
}

int gc_s3_get(gc_s3 *s, const char *key, void **data, size_t *len)
{
    xfer x;
    int rc = s3_request(s, "GET", key, NULL, NULL, 0, NULL, &x);
    if (rc) return rc;
    if (x.status == 404) { free(x.buf); gc_set_error("GET %s: not found", key); return GC_ENOENT; }
    if (x.status != 200) { rc = fail_status("GET", key, &x); free(x.buf); return rc; }
    if (!x.buf) { x.buf = malloc(1); x.len = 0; }
    *data = x.buf; *len = x.len;
    return GC_OK;
}

int gc_s3_head(gc_s3 *s, const char *key, uint64_t *size)
{
    xfer x;
    int rc = s3_request(s, "HEAD", key, NULL, NULL, 0, NULL, &x);
    if (rc) return rc;
    free(x.buf);
    if (x.status == 404) return GC_ENOENT;
    if (x.status != 200) return fail_status("HEAD", key, &x);
    if (size) *size = x.content_length >= 0 ? (uint64_t)x.content_length : 0;
    return GC_OK;
}

int gc_s3_delete(gc_s3 *s, const char *key, uint64_t *size)
{
    uint64_t sz = 0;
    int rc = gc_s3_head(s, key, &sz);
    if (rc) return rc;
    xfer x;
    rc = s3_request(s, "DELETE", key, NULL, NULL, 0, NULL, &x);
    if (rc) return rc;
    free(x.buf);
    if (x.status != 204 && x.status != 200) return fail_status("DELETE", key, &x);
    if (size) *size = sz;
    return GC_OK;
}

/* minimal XML helpers for ListObjectsV2 responses */
static const char *tag(const char *s, const char *end, const char *name, size_t *len)
{
    char open[64], close[64];
    snprintf(open, sizeof open, "<%s>", name);
    snprintf(close, sizeof close, "</%s>", name);
    const char *a = s;
    while (a < end) {
        const char *f = memmem(a, (size_t)(end - a), open, strlen(open));
        if (!f) return NULL;
        f += strlen(open);
        const char *e = memmem(f, (size_t)(end - f), close, strlen(close));
        if (!e) return NULL;
        *len = (size_t)(e - f);
        return f;
    }
    return NULL;
}

static void xml_unescape(char *s)
{
    char *o = s;
    for (char *p = s; *p; ) {
        if (*p == '&') {
            if (!strncmp(p, "&amp;", 5)) { *o++ = '&'; p += 5; continue; }
            if (!strncmp(p, "&lt;", 4)) { *o++ = '<'; p += 4; continue; }
            if (!strncmp(p, "&gt;", 4)) { *o++ = '>'; p += 4; continue; }
            if (!strncmp(p, "&quot;", 6)) { *o++ = '"'; p += 6; continue; }
            if (!strncmp(p, "&apos;", 6)) { *o++ = '\''; p += 6; continue; }
        }
        *o++ = *p++;
    }
    *o = 0;
}

static int64_t parse_iso8601(const char *s, size_t n)
{
    char buf[40];
    if (n >= sizeof buf) return -1;
    memcpy(buf, s, n); buf[n] = 0;
    struct tm tm; memset(&tm, 0, sizeof tm);
    char *rest = strptime(buf, "%Y-%m-%dT%H:%M:%S", &tm);
    if (!rest) return -1;
    return (int64_t)timegm(&tm);
}

int gc_s3_list(gc_s3 *s, const char *prefix, gc_obj_list_cb cb, void *u)
{
    char token[2048] = "";
    size_t plen = strlen(s->prefix);
    for (;;) {
        char fullprefix[1024], enc_prefix[3072], enc_token[6200], query[9000];
        snprintf(fullprefix, sizeof fullprefix, "%s%s", s->prefix, prefix);
        if (uri_encode(fullprefix, 0, enc_prefix, sizeof enc_prefix)) return GC_EINVAL;
        if (token[0]) {
            if (uri_encode(token, 0, enc_token, sizeof enc_token)) return GC_EINVAL;
            snprintf(query, sizeof query, "continuation-token=%s&list-type=2&max-keys=1000&prefix=%s", enc_token, enc_prefix);
        } else {
            snprintf(query, sizeof query, "list-type=2&max-keys=1000&prefix=%s", enc_prefix);
        }
        xfer x;
        int rc = s3_request(s, "GET", "", query, NULL, 0, NULL, &x);
        if (rc) return rc;
        if (x.status != 200) { rc = fail_status("LIST", prefix, &x); free(x.buf); return rc; }
        const char *p = x.buf ? x.buf : "", *end = p + x.len;
        for (;;) {
            const char *c = memmem(p, (size_t)(end - p), "<Contents>", 10);
            if (!c) break;
            const char *ce = memmem(c, (size_t)(end - c), "</Contents>", 11);
            if (!ce) break;
            size_t kl = 0, sl = 0, ml = 0;
            const char *k = tag(c, ce, "Key", &kl);
            const char *sz = tag(c, ce, "Size", &sl);
            const char *lm = tag(c, ce, "LastModified", &ml);
            if (k && kl < 1024 && kl > plen) {
                char key[1024];
                memcpy(key, k, kl); key[kl] = 0;
                xml_unescape(key);
                gc_obj_info info = { key + plen, sz ? strtoull(sz, NULL, 10) : 0, lm ? parse_iso8601(lm, ml) : -1 };
                rc = cb(&info, u);
                if (rc) { free(x.buf); return rc; }
            }
            p = ce + 11;
        }
        size_t tl = 0;
        const char *tr = tag(x.buf ? x.buf : "", end, "IsTruncated", &tl);
        int truncated = tr && tl == 4 && !strncmp(tr, "true", 4);
        size_t ntl = 0;
        const char *nt = truncated ? tag(x.buf, end, "NextContinuationToken", &ntl) : NULL;
        if (nt && ntl < sizeof token) { memcpy(token, nt, ntl); token[ntl] = 0; xml_unescape(token); }
        else truncated = 0;
        free(x.buf);
        if (!truncated) return GC_OK;
    }
}
#endif
