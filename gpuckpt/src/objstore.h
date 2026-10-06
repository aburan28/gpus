/* Object-storage layer: the five operations the chunk store, manifests and
 * gc need, with a local-directory implementation and an S3 implementation.
 * Keys are the same in both ("chunks/<hh>/<hex>", "snapshots/<id>.manifest",
 * "config"), so `aws s3 sync` of a local repo produces a valid S3 repo and
 * vice versa. Every write is write-once: gc_obj_put returns GC_EEXIST when
 * the key already exists and never overwrites. */
#ifndef GC_OBJSTORE_H
#define GC_OBJSTORE_H
#include "internal.h"

typedef struct gc_obj_info {
    const char *key;      /* full key, e.g. "chunks/ab/ab12..." */
    uint64_t    size;
    int64_t     mtime;    /* unix seconds, -1 if unknown */
} gc_obj_info;
typedef int (*gc_obj_list_cb)(const gc_obj_info *info, void *u);

int gc_obj_put(gc_repo *r, const char *key, const void *data, size_t len);
int gc_obj_get(gc_repo *r, const char *key, void **data, size_t *len);
int gc_obj_head(gc_repo *r, const char *key, uint64_t *size);
int gc_obj_delete(gc_repo *r, const char *key, uint64_t *size);
int gc_obj_list(gc_repo *r, const char *prefix, gc_obj_list_cb cb, void *u);

/* S3 client (src/s3.c). Configuration comes from the repo URL
 * s3://bucket[/prefix] and the environment: AWS_ACCESS_KEY_ID,
 * AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional), AWS_REGION or
 * AWS_DEFAULT_REGION (default us-east-1), GPUCKPT_S3_ENDPOINT (e.g.
 * http://127.0.0.1:9000 for MinIO or a mock; default
 * https://s3.<region>.amazonaws.com), GPUCKPT_S3_PATH_STYLE (1 forces
 * path-style; default 1 when an endpoint is given, 0 otherwise),
 * AWS_CA_BUNDLE (TLS trust file). */
typedef struct gc_s3 gc_s3;
int  gc_s3_open(const char *url, gc_s3 **out);
void gc_s3_close(gc_s3 *s);
int  gc_s3_put(gc_s3 *s, const char *key, const void *data, size_t len);
int  gc_s3_get(gc_s3 *s, const char *key, void **data, size_t *len);
int  gc_s3_head(gc_s3 *s, const char *key, uint64_t *size);
int  gc_s3_delete(gc_s3 *s, const char *key, uint64_t *size);
int  gc_s3_list(gc_s3 *s, const char *prefix, gc_obj_list_cb cb, void *u);
const char *gc_s3_impl(void);          /* "libcurl" or "unavailable" */

/* Signature V4 primitives, exposed for the test suite. */
void gc_hmac_sha256(const void *key, size_t klen, const void *data, size_t dlen, uint8_t out[GC_HASH_LEN]);
int  gc_s3_sign(const char *method, const char *host, const char *canonical_uri,
                const char *canonical_query, const char *payload_hash_hex,
                const char *amz_date, const char *region, const char *access_key,
                const char *secret_key, const char *session_token,
                char *authorization, size_t authorization_len);
#endif
