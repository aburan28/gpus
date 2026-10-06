/* Content-addressed chunk store over the object layer.
 *
 * Key: chunks/<hh>/<64 hex>   (hh = first byte of the hash)
 *
 * A chunk is written once. Concurrent writers of the same hash race on the
 * object layer's write-once put (link(2) locally, If-None-Match on S3); the
 * first wins, the rest see GC_EEXIST and report the chunk as not new. Byte
 * content is the only identity: a chunk's name is the SHA-256 of its bytes,
 * and readers may re-hash on read.
 */
#include "objstore.h"

int gc_store_key(const uint8_t *hash, char *out, size_t outlen)
{
    char hex[GC_HEX_LEN + 1];
    gc_hex(hash, GC_HASH_LEN, hex);
    int n = snprintf(out, outlen, "chunks/%.2s/%s", hex, hex);
    return (n < 0 || (size_t)n >= outlen) ? GC_EINVAL : GC_OK;
}

int gc_store_key_parse(const char *key, uint8_t *hash)
{
    /* chunks/hh/<64 hex> = 7 + 2 + 1 + 64 */
    if (strlen(key) != 74 || strncmp(key, "chunks/", 7) != 0 || key[9] != '/') return GC_EINVAL;
    if (strncmp(key + 7, key + 10, 2) != 0) return GC_EINVAL;
    return gc_unhex(key + 10, hash, GC_HASH_LEN);
}

int gc_store_has(gc_repo *r, const uint8_t *hash)
{
    char k[128];
    if (gc_store_key(hash, k, sizeof k)) return 0;
    return gc_obj_head(r, k, NULL) == GC_OK;
}

int gc_store_put(gc_repo *r, const uint8_t *hash, const void *data, size_t len, int *was_new)
{
    char k[128];
    int rc = gc_store_key(hash, k, sizeof k);
    if (rc) return rc;
    if (was_new) *was_new = 0;
    if (gc_obj_head(r, k, NULL) == GC_OK) return GC_OK;
    rc = gc_obj_put(r, k, data, len);
    if (rc == GC_EEXIST) return GC_OK;   /* another writer landed first: same bytes */
    if (rc == GC_OK && was_new) *was_new = 1;
    return rc;
}

/* Read a chunk into buf, verifying its hash when verify is set (callers that
 * time hashing separately pass 0 and hash themselves). GC_ENOENT, GC_ECORRUPT,
 * GC_EINVAL (buffer too small) or GC_OK with *len_out set. */
int gc_store_get(gc_repo *r, const uint8_t *hash, void *buf, size_t buflen, size_t *len_out, int verify)
{
    char k[128];
    int rc = gc_store_key(hash, k, sizeof k);
    if (rc) return rc;
    void *data = NULL;
    size_t n = 0;
    rc = gc_obj_get(r, k, &data, &n);
    if (rc) return rc;
    if (n > buflen) { free(data); return GC_EINVAL; }
    if (verify) {
        uint8_t h[GC_HASH_LEN];
        gc_sha256(data, n, h);
        if (memcmp(h, hash, GC_HASH_LEN) != 0) { free(data); return GC_ECORRUPT; }
    }
    memcpy(buf, data, n);
    free(data);
    *len_out = n;
    return GC_OK;
}

int gc_store_delete(gc_repo *r, const uint8_t *hash, uint64_t *bytes)
{
    char k[128];
    int rc = gc_store_key(hash, k, sizeof k);
    if (rc) return rc;
    return gc_obj_delete(r, k, bytes);
}
