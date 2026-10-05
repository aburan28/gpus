/* Content-addressed chunk store.
 *
 * Layout: <repo>/chunks/<hh>/<64 hex>   (hh = first byte of the hash)
 *         <repo>/tmp/                   staging for atomic writes
 *
 * A chunk file is written once. Concurrent writers of the same hash each
 * stage a private temp file; the first rename wins and the others discard
 * their copy, so the store is safe for N snapshot threads and N processes
 * without any shared in-memory state. Byte content is the only identity:
 * a chunk's name is the SHA-256 of its bytes, and readers re-hash on read.
 */
#include "internal.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

int gc_store_path(const gc_repo *r, const uint8_t *hash, char *out, size_t outlen)
{
    char hex[GC_HEX_LEN + 1];
    gc_hex(hash, GC_HASH_LEN, hex);
    int n = snprintf(out, outlen, "%s/chunks/%.2s/%s", r->path, hex, hex);
    return (n < 0 || (size_t)n >= outlen) ? GC_EINVAL : GC_OK;
}

int gc_store_has(const gc_repo *r, const uint8_t *hash)
{
    char p[PATH_MAX];
    if (gc_store_path(r, hash, p, sizeof p) != GC_OK) return 0;
    return gc_file_exists(p);
}

int gc_store_put(const gc_repo *r, const uint8_t *hash, const void *data, size_t len, int *was_new)
{
    char p[PATH_MAX], tmp[PATH_MAX];
    int rc = gc_store_path(r, hash, p, sizeof p);
    if (rc) return rc;
    if (was_new) *was_new = 0;
    if (gc_file_exists(p)) return GC_OK;
    /* ensure the fan-out directory exists */
    char *slash = strrchr(p, '/');
    *slash = 0;
    rc = gc_mkdir_p(p);
    *slash = '/';
    if (rc) return rc;
    snprintf(tmp, sizeof tmp, "%s/tmp", r->path);
    rc = gc_write_file_atomic(p, tmp, data, len, 1);
    if (rc == GC_EEXIST) return GC_OK;   /* another writer landed first: same bytes */
    if (rc == GC_OK && was_new) *was_new = 1;
    return rc;
}

/* Read a chunk into buf, verifying its hash when verify is set (callers that
 * time hashing separately pass 0 and hash themselves). Returns GC_ENOENT, GC_ECORRUPT,
 * GC_EINVAL (buffer too small) or GC_OK with *len_out set. */
int gc_store_get(const gc_repo *r, const uint8_t *hash, void *buf, size_t buflen, size_t *len_out, int verify)
{
    char p[PATH_MAX];
    int rc = gc_store_path(r, hash, p, sizeof p);
    if (rc) return rc;
    void *data = NULL;
    size_t n = 0;
    rc = gc_read_file(p, &data, &n);
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

int gc_store_delete(const gc_repo *r, const uint8_t *hash, uint64_t *bytes)
{
    char p[PATH_MAX];
    int rc = gc_store_path(r, hash, p, sizeof p);
    if (rc) return rc;
    struct stat st;
    if (stat(p, &st) != 0) return GC_ENOENT;
    if (unlink(p) != 0) { gc_set_error("unlink %s: %s", p, strerror(errno)); return GC_EIO; }
    if (bytes) *bytes = (uint64_t)st.st_size;
    return GC_OK;
}

int gc_store_walk(const gc_repo *r, int (*cb)(const uint8_t *hash, const char *path, void *u), void *u)
{
    char root[PATH_MAX];
    snprintf(root, sizeof root, "%s/chunks", r->path);
    DIR *d = opendir(root);
    if (!d) return GC_OK; /* empty store */
    struct dirent *e;
    int rc = GC_OK;
    while (rc == GC_OK && (e = readdir(d))) {
        if (strlen(e->d_name) != 2) continue;
        char sub[PATH_MAX];
        snprintf(sub, sizeof sub, "%s/%s", root, e->d_name);
        DIR *sd = opendir(sub);
        if (!sd) continue;
        struct dirent *f;
        while (rc == GC_OK && (f = readdir(sd))) {
            if (strlen(f->d_name) != GC_HEX_LEN) continue;
            uint8_t h[GC_HASH_LEN];
            if (gc_unhex(f->d_name, h, GC_HASH_LEN) != GC_OK) continue;
            char fp[PATH_MAX];
            snprintf(fp, sizeof fp, "%s/%s", sub, f->d_name);
            rc = cb(h, fp, u);
        }
        closedir(sd);
    }
    closedir(d);
    return rc;
}
