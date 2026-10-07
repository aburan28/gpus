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
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>

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

/* ------------------------------------------------------- direct (GDS) I/O */

static int open_odirect(const char *path, int flags, int odirect, int *used)
{
    int fd = -1;
    *used = 0;
    if (odirect) {
        fd = open(path, flags | O_DIRECT, 0644);
        if (fd >= 0) { *used = 1; return fd; }
        if (errno != EINVAL) return -1;          /* EINVAL: filesystem has no O_DIRECT (tmpfs) */
    }
    return open(path, flags, 0644);
}

/* Local repos only. fill() writes exactly len bytes into a fresh file (from
 * device memory, via GPUDirect Storage); the file is then fsynced and
 * published under the chunk's name with the same write-once link(2) rule as
 * gc_store_put. *was_new is 0 when the chunk already existed or another
 * writer won the race. */
int gc_store_put_direct(gc_repo *r, const uint8_t *hash, uint64_t len, int odirect,
                        int (*fill)(int fd, void *u), void *u, int *was_new, int *odirect_used)
{
    if (r->kind != GC_REPO_LOCAL) return GC_EINVAL;
    char key[128], final_path[PATH_MAX], tmp[PATH_MAX];
    int rc = gc_store_key(hash, key, sizeof key);
    if (rc) return rc;
    if (was_new) *was_new = 0;
    snprintf(final_path, sizeof final_path, "%s/%s", r->path, key);
    if (gc_file_exists(final_path)) return GC_OK;
    char *slash = strrchr(final_path, '/');
    *slash = 0; rc = gc_mkdir_p(final_path); *slash = '/';
    if (rc) return rc;
    static __thread unsigned seq;
    snprintf(tmp, sizeof tmp, "%s/tmp/.tmp.%ld.%lu.d%u", r->path, (long)getpid(), (unsigned long)pthread_self(), seq++);
    int used = 0;
    int fd = open_odirect(tmp, O_RDWR | O_CREAT | O_EXCL, odirect, &used);
    if (fd < 0) { gc_set_error("open %s: %s", tmp, strerror(errno)); return GC_EIO; }
    if (odirect_used) *odirect_used = used;
    rc = fill(fd, u);
    struct stat st;
    if (rc == GC_OK && (fstat(fd, &st) != 0 || (uint64_t)st.st_size != len)) {
        gc_set_error("direct write produced %lld of %llu bytes", (long long)st.st_size, (unsigned long long)len);
        rc = GC_EIO;
    }
    if (rc == GC_OK && fsync(fd) != 0) { gc_set_error("fsync %s: %s", tmp, strerror(errno)); rc = GC_EIO; }
    close(fd);
    if (rc == GC_OK) {
        if (link(tmp, final_path) == 0) { if (was_new) *was_new = 1; }
        else if (errno != EEXIST) { gc_set_error("link %s: %s", final_path, strerror(errno)); rc = GC_EIO; }
    }
    unlink(tmp);
    return rc;
}

/* Local repos only: open a stored chunk for direct reading into device
 * memory. GC_ENOENT if absent. */
int gc_store_open_chunk(gc_repo *r, const uint8_t *hash, int odirect, int *fd_out, uint64_t *size, int *odirect_used)
{
    if (r->kind != GC_REPO_LOCAL) return GC_EINVAL;
    char key[128], path[PATH_MAX];
    int rc = gc_store_key(hash, key, sizeof key);
    if (rc) return rc;
    snprintf(path, sizeof path, "%s/%s", r->path, key);
    int used = 0;
    int fd = open_odirect(path, O_RDONLY, odirect, &used);
    if (fd < 0) { if (errno == ENOENT) return GC_ENOENT; gc_set_error("open %s: %s", path, strerror(errno)); return GC_EIO; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return GC_EIO; }
    *fd_out = fd;
    *size = (uint64_t)st.st_size;
    if (odirect_used) *odirect_used = used;
    return GC_OK;
}
