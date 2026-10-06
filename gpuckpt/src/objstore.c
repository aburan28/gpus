/* Local-directory object store and dispatch to S3. */
#include "objstore.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

static int key_path(const gc_repo *r, const char *key, char *out, size_t n)
{
    int w = snprintf(out, n, "%s/%s", r->path, key);
    return (w < 0 || (size_t)w >= n) ? GC_EINVAL : GC_OK;
}

static int local_put(gc_repo *r, const char *key, const void *data, size_t len)
{
    char p[PATH_MAX], tmp[PATH_MAX];
    int rc = key_path(r, key, p, sizeof p);
    if (rc) return rc;
    char *slash = strrchr(p, '/');
    if (slash) { *slash = 0; rc = gc_mkdir_p(p); *slash = '/'; if (rc) return rc; }
    snprintf(tmp, sizeof tmp, "%s/tmp", r->path);
    return gc_write_file_atomic(p, tmp, data, len, 1);
}

static int local_get(gc_repo *r, const char *key, void **data, size_t *len)
{
    char p[PATH_MAX];
    int rc = key_path(r, key, p, sizeof p);
    if (rc) return rc;
    return gc_read_file(p, data, len);
}

static int local_head(gc_repo *r, const char *key, uint64_t *size)
{
    char p[PATH_MAX];
    int rc = key_path(r, key, p, sizeof p);
    if (rc) return rc;
    struct stat st;
    if (stat(p, &st) != 0) return GC_ENOENT;
    if (size) *size = (uint64_t)st.st_size;
    return GC_OK;
}

static int local_delete(gc_repo *r, const char *key, uint64_t *size)
{
    char p[PATH_MAX];
    int rc = key_path(r, key, p, sizeof p);
    if (rc) return rc;
    struct stat st;
    if (stat(p, &st) != 0) return GC_ENOENT;
    if (unlink(p) != 0) { gc_set_error("unlink %s: %s", p, strerror(errno)); return errno == ENOENT ? GC_ENOENT : GC_EIO; }
    if (size) *size = (uint64_t)st.st_size;
    return GC_OK;
}

static int walk(const char *dir, const char *keyprefix, gc_obj_list_cb cb, void *u)
{
    DIR *d = opendir(dir);
    if (!d) return GC_OK;
    struct dirent *e;
    int rc = GC_OK;
    while (rc == GC_OK && (e = readdir(d))) {
        if (e->d_name[0] == '.') continue;   /* ., .., .tmp.* staging files */
        char fp[PATH_MAX], key[PATH_MAX];
        if (snprintf(fp, sizeof fp, "%s/%s", dir, e->d_name) >= (int)sizeof fp) continue;
        if (snprintf(key, sizeof key, "%s%s", keyprefix, e->d_name) >= (int)sizeof key) continue;
        struct stat st;
        if (stat(fp, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            strncat(key, "/", sizeof key - strlen(key) - 1);
            rc = walk(fp, key, cb, u);
        } else if (S_ISREG(st.st_mode)) {
            gc_obj_info info = { key, (uint64_t)st.st_size, (int64_t)st.st_mtime };
            rc = cb(&info, u);
        }
    }
    closedir(d);
    return rc;
}

/* prefix is a directory prefix ending in '/' (the only form the callers use) */
static int local_list(gc_repo *r, const char *prefix, gc_obj_list_cb cb, void *u)
{
    char dir[PATH_MAX];
    size_t n = strlen(prefix);
    if (n == 0 || prefix[n - 1] != '/') return GC_EINVAL;
    if (snprintf(dir, sizeof dir, "%s/%.*s", r->path, (int)(n - 1), prefix) >= (int)sizeof dir) return GC_EINVAL;
    return walk(dir, prefix, cb, u);
}

int gc_obj_put(gc_repo *r, const char *key, const void *data, size_t len)
{
    return r->kind == GC_REPO_S3 ? gc_s3_put(r->s3, key, data, len) : local_put(r, key, data, len);
}
int gc_obj_get(gc_repo *r, const char *key, void **data, size_t *len)
{
    return r->kind == GC_REPO_S3 ? gc_s3_get(r->s3, key, data, len) : local_get(r, key, data, len);
}
int gc_obj_head(gc_repo *r, const char *key, uint64_t *size)
{
    return r->kind == GC_REPO_S3 ? gc_s3_head(r->s3, key, size) : local_head(r, key, size);
}
int gc_obj_delete(gc_repo *r, const char *key, uint64_t *size)
{
    return r->kind == GC_REPO_S3 ? gc_s3_delete(r->s3, key, size) : local_delete(r, key, size);
}
int gc_obj_list(gc_repo *r, const char *prefix, gc_obj_list_cb cb, void *u)
{
    return r->kind == GC_REPO_S3 ? gc_s3_list(r->s3, prefix, cb, u) : local_list(r, prefix, cb, u);
}
