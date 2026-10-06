#ifndef GC_INTERNAL_H
#define GC_INTERNAL_H
#define _GNU_SOURCE
#pragma GCC diagnostic ignored "-Wformat-truncation"
#include "gpuckpt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <limits.h>

enum { GC_REPO_LOCAL = 0, GC_REPO_S3 = 1 };
struct gc_s3;

struct gc_repo {
    char     path[PATH_MAX];   /* local: directory; s3: the s3:// url */
    int      kind;
    uint64_t chunk_size;
    int      lock_fd;          /* local: repo/lock, flock()ed shared during snapshot/restore,
                                  exclusive during gc; -1 on s3 (no locking, see gc grace) */
    struct gc_s3 *s3;
    int      gc_grace_seconds; /* gc leaves unreferenced chunks younger than this alone */
};

/* util.c */
uint64_t gc_mono_ns(void);
uint64_t gc_wall_ns(void);
void     gc_hex(const uint8_t *h, size_t n, char *out);      /* out has 2n+1 bytes */
int      gc_unhex(const char *hex, uint8_t *out, size_t n);
int      gc_mkdir_p(const char *path);
int      gc_write_file_atomic(const char *final_path, const char *tmp_dir,
                              const void *data, size_t len, int allow_exists);
int      gc_read_file(const char *path, void **data, size_t *len);
int      gc_file_exists(const char *path);
void     gc_set_error(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void     gc_gen_id(char out[GC_ID_LEN]);
int      gc_valid_id(const char *id);
int      gc_nprocs(void);

/* sha256.c */
void gc_sha256(const void *data, size_t len, uint8_t out[GC_HASH_LEN]);
typedef struct gc_sha256_ctx { uint8_t opaque[256]; } gc_sha256_ctx;
void gc_sha256_init(gc_sha256_ctx *c);
void gc_sha256_update(gc_sha256_ctx *c, const void *data, size_t len);
void gc_sha256_final(gc_sha256_ctx *c, uint8_t out[GC_HASH_LEN]);
const char *gc_sha256_impl(void);

/* store.c */
int gc_store_key(const uint8_t *hash, char *out, size_t outlen);          /* "chunks/hh/<hex>" */
int gc_store_key_parse(const char *key, uint8_t *hash);                   /* inverse; GC_EINVAL if not a chunk key */
int gc_store_has(gc_repo *r, const uint8_t *hash);
int gc_store_put(gc_repo *r, const uint8_t *hash, const void *data, size_t len, int *was_new);
int gc_store_get(gc_repo *r, const uint8_t *hash, void *buf, size_t buflen, size_t *len_out, int verify);
int gc_store_delete(gc_repo *r, const uint8_t *hash, uint64_t *bytes);

/* manifest.c */
int gc_manifest_key(const char *id, char *out, size_t outlen);           /* "snapshots/<id>.manifest" */
int gc_manifest_save(gc_repo *r, const gc_manifest *m);
int gc_manifest_alloc_dev(gc_manifest *m, uint32_t d, uint64_t size, uint64_t chunk_size);

/* repo locking */
int gc_repo_lock(gc_repo *r, int exclusive);
void gc_repo_unlock(gc_repo *r);

#endif
