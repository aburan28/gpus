/* gpuckpt public API.
 *
 * Layers (bottom to top):
 *   store     content-addressed chunk store on disk, write-once per hash
 *   manifest  per-snapshot record: devices, sizes, ordered chunk hashes, stats
 *   snapshot  create / restore / verify / forget / gc over an abstract image
 *   image     abstract memory image: N devices, each a contiguous byte range,
 *             served by a backend (host files, or CUDA custom-storage mappings)
 *
 * Every snapshot manifest is complete on its own: restoration never depends
 * on another manifest, only on the chunks it lists. "Incremental" therefore
 * means only chunks absent from the store are written, never that a manifest
 * references a parent. The parent id is recorded for reporting only.
 */
#ifndef GPUCKPT_H
#define GPUCKPT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GC_VERSION        "0.1.0"
#define GC_HASH_LEN       32
#define GC_HEX_LEN        64
#define GC_ID_LEN         64
#define GC_UUID_LEN       72
#define GC_MAX_DEVICES    32
#define GC_FORMAT_VERSION 1

/* Error codes. Negative. Infra failures are never "the data is bad". */
enum {
    GC_OK            = 0,
    GC_EINVAL        = -1,   /* bad argument / bad format */
    GC_EIO           = -2,   /* OS I/O failure (errno set where possible) */
    GC_ENOENT        = -3,   /* snapshot / chunk / repo missing */
    GC_EEXIST        = -4,   /* immutable record already exists */
    GC_ECORRUPT      = -5,   /* stored bytes do not match their hash */
    GC_EMISMATCH     = -6,   /* image shape does not match manifest */
    GC_EBACKEND      = -7,   /* backend (CUDA) failure; see gc_last_error() */
    GC_ESTATE        = -8,   /* target process in wrong checkpoint state */
    GC_ENOMEM        = -9,
};

const char *gc_strerror(int rc);
/* Last backend error message for this thread (empty string if none). */
const char *gc_last_error(void);

/* ---------------------------------------------------------------- image --- */

typedef struct gc_image gc_image;

typedef struct gc_image_dev {
    uint64_t size;              /* bytes in this device's image */
    char     uuid[GC_UUID_LEN]; /* "GPU-..." or "-" when unknown */
    int      ordinal;           /* backend device ordinal, -1 if unknown */
    void    *handle;            /* backend private */
} gc_image_dev;

struct gc_image {
    uint32_t     device_count;
    gc_image_dev dev[GC_MAX_DEVICES];
    char         backend[32];   /* "file", "cuda", ... recorded in manifest */

    /* Copy len bytes at byte offset off of device d into/out of host memory.
     * Called concurrently from worker threads; must be thread-safe. */
    int   (*read)(gc_image *img, uint32_t d, uint64_t off, void *dst, size_t len);
    int   (*write)(gc_image *img, uint32_t d, uint64_t off, const void *src, size_t len);
    /* Optional per-thread setup/teardown (e.g. make a CUDA context current). */
    int   (*thread_attach)(gc_image *img);
    void  (*thread_detach)(gc_image *img);
    /* Optional staging-buffer allocator (e.g. pinned host memory). */
    void *(*buf_alloc)(gc_image *img, size_t len);
    void  (*buf_free)(gc_image *img, void *p, size_t len);

    void *priv;
};

/* Host-file backend: one file per device. mode "r" opens read-only (snapshot
 * source), "w" creates/truncates files to the sizes given (restore target). */
int  gc_image_file_open(gc_image *img, const char *mode, uint32_t n,
                        const char *const *paths, const uint64_t *sizes);
void gc_image_file_close(gc_image *img);

/* ----------------------------------------------------------------- repo --- */

typedef struct gc_repo gc_repo;

int      gc_repo_init(const char *path, uint64_t chunk_size);
int      gc_repo_open(const char *path, gc_repo **out);
void     gc_repo_close(gc_repo *r);
uint64_t gc_repo_chunk_size(const gc_repo *r);
const char *gc_repo_path(const gc_repo *r);

/* ------------------------------------------------------------- snapshot --- */

typedef struct gc_stats {
    uint64_t bytes_total;        /* image bytes seen */
    uint64_t chunks_total;
    uint64_t chunks_new;         /* chunks written to the store */
    uint64_t bytes_new;
    uint64_t chunks_same_as_parent; /* informational: same hash at same (dev,idx) */
    uint64_t chunks_missing;     /* verify/restore: not in store */
    uint64_t chunks_corrupt;     /* verify/restore: hash mismatch */
    uint64_t ns_read;            /* summed over threads */
    uint64_t ns_hash;
    uint64_t ns_write;           /* store writes (snapshot) / image writes (restore) */
    uint64_t ns_wall;
    int      threads;
} gc_stats;

typedef struct gc_snapshot_opts {
    const char *id;       /* NULL: generated */
    const char *parent;   /* NULL or a snapshot id; reporting only */
    int         pid;      /* recorded, 0 if none */
    int         threads;  /* 0: auto */
    const char *note;     /* free text, single line, may be NULL */
} gc_snapshot_opts;

int gc_snapshot_create(gc_repo *r, gc_image *img, const gc_snapshot_opts *o,
                       gc_stats *st, char id_out[GC_ID_LEN]);
int gc_snapshot_restore(gc_repo *r, const char *id, gc_image *img, int threads,
                        gc_stats *st);
int gc_snapshot_verify(gc_repo *r, const char *id, gc_stats *st);
int gc_snapshot_forget(gc_repo *r, const char *id);
int gc_repo_gc(gc_repo *r, uint64_t *chunks_deleted, uint64_t *bytes_freed);
int gc_repo_list(gc_repo *r, int (*cb)(const char *id, void *u), void *u);

/* ------------------------------------------------------------- manifest --- */

typedef struct gc_manifest_dev {
    uint64_t  size;
    char      uuid[GC_UUID_LEN];
    int       ordinal;
    uint64_t  nchunks;
    uint8_t (*hashes)[GC_HASH_LEN];
} gc_manifest_dev;

typedef struct gc_manifest {
    char     id[GC_ID_LEN];
    char     parent[GC_ID_LEN];   /* "-" if none */
    uint64_t created_unix_ns;
    int      pid;
    uint64_t chunk_size;
    char     backend[32];
    char     note[256];
    uint32_t device_count;
    gc_manifest_dev dev[GC_MAX_DEVICES];
    gc_stats stats;
} gc_manifest;

int  gc_manifest_load(gc_repo *r, const char *id, gc_manifest **out);
void gc_manifest_free(gc_manifest *m);
void gc_manifest_print(const gc_manifest *m, int with_chunks, void *fp);

/* ----------------------------------------------------------------- cuda --- */

/* Thin adapter over the CUDA driver checkpoint API. libcuda is dlopen'ed at
 * runtime (path: GPUCKPT_LIBCUDA env, else "libcuda.so.1"), so the binary
 * has no link-time CUDA dependency and symbol resolution is an explicit,
 * reportable step. */
typedef struct gc_cuda gc_cuda;

enum gc_proc_state {
    GC_PS_RUNNING = 0, GC_PS_LOCKED, GC_PS_CHECKPOINTING,
    GC_PS_CHECKPOINTED, GC_PS_RESTORING, GC_PS_FAILED, GC_PS_UNKNOWN
};

int  gc_cuda_open(gc_cuda **out, const char *libpath);
void gc_cuda_close(gc_cuda *c);
const char *gc_cuda_header_mode(void);           /* "real" or "mock (UNVERIFIED)" */
const char *gc_proc_state_name(int s);
int  gc_cuda_get_state(gc_cuda *c, int pid, int *state);
int  gc_cuda_lock(gc_cuda *c, int pid, unsigned timeout_ms);
int  gc_cuda_unlock(gc_cuda *c, int pid);
int  gc_cuda_restore_thread_id(gc_cuda *c, int pid, int *tid);
/* Begin a custom-storage checkpoint; on success img describes the mapped
 * GPU memory (read-only). Target must be LOCKED; becomes CHECKPOINTING. */
int  gc_cuda_checkpoint_begin(gc_cuda *c, int pid, gc_image *img);
/* Begin a custom-storage restore; on success img describes the mapped GPU
 * memory to fill. Target must be CHECKPOINTED; becomes RESTORING. */
int  gc_cuda_restore_begin(gc_cuda *c, int pid, gc_image *img);
/* Complete the operation begun on img. Checkpoint -> CHECKPOINTED (GPU
 * memory released), restore -> LOCKED. Mappings in img are invalid after. */
int  gc_cuda_op_complete(gc_cuda *c, gc_image *img);
void gc_cuda_image_close(gc_cuda *c, gc_image *img);

#ifdef __cplusplus
}
#endif
#endif
