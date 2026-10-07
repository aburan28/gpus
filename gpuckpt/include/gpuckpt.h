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

    /* Required for snapshot (read) / restore (write): synchronous copies of
     * len bytes at byte offset off of device d. Called concurrently from
     * worker threads; must be thread-safe. */
    int   (*read)(gc_image *img, uint32_t d, uint64_t off, void *dst, size_t len);
    int   (*write)(gc_image *img, uint32_t d, uint64_t off, const void *src, size_t len);

    /* Optional per-worker state (e.g. CUDA streams). */
    int   (*thread_attach)(gc_image *img, void **tctx);
    void  (*thread_detach)(gc_image *img, void *tctx);

    /* Optional asynchronous copies through staging slot `slot`
     * (0 <= slot < pipeline depth). io_wait blocks until that slot's copy
     * has finished and returns its status. Without these the library falls
     * back to read/write and there is no copy/compute overlap. */
    int   (*read_start)(gc_image *img, void *tctx, int slot, uint32_t d, uint64_t off, void *dst, size_t len);
    int   (*write_start)(gc_image *img, void *tctx, int slot, uint32_t d, uint64_t off, const void *src, size_t len);
    int   (*io_wait)(gc_image *img, void *tctx, int slot);

    /* Optional page-locking of staging memory so DMA engines can use it
     * directly. host_register pins memory the library allocated (keeps
     * huge-page backing); host_alloc_pinned lets the backend allocate it.
     * On failure they return nonzero / NULL and write a reason into why. */
    int   (*host_register)(gc_image *img, void *p, size_t len, char *why, size_t whylen);
    void  (*host_unregister)(gc_image *img, void *p);
    void *(*host_alloc_pinned)(gc_image *img, size_t len, char *why, size_t whylen);
    void  (*host_free_pinned)(gc_image *img, void *p);

    /* Optional device-side hashing: SHA-256 of every chunk of device d,
     * written to out[0 .. ceil(size/chunk_size)-1]. Identical digests to
     * the CPU path. */
    int   (*hash_chunks)(gc_image *img, uint32_t d, uint64_t chunk_size, uint8_t (*out)[GC_HASH_LEN]);

    /* Optional direct storage I/O (GPUDirect Storage): move len bytes
     * between device offset off and file fd (file offset 0) without host
     * staging. direct_odirect: open files with O_DIRECT for this path. */
    int   (*dev_to_file)(gc_image *img, void *tctx, uint32_t d, uint64_t off, size_t len, int fd);
    int   (*file_to_dev)(gc_image *img, void *tctx, uint32_t d, uint64_t off, size_t len, int fd);
    int    direct_odirect;

    void *priv;
};

/* Host-file backend: one file per device. mode "r" opens read-only (snapshot
 * source), "w" creates/truncates files to the sizes given (restore target). */
int  gc_image_file_open(gc_image *img, const char *mode, uint32_t n,
                        const char *const *paths, const uint64_t *sizes);
void gc_image_file_close(gc_image *img);

/* ------------------------------------------------------------- I/O opts --- */

enum { GC_HOSTMEM_MALLOC = 0, GC_HOSTMEM_THP, GC_HOSTMEM_HUGETLB, GC_HOSTMEM_HUGETLB_1G };
enum { GC_PIN_NONE = 0, GC_PIN_REGISTER, GC_PIN_ALLOC };
enum { GC_OFF = 0, GC_ON = 1 };

/* How data moves. Every option degrades gracefully: what was requested and
 * what was obtained are both reported in gc_io_report. */
typedef struct gc_io_opts {
    int hostmem;        /* staging memory: GC_HOSTMEM_*; default THP */
    int mlock;          /* mlock(2) the staging arena; default off */
    int pin;            /* GC_PIN_*: page-lock staging for DMA; default REGISTER */
    int pipeline;       /* staging slots per worker: 1 = no overlap, 2 = double buffer (default) */
    int gpu_hash;       /* hash chunks on the GPU (GC_ON/GC_OFF); default OFF, see docs/gates.md */
    int gpu_hash_check; /* chunks re-hashed on the CPU to cross-check GPU digests; default 8 */
    int gds;            /* GPUDirect Storage to/from a local repo (GC_ON/GC_OFF); default OFF */
} gc_io_opts;

void gc_io_opts_default(gc_io_opts *o);


typedef struct gc_io_report {
    char hostmem[192];
    char mlock[128];
    char pin[160];
    char pipeline[96];
    char gpu_hash[224];
    char gds[224];
} gc_io_report;

/* ----------------------------------------------------------------- repo --- */

typedef struct gc_repo gc_repo;

/* path is a directory or an s3://bucket[/prefix] url. */
int      gc_repo_init(const char *path, uint64_t chunk_size);
int      gc_repo_open(const char *path, gc_repo **out);
void     gc_repo_close(gc_repo *r);
uint64_t gc_repo_chunk_size(const gc_repo *r);
const char *gc_repo_path(const gc_repo *r);
const char *gc_repo_kind(const gc_repo *r);        /* "local" or "s3" */
/* gc never deletes an unreferenced chunk younger than this. On a local repo
 * the exclusive lock makes it unnecessary (default 0); on S3, where there is
 * no lock, it protects chunks a concurrent snapshot has just deduplicated
 * against (default 3600). */
void gc_repo_set_gc_grace(gc_repo *r, int seconds);
const char *gc_s3_backend(void);                   /* "libcurl" or "unavailable" */

/* Staging memory prepared ahead of time, so allocation, pre-faulting and
 * DMA registration happen before the target is locked rather than inside
 * the checkpoint pause. pin_with supplies host_register/host_alloc_pinned
 * (e.g. gc_cuda_pinning_image) or is NULL. An operation uses the arena if it
 * is large enough (see gc_staging_bytes) and otherwise allocates its own. */
typedef struct gc_staging gc_staging;
size_t gc_staging_bytes(const gc_repo *r, int threads, const gc_io_opts *io);
int    gc_staging_create(gc_staging **out, size_t bytes, const gc_io_opts *io, gc_image *pin_with,
                         gc_io_report *rep, uint64_t *ns_setup);
void   gc_staging_destroy(gc_staging *s);

/* ------------------------------------------------------------- snapshot --- */

typedef struct gc_stats {
    uint64_t bytes_total;        /* image bytes seen */
    uint64_t chunks_total;
    uint64_t chunks_new;         /* chunks written to the store */
    uint64_t bytes_new;
    uint64_t chunks_same_as_parent; /* informational: same hash at same (dev,idx) */
    uint64_t chunks_missing;     /* verify/restore: not in store */
    uint64_t chunks_corrupt;     /* verify/restore: hash mismatch */
    uint64_t ns_read;            /* summed over threads; with pipelining, only time spent waiting */
    uint64_t ns_hash;
    uint64_t ns_write;           /* store writes (snapshot) / image writes (restore) */
    uint64_t ns_wall;
    uint64_t bytes_staged;       /* image bytes that passed through host staging buffers */
    uint64_t bytes_direct;       /* image bytes moved by direct storage I/O (GDS), never in host memory */
    uint64_t ns_gpu_hash;        /* device-side hashing, wall time */
    uint64_t chunks_cpu_checked; /* GPU digests re-verified on the CPU */
    uint64_t hostmem_huge_kb;    /* staging arena kB backed by huge pages */
    int      threads;
} gc_stats;

typedef struct gc_snapshot_opts {
    const char *id;       /* NULL: generated */
    const char *parent;   /* NULL or a snapshot id; reporting only */
    int         pid;      /* recorded, 0 if none */
    int         threads;  /* 0: auto */
    const char *note;     /* free text, single line, may be NULL */
    const gc_io_opts *io; /* NULL: gc_io_opts_default */
    gc_io_report *report; /* optional out: what the I/O options actually obtained */
    gc_staging  *staging; /* optional pre-built staging arena */
} gc_snapshot_opts;

typedef struct gc_restore_opts {
    int         threads;  /* 0: auto */
    const gc_io_opts *io; /* NULL: gc_io_opts_default */
    gc_io_report *report; /* optional out */
    gc_staging  *staging; /* optional pre-built staging arena */
} gc_restore_opts;

int gc_snapshot_create(gc_repo *r, gc_image *img, const gc_snapshot_opts *o,
                       gc_stats *st, char id_out[GC_ID_LEN]);
int gc_snapshot_restore(gc_repo *r, const char *id, gc_image *img,
                        const gc_restore_opts *o, gc_stats *st);
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

/* Acceleration, enabled before *_begin so the image exposes it.
 * gpu_hash: compile the SHA-256 kernel with NVRTC (GPUCKPT_LIBNVRTC, else
 * libnvrtc.so.{13,12}, libnvrtc.so) or load prebuilt PTX from
 * GPUCKPT_KERNEL_PTX. gds: open libcufile (GPUCKPT_LIBCUFILE, else
 * libcufile.so.0) and the cuFile driver. On failure the reason is in why and
 * the feature stays off; nothing else changes. */
int  gc_cuda_enable_gpu_hash(gc_cuda *c, char *why, size_t whylen);
/* A stub image carrying only the pinning entry points, for gc_staging_create
 * before any checkpoint operation exists. Release with gc_cuda_image_close. */
int  gc_cuda_pinning_image(gc_cuda *c, gc_image *img);
int  gc_cuda_enable_gds(gc_cuda *c, char *why, size_t whylen);

#ifdef __cplusplus
}
#endif
#endif
