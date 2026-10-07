/* Mock libcuda for the test suite.
 *
 * Implements the checkpoint state machine from the CUDA 13.4 reference over
 * host memory, so the adapter's full control flow runs without a GPU:
 *
 *   RUNNING --lock--> LOCKED --checkpoint(custom)--> CHECKPOINTING
 *     --complete--> CHECKPOINTED --restore(custom)--> RESTORING
 *     --complete--> LOCKED --unlock--> RUNNING
 *
 * The "GPU memory" of target pid is the set of files
 *   $GPUCKPT_MOCK_STATE_DIR/<pid>.mem.<i>
 * seeded on first contact from the comma-separated list in
 * $GPUCKPT_MOCK_IMAGES. Completing a checkpoint DELETES them (the real
 * driver releases GPU memory), completing a restore recreates them from
 * what the helper copied in. Tests read and modify them directly.
 *
 * Process state persists in $GPUCKPT_MOCK_STATE_DIR/<pid>.state so separate
 * CLI invocations (snapshot, then restore) see one target.
 *
 * Kernels: cuLaunchKernel runs the per-chunk code of
 * src/kernels/sha256_chunks.cu, compiled here as C, once per grid thread, so
 * device hashing is exercised with the exact arithmetic the GPU would run.
 *
 * Accounting: when GPUCKPT_MOCK_COUNTERS names a directory, the process
 * writes cuda.txt there at exit with the bytes moved device->host and
 * host->device, kernel launches, streams, and pinned-memory registrations.
 * The tests use it to prove which bytes did and did not cross into host
 * memory.
 *
 * Fault injection:
 *   GPUCKPT_MOCK_LOCK_TIMEOUT=1      lock returns CUDA_ERROR_TIMEOUT, stays RUNNING
 *   GPUCKPT_MOCK_FAIL_COMPLETE=1     complete returns CUDA_ERROR_UNKNOWN
 *   GPUCKPT_MOCK_KERNEL_FAULT=1      kernel faults (CUDA_ERROR_ILLEGAL_ADDRESS at sync)
 *   GPUCKPT_MOCK_KERNEL_CORRUPT=1    kernel writes a wrong digest for chunk 0
 *   GPUCKPT_MOCK_REGISTER_FAIL=1     cuMemHostRegister returns CUDA_ERROR_NOT_SUPPORTED
 *   GPUCKPT_MOCK_ALLOC_INFO=1        driver allocates the info struct instead of
 *                                    filling the caller's (both are plausible
 *                                    readings of the reference; adapter handles both)
 * Not thread-safe across processes beyond what the state file gives; the
 * test suite never races two helpers on one pid.
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAXDEV 8

#include "kernels/sha256_chunks.cu"

/* ---- accounting ----
 * Worker threads copy concurrently, so every counter update is atomic;
 * plain increments lose updates under contention and the tests compare
 * these counts exactly. */
static unsigned long long c_dtoh, c_htod, c_async, c_launch, c_streams, c_register, c_register_bytes, c_alloc_host;
#define CADD(var, n) __atomic_fetch_add(&(var), (unsigned long long)(n), __ATOMIC_RELAXED)

__attribute__((destructor)) static void dump_counters(void)
{
    const char *dir = getenv("GPUCKPT_MOCK_COUNTERS");
    if (!dir || !*dir) return;
    char p[4096];
    snprintf(p, sizeof p, "%s/cuda.txt", dir);
    FILE *f = fopen(p, "w");
    if (!f) return;
    fprintf(f, "dtoh_bytes %llu\nhtod_bytes %llu\nasync_copies %llu\nkernel_launches %llu\nstreams %llu\n"
               "host_register %llu\nhost_register_bytes %llu\nhost_alloc_bytes %llu\n",
            c_dtoh, c_htod, c_async, c_launch, c_streams, c_register, c_register_bytes, c_alloc_host);
    fclose(f);
}

/* ---- device allocations (cuMemAlloc), tracked so copies can be checked ---- */
#define MAXALLOC 64
static struct { unsigned char *p; size_t n; } g_alloc[MAXALLOC];
static int g_kernel_fault_pending;

struct op {
    int pid;
    int is_restore;
    unsigned n;
    unsigned char *buf[MAXDEV];
    size_t size[MAXDEV];
    CUcheckpointCustomStoragePerDeviceData pdd[MAXDEV];
    CUcheckpointCustomStorageInfo *info;
    int info_allocated;
    int live;
};

static struct op g_op;   /* one operation in flight per helper process */

/* A helper that exits with an operation still open leaves the target stuck
 * (state file keeps CHECKPOINTING/RESTORING); free our buffers so leak
 * checkers see only real leaks. */
__attribute__((destructor)) static void mock_fini(void)
{
    if (!g_op.live) return;
    for (unsigned i = 0; i < g_op.n; i++) free(g_op.buf[i]);
    if (g_op.info_allocated) free(g_op.info);
    g_op.live = 0;
}

static const char *state_dir(void)
{
    const char *d = getenv("GPUCKPT_MOCK_STATE_DIR");
    return d && *d ? d : "/tmp/gpuckpt-mock";
}

static void mem_path(int pid, unsigned i, char *out, size_t n)
{
    snprintf(out, n, "%s/%d.mem.%u", state_dir(), pid, i);
}

static int split_images(char paths[MAXDEV][4096])
{
    const char *s = getenv("GPUCKPT_MOCK_IMAGES");
    if (!s || !*s) return 0;
    int n = 0;
    char *dup = strdup(s), *save = NULL;
    for (char *t = strtok_r(dup, ",", &save); t && n < MAXDEV; t = strtok_r(NULL, ",", &save))
        snprintf(paths[n++], 4096, "%s", t);
    free(dup);
    return n;
}

static int copy_file(const char *from, const char *to)
{
    FILE *a = fopen(from, "rb"), *b = fopen(to, "wb");
    if (!a || !b) { if (a) fclose(a); if (b) fclose(b); return -1; }
    char buf[1 << 16];
    size_t r;
    while ((r = fread(buf, 1, sizeof buf, a)) > 0) fwrite(buf, 1, r, b);
    fclose(a); fclose(b);
    return 0;
}

/* state file: "state N\nndev N\nsize i N\n..." */
struct pstate { int state; unsigned ndev; size_t size[MAXDEV]; };

static int load_state(int pid, struct pstate *ps)
{
    char p[4096];
    snprintf(p, sizeof p, "%s/%d.state", state_dir(), pid);
    FILE *f = fopen(p, "r");
    memset(ps, 0, sizeof *ps);
    if (!f) {
        /* first contact: seed memory from GPUCKPT_MOCK_IMAGES */
        char paths[MAXDEV][4096];
        int n = split_images(paths);
        if (n == 0) return -1;
        mkdir(state_dir(), 0755);
        for (int i = 0; i < n; i++) {
            char mp[4096]; mem_path(pid, (unsigned)i, mp, sizeof mp);
            if (copy_file(paths[i], mp) != 0) return -1;
            struct stat st; stat(mp, &st);
            ps->size[i] = (size_t)st.st_size;
        }
        ps->ndev = (unsigned)n;
        ps->state = CU_PROCESS_STATE_RUNNING;
        return 0;
    }
    char key[32]; unsigned long long v; unsigned i;
    while (fscanf(f, "%31s", key) == 1) {
        if (!strcmp(key, "state") && fscanf(f, "%llu", &v) == 1) ps->state = (int)v;
        else if (!strcmp(key, "ndev") && fscanf(f, "%llu", &v) == 1) ps->ndev = (unsigned)v;
        else if (!strcmp(key, "size") && fscanf(f, "%u %llu", &i, &v) == 2 && i < MAXDEV) ps->size[i] = (size_t)v;
    }
    fclose(f);
    return 0;
}

static void save_state(int pid, const struct pstate *ps)
{
    char p[4096];
    mkdir(state_dir(), 0755);
    snprintf(p, sizeof p, "%s/%d.state", state_dir(), pid);
    FILE *f = fopen(p, "w");
    if (!f) return;
    fprintf(f, "state %d\nndev %u\n", ps->state, ps->ndev);
    for (unsigned i = 0; i < ps->ndev; i++) fprintf(f, "size %u %zu\n", i, ps->size[i]);
    fclose(f);
}

/* ------------------------------------------------------------ core API */

CUresult cuInit(unsigned flags) { (void)flags; return CUDA_SUCCESS; }

CUresult cuGetErrorString(CUresult r, const char **s)
{
    switch (r) {
    case CUDA_SUCCESS: *s = "no error"; break;
    case CUDA_ERROR_INVALID_VALUE: *s = "invalid argument"; break;
    case CUDA_ERROR_ILLEGAL_STATE: *s = "operation not permitted when stream is capturing / illegal state"; break;
    case CUDA_ERROR_TIMEOUT: *s = "timeout"; break;
    case CUDA_ERROR_INVALID_HANDLE: *s = "invalid resource handle"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *s = "operation not supported"; break;
    default: *s = "unknown error"; break;
    }
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *n)
{
    const char *e = getenv("GPUCKPT_MOCK_DEVICES");
    *n = e ? atoi(e) : 2;
    return CUDA_SUCCESS;
}
CUresult cuDeviceGet(CUdevice *d, int ordinal) { *d = ordinal; return CUDA_SUCCESS; }
CUresult cuDeviceGetUuid(CUuuid *u, CUdevice d)
{
    memset(u->bytes, 0, 16);
    u->bytes[0] = (char)0xa0; u->bytes[15] = (char)d;
    return CUDA_SUCCESS;
}
CUresult cuDeviceGetUuid_v2(CUuuid *u, CUdevice d) { return cuDeviceGetUuid(u, d); }
CUresult cuDevicePrimaryCtxRetain(CUcontext *c, CUdevice d) { *c = (CUcontext)(uintptr_t)(d + 1); return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxRelease_v2(CUdevice d) { (void)d; return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext c) { (void)c; return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream s) { (void)s; return CUDA_SUCCESS; }
CUresult cuMemHostAlloc(void **p, size_t n, unsigned flags) { (void)flags; *p = malloc(n); CADD(c_alloc_host, n); return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
CUresult cuMemHostRegister_v2(void *p, size_t n, unsigned flags)
{
    (void)flags;
    if (getenv("GPUCKPT_MOCK_REGISTER_FAIL")) return CUDA_ERROR_NOT_SUPPORTED;
    if (!p || !n) return CUDA_ERROR_INVALID_VALUE;
    CADD(c_register, 1); CADD(c_register_bytes, n);
    return CUDA_SUCCESS;
}
CUresult cuMemHostUnregister(void *p) { return p ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE; }
CUresult cuDeviceGetAttribute(int *v, CUdevice_attribute a, CUdevice d)
{
    (void)d;
    if (a == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) { *v = 9; return CUDA_SUCCESS; }
    if (a == CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) { *v = 0; return CUDA_SUCCESS; }
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuStreamCreate(CUstream *s, unsigned flags)
{
    (void)flags;
    static long n;
    *s = (CUstream)(__atomic_add_fetch(&n, 1, __ATOMIC_RELAXED));
    CADD(c_streams, 1);
    return CUDA_SUCCESS;
}
CUresult cuStreamDestroy_v2(CUstream s) { (void)s; return CUDA_SUCCESS; }
CUresult cuMemAlloc_v2(CUdeviceptr *p, size_t n)
{
    for (int i = 0; i < MAXALLOC; i++)
        if (!g_alloc[i].p) { g_alloc[i].p = calloc(1, n ? n : 1); g_alloc[i].n = n; *p = (CUdeviceptr)(uintptr_t)g_alloc[i].p; return CUDA_SUCCESS; }
    return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult cuMemFree_v2(CUdeviceptr p)
{
    for (int i = 0; i < MAXALLOC; i++)
        if ((uintptr_t)g_alloc[i].p == (uintptr_t)p) { free(g_alloc[i].p); g_alloc[i].p = NULL; return CUDA_SUCCESS; }
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuModuleLoadData(CUmodule *m, const void *image)
{
    /* accept only what the mock NVRTC produced for this mock device, or a
     * PTX file tagged for the mock */
    const char *s = image;
    if (!strncmp(s, "// MOCKPTX arch=compute_90\n", 27) || !strncmp(s, "// MOCKPTX-FILE\n", 16)) {
        *m = (CUmodule)(uintptr_t)0x4d4f44;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_IMAGE;
}
CUresult cuModuleGetFunction(CUfunction *f, CUmodule m, const char *name)
{
    (void)m;
    if (strcmp(name, "gc_sha256_chunks")) return CUDA_ERROR_NOT_FOUND;
    *f = (CUfunction)(uintptr_t)0x46554e;
    return CUDA_SUCCESS;
}
CUresult cuModuleUnload(CUmodule m) { (void)m; return CUDA_SUCCESS; }
CUresult cuCtxSynchronize(void)
{
    if (g_kernel_fault_pending) { g_kernel_fault_pending = 0; return CUDA_ERROR_ILLEGAL_ADDRESS; }
    return CUDA_SUCCESS;
}
CUresult cuMemFreeHost(void *p) { free(p); return CUDA_SUCCESS; }

static int find_dev(CUdeviceptr p, size_t len)
{
    if (!g_op.live) return -1;
    for (unsigned i = 0; i < g_op.n; i++) {
        uintptr_t b = (uintptr_t)g_op.buf[i];
        if ((uintptr_t)p >= b && (uintptr_t)p + len <= b + g_op.size[i]) return (int)i;
    }
    return -1;
}

static int find_alloc(CUdeviceptr p, size_t len)
{
    for (int i = 0; i < MAXALLOC; i++) {
        uintptr_t b = (uintptr_t)g_alloc[i].p;
        if (b && (uintptr_t)p >= b && (uintptr_t)p + len <= b + g_alloc[i].n) return i;
    }
    return -1;
}

CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                        unsigned shmem, CUstream s, void **params, void **extra)
{
    (void)f; (void)shmem; (void)s; (void)extra;
    if (gy != 1 || gz != 1 || by != 1 || bz != 1 || !params) return CUDA_ERROR_INVALID_VALUE;
    CADD(c_launch, 1);
    CUdeviceptr base = *(CUdeviceptr *)params[0];
    uint64_t size = *(uint64_t *)params[1], cs = *(uint64_t *)params[2], n = *(uint64_t *)params[3];
    CUdeviceptr out = *(CUdeviceptr *)params[4];
    if ((uint64_t)gx * bx < n) return CUDA_ERROR_INVALID_VALUE;           /* grid too small */
    if (getenv("GPUCKPT_MOCK_KERNEL_FAULT") || find_dev(base, size) < 0 || find_alloc(out, n * 32) < 0) {
        g_kernel_fault_pending = 1;                                       /* reported at sync, like a real fault */
        return CUDA_SUCCESS;
    }
    for (uint64_t t = 0; t < (uint64_t)gx * bx; t++)
        gc_sha256_chunk_at(t, (const gc_u8 *)(uintptr_t)base, size, cs, n, (gc_u8 *)(uintptr_t)out);
    if (getenv("GPUCKPT_MOCK_KERNEL_CORRUPT")) ((unsigned char *)(uintptr_t)out)[0] ^= 0xff;
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n)
{
    if (find_dev(src, n) < 0 && find_alloc(src, n) < 0) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dst, (void *)(uintptr_t)src, n);
    CADD(c_dtoh, n);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n)
{
    if (find_dev(dst, n) < 0 && find_alloc(dst, n) < 0) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void *)(uintptr_t)dst, src, n);
    CADD(c_htod, n);
    return CUDA_SUCCESS;
}
/* Async copies complete immediately in the mock; ordering is the caller's
 * job (it must still wait before reusing the buffer, which the real driver
 * enforces and the test suite cannot). */
CUresult cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t n, CUstream s)
{
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    CADD(c_async, 1);
    return cuMemcpyDtoH_v2(dst, src, n);
}
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t n, CUstream s)
{
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    CADD(c_async, 1);
    return cuMemcpyHtoD_v2(dst, src, n);
}
CUresult cuPointerGetAttribute(void *out, CUpointer_attribute a, CUdeviceptr p)
{
    if (a != CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL) return CUDA_ERROR_NOT_SUPPORTED;
    int d = find_dev(p, 1);
    if (d < 0) return CUDA_ERROR_INVALID_VALUE;
    *(int *)out = d;
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------ checkpoint API */

CUresult cuCheckpointProcessGetState(int pid, CUprocessState *s)
{
    struct pstate ps;
    if (load_state(pid, &ps) != 0) return CUDA_ERROR_INVALID_VALUE;
    *s = (CUprocessState)ps.state;
    return CUDA_SUCCESS;
}

CUresult cuCheckpointProcessLock(int pid, CUcheckpointLockArgs *a)
{
    (void)a;
    struct pstate ps;
    if (load_state(pid, &ps) != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ps.state != CU_PROCESS_STATE_RUNNING) return CUDA_ERROR_ILLEGAL_STATE;
    if (getenv("GPUCKPT_MOCK_LOCK_TIMEOUT")) { save_state(pid, &ps); return CUDA_ERROR_TIMEOUT; }
    ps.state = CU_PROCESS_STATE_LOCKED;
    save_state(pid, &ps);
    return CUDA_SUCCESS;
}

CUresult cuCheckpointProcessUnlock(int pid, CUcheckpointUnlockArgs *a)
{
    (void)a;
    struct pstate ps;
    if (load_state(pid, &ps) != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ps.state != CU_PROCESS_STATE_LOCKED) return CUDA_ERROR_ILLEGAL_STATE;
    ps.state = CU_PROCESS_STATE_RUNNING;
    save_state(pid, &ps);
    return CUDA_SUCCESS;
}

CUresult cuCheckpointProcessGetRestoreThreadId(int pid, int *tid) { *tid = pid + 1; return CUDA_SUCCESS; }

static CUcheckpointCustomStorageInfo *publish_info(CUcheckpointCustomStorageInfo **out)
{
    CUcheckpointCustomStorageInfo *info;
    if (getenv("GPUCKPT_MOCK_ALLOC_INFO") || !*out) {
        info = calloc(1, sizeof *info);
        g_op.info_allocated = 1;
        *out = info;
    } else {
        info = *out;
    }
    info->handle = (CUcheckpointOperationHandle)&g_op;
    info->perDeviceData = g_op.pdd;
    info->deviceCount = g_op.n;
    g_op.info = info;
    return info;
}

CUresult cuCheckpointProcessCheckpoint(int pid, CUcheckpointCheckpointArgs *a)
{
    struct pstate ps;
    if (load_state(pid, &ps) != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ps.state != CU_PROCESS_STATE_LOCKED) return CUDA_ERROR_ILLEGAL_STATE;
    if (g_op.live) return CUDA_ERROR_ILLEGAL_STATE;
    if (!a->customStorageInfo_out) {
        /* host-storage path: not modelled beyond the state change */
        ps.state = CU_PROCESS_STATE_CHECKPOINTED;
        save_state(pid, &ps);
        return CUDA_SUCCESS;
    }
    memset(&g_op, 0, sizeof g_op);
    g_op.pid = pid;
    g_op.n = ps.ndev;
    for (unsigned i = 0; i < ps.ndev; i++) {
        char mp[4096]; mem_path(pid, i, mp, sizeof mp);
        FILE *f = fopen(mp, "rb");
        if (!f) return CUDA_ERROR_UNKNOWN;
        g_op.size[i] = ps.size[i];
        g_op.buf[i] = malloc(ps.size[i] ? ps.size[i] : 1);
        size_t got = fread(g_op.buf[i], 1, ps.size[i], f);
        fclose(f);
        if (got != ps.size[i]) return CUDA_ERROR_UNKNOWN;
        g_op.pdd[i].devPtr = (CUdeviceptr)(uintptr_t)g_op.buf[i];
        g_op.pdd[i].size = ps.size[i];
        g_op.pdd[i].stream = NULL;
    }
    g_op.live = 1;
    publish_info(a->customStorageInfo_out);
    ps.state = CU_PROCESS_STATE_CHECKPOINTING;
    save_state(pid, &ps);
    return CUDA_SUCCESS;
}

CUresult cuCheckpointProcessRestore(int pid, CUcheckpointRestoreArgs *a)
{
    struct pstate ps;
    if (load_state(pid, &ps) != 0) return CUDA_ERROR_INVALID_VALUE;
    if (ps.state != CU_PROCESS_STATE_CHECKPOINTED) return CUDA_ERROR_ILLEGAL_STATE;
    if (g_op.live) return CUDA_ERROR_ILLEGAL_STATE;
    if (!a->customStorageInfo_out) {
        ps.state = CU_PROCESS_STATE_LOCKED;
        save_state(pid, &ps);
        return CUDA_SUCCESS;
    }
    memset(&g_op, 0, sizeof g_op);
    g_op.pid = pid;
    g_op.is_restore = 1;
    g_op.n = ps.ndev;
    for (unsigned i = 0; i < ps.ndev; i++) {
        g_op.size[i] = ps.size[i];
        g_op.buf[i] = calloc(1, ps.size[i] ? ps.size[i] : 1);  /* fresh GPU memory: zeros */
        g_op.pdd[i].devPtr = (CUdeviceptr)(uintptr_t)g_op.buf[i];
        g_op.pdd[i].size = ps.size[i];
        g_op.pdd[i].stream = NULL;
    }
    g_op.live = 1;
    publish_info(a->customStorageInfo_out);
    ps.state = CU_PROCESS_STATE_RESTORING;
    save_state(pid, &ps);
    return CUDA_SUCCESS;
}

CUresult cuCheckpointOperationComplete(CUcheckpointOperationHandle h)
{
    if ((void *)h != (void *)&g_op || !g_op.live) return CUDA_ERROR_INVALID_HANDLE;
    if (getenv("GPUCKPT_MOCK_FAIL_COMPLETE")) return CUDA_ERROR_UNKNOWN;
    struct pstate ps;
    if (load_state(g_op.pid, &ps) != 0) return CUDA_ERROR_UNKNOWN;
    for (unsigned i = 0; i < g_op.n; i++) {
        char mp[4096]; mem_path(g_op.pid, i, mp, sizeof mp);
        if (g_op.is_restore) {
            FILE *f = fopen(mp, "wb");
            if (!f) return CUDA_ERROR_UNKNOWN;
            fwrite(g_op.buf[i], 1, g_op.size[i], f);
            fclose(f);
        } else {
            unlink(mp);   /* GPU memory released */
        }
        free(g_op.buf[i]);
        g_op.buf[i] = NULL;
    }
    ps.state = g_op.is_restore ? CU_PROCESS_STATE_LOCKED : CU_PROCESS_STATE_CHECKPOINTED;
    save_state(g_op.pid, &ps);
    if (g_op.info_allocated) free(g_op.info);
    memset(&g_op, 0, sizeof g_op);
    return CUDA_SUCCESS;
}
