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
 * Fault injection:
 *   GPUCKPT_MOCK_LOCK_TIMEOUT=1      lock returns CUDA_ERROR_TIMEOUT, stays RUNNING
 *   GPUCKPT_MOCK_FAIL_COMPLETE=1     complete returns CUDA_ERROR_UNKNOWN
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
CUresult cuMemHostAlloc(void **p, size_t n, unsigned flags) { (void)flags; *p = malloc(n); return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY; }
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

CUresult cuMemcpyDtoH_v2(void *dst, CUdeviceptr src, size_t n)
{
    if (find_dev(src, n) < 0) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dst, (void *)(uintptr_t)src, n);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyHtoD_v2(CUdeviceptr dst, const void *src, size_t n)
{
    if (find_dev(dst, n) < 0) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void *)(uintptr_t)dst, src, n);
    return CUDA_SUCCESS;
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
