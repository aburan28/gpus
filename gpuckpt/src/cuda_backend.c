/* CUDA driver checkpoint adapter.
 *
 * Runtime binding: libcuda is dlopen'ed and every symbol is resolved by
 * name, so (a) the binary runs on hosts without a driver for file-backed
 * work, (b) a missing checkpoint symbol is a clear, reportable error rather
 * than a load failure, and (c) the test suite can substitute a mock driver
 * through GPUCKPT_LIBCUDA. Compile-time binding: struct layouts come from
 * <cuda.h>; see docs/design.md "Build modes" for the real/mock distinction.
 *
 * Custom-storage protocol as documented (CUDA 13.4 reference):
 *   lock(pid)                         RUNNING  -> LOCKED
 *   checkpoint(pid, custom)           LOCKED   -> CHECKPOINTING, GPU memory mapped into us
 *   ... copy out ...
 *   complete(handle)                  CHECKPOINTING -> CHECKPOINTED, GPU memory released
 *   restore(pid, custom)              CHECKPOINTED -> RESTORING, fresh GPU memory mapped into us
 *   ... copy in ...
 *   complete(handle)                  RESTORING -> LOCKED
 *   unlock(pid)                       LOCKED   -> RUNNING
 *
 * Custom storage requires cuInit in this process and retained primary
 * contexts for every device the target uses; we retain all of them.
 * Checkpointing or restoring our own process this way is unsupported.
 */
#include "internal.h"
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>

typedef CUresult (*fn_cuInit)(unsigned);
typedef CUresult (*fn_cuGetErrorString)(CUresult, const char **);
typedef CUresult (*fn_cuDeviceGetCount)(int *);
typedef CUresult (*fn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*fn_cuDeviceGetUuid)(CUuuid *, CUdevice);
typedef CUresult (*fn_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
typedef CUresult (*fn_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*fn_cuMemHostAlloc)(void **, size_t, unsigned);
typedef CUresult (*fn_cuMemFreeHost)(void *);
typedef CUresult (*fn_cuPointerGetAttribute)(void *, CUpointer_attribute, CUdeviceptr);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuCheckpointProcessGetState)(int, CUprocessState *);
typedef CUresult (*fn_cuCheckpointProcessLock)(int, CUcheckpointLockArgs *);
typedef CUresult (*fn_cuCheckpointProcessUnlock)(int, CUcheckpointUnlockArgs *);
typedef CUresult (*fn_cuCheckpointProcessCheckpoint)(int, CUcheckpointCheckpointArgs *);
typedef CUresult (*fn_cuCheckpointProcessRestore)(int, CUcheckpointRestoreArgs *);
typedef CUresult (*fn_cuCheckpointOperationComplete)(CUcheckpointOperationHandle);
typedef CUresult (*fn_cuCheckpointProcessGetRestoreThreadId)(int, int *);

struct gc_cuda {
    void *lib;
    char  libpath[PATH_MAX];
    fn_cuInit cuInit;
    fn_cuGetErrorString cuGetErrorString;
    fn_cuDeviceGetCount cuDeviceGetCount;
    fn_cuDeviceGet cuDeviceGet;
    fn_cuDeviceGetUuid cuDeviceGetUuid;
    fn_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain;
    fn_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease;
    fn_cuCtxSetCurrent cuCtxSetCurrent;
    fn_cuMemcpyDtoH cuMemcpyDtoH;
    fn_cuMemcpyHtoD cuMemcpyHtoD;
    fn_cuMemHostAlloc cuMemHostAlloc;
    fn_cuMemFreeHost cuMemFreeHost;
    fn_cuPointerGetAttribute cuPointerGetAttribute;
    fn_cuStreamSynchronize cuStreamSynchronize;
    fn_cuCheckpointProcessGetState cuCheckpointProcessGetState;
    fn_cuCheckpointProcessLock cuCheckpointProcessLock;
    fn_cuCheckpointProcessUnlock cuCheckpointProcessUnlock;
    fn_cuCheckpointProcessCheckpoint cuCheckpointProcessCheckpoint;
    fn_cuCheckpointProcessRestore cuCheckpointProcessRestore;
    fn_cuCheckpointOperationComplete cuCheckpointOperationComplete;
    fn_cuCheckpointProcessGetRestoreThreadId cuCheckpointProcessGetRestoreThreadId;
    int       ndev;
    CUdevice  devs[GC_MAX_DEVICES];
    CUcontext ctx[GC_MAX_DEVICES];
    int       retained[GC_MAX_DEVICES];
};

typedef struct {
    gc_cuda *c;
    int is_restore;
    int completed;
    CUcheckpointOperationHandle handle;
    CUcheckpointCustomStorageInfo  info_storage;  /* zeroed; driver may fill it ... */
    CUcheckpointCustomStorageInfo *info;          /* ... or replace this pointer */
    CUdeviceptr ptr[GC_MAX_DEVICES];
    CUstream    stream[GC_MAX_DEVICES];
} cuda_priv;

const char *gc_cuda_header_mode(void)
{
#if GPUCKPT_CUDA_HEADER_REAL
    return "real";
#else
    return "mock (UNVERIFIED)";
#endif
}

const char *gc_proc_state_name(int s)
{
    switch (s) {
    case GC_PS_RUNNING: return "RUNNING";
    case GC_PS_LOCKED: return "LOCKED";
    case GC_PS_CHECKPOINTING: return "CHECKPOINTING";
    case GC_PS_CHECKPOINTED: return "CHECKPOINTED";
    case GC_PS_RESTORING: return "RESTORING";
    case GC_PS_FAILED: return "FAILED";
    default: return "UNKNOWN";
    }
}

static int cu_fail(gc_cuda *c, const char *what, CUresult r)
{
    const char *s = NULL;
    if (c->cuGetErrorString && c->cuGetErrorString(r, &s) != CUDA_SUCCESS) s = NULL;
    gc_set_error("%s: CUresult %d (%s)", what, (int)r, s ? s : "no description");
    return GC_EBACKEND;
}

static void *sym(gc_cuda *c, const char *name, const char *alt, int required, int *ok)
{
    void *p = dlsym(c->lib, name);
    if (!p && alt) p = dlsym(c->lib, alt);
    if (!p && required) {
        gc_set_error("%s: symbol %s not found (driver too old for the checkpoint API?)", c->libpath, name);
        *ok = 0;
    }
    return p;
}

int gc_cuda_open(gc_cuda **out, const char *libpath)
{
    gc_cuda *c = calloc(1, sizeof *c);
    if (!c) return GC_ENOMEM;
    if (!libpath) libpath = getenv("GPUCKPT_LIBCUDA");
    if (!libpath || !*libpath) libpath = "libcuda.so.1";
    snprintf(c->libpath, sizeof c->libpath, "%s", libpath);
    c->lib = dlopen(libpath, RTLD_NOW | RTLD_LOCAL);
    if (!c->lib) { gc_set_error("dlopen %s: %s", libpath, dlerror()); free(c); return GC_EBACKEND; }
    int ok = 1;
    c->cuInit = sym(c, "cuInit", NULL, 1, &ok);
    c->cuGetErrorString = sym(c, "cuGetErrorString", NULL, 1, &ok);
    c->cuDeviceGetCount = sym(c, "cuDeviceGetCount", NULL, 1, &ok);
    c->cuDeviceGet = sym(c, "cuDeviceGet", NULL, 1, &ok);
    c->cuDeviceGetUuid = sym(c, "cuDeviceGetUuid_v2", "cuDeviceGetUuid", 1, &ok);
    c->cuDevicePrimaryCtxRetain = sym(c, "cuDevicePrimaryCtxRetain", NULL, 1, &ok);
    c->cuDevicePrimaryCtxRelease = sym(c, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease", 1, &ok);
    c->cuCtxSetCurrent = sym(c, "cuCtxSetCurrent", NULL, 1, &ok);
    c->cuMemcpyDtoH = sym(c, "cuMemcpyDtoH_v2", NULL, 1, &ok);
    c->cuMemcpyHtoD = sym(c, "cuMemcpyHtoD_v2", NULL, 1, &ok);
    c->cuMemHostAlloc = sym(c, "cuMemHostAlloc", NULL, 0, &ok);
    c->cuMemFreeHost = sym(c, "cuMemFreeHost", NULL, 0, &ok);
    c->cuPointerGetAttribute = sym(c, "cuPointerGetAttribute", NULL, 0, &ok);
    c->cuStreamSynchronize = sym(c, "cuStreamSynchronize", NULL, 1, &ok);
    c->cuCheckpointProcessGetState = sym(c, "cuCheckpointProcessGetState", NULL, 1, &ok);
    c->cuCheckpointProcessLock = sym(c, "cuCheckpointProcessLock", NULL, 1, &ok);
    c->cuCheckpointProcessUnlock = sym(c, "cuCheckpointProcessUnlock", NULL, 1, &ok);
    c->cuCheckpointProcessCheckpoint = sym(c, "cuCheckpointProcessCheckpoint", NULL, 1, &ok);
    c->cuCheckpointProcessRestore = sym(c, "cuCheckpointProcessRestore", NULL, 1, &ok);
    c->cuCheckpointOperationComplete = sym(c, "cuCheckpointOperationComplete", NULL, 1, &ok);
    c->cuCheckpointProcessGetRestoreThreadId = sym(c, "cuCheckpointProcessGetRestoreThreadId", NULL, 1, &ok);
    if (!ok) { dlclose(c->lib); free(c); return GC_EBACKEND; }

    CUresult r = c->cuInit(0);
    if (r != CUDA_SUCCESS) { int rc = cu_fail(c, "cuInit", r); dlclose(c->lib); free(c); return rc; }
    r = c->cuDeviceGetCount(&c->ndev);
    if (r != CUDA_SUCCESS) { int rc = cu_fail(c, "cuDeviceGetCount", r); dlclose(c->lib); free(c); return rc; }
    if (c->ndev > GC_MAX_DEVICES) c->ndev = GC_MAX_DEVICES;
    for (int i = 0; i < c->ndev; i++) {
        r = c->cuDeviceGet(&c->devs[i], i);
        if (r != CUDA_SUCCESS) { int rc = cu_fail(c, "cuDeviceGet", r); gc_cuda_close(c); return rc; }
        r = c->cuDevicePrimaryCtxRetain(&c->ctx[i], c->devs[i]);
        if (r != CUDA_SUCCESS) { int rc = cu_fail(c, "cuDevicePrimaryCtxRetain", r); gc_cuda_close(c); return rc; }
        c->retained[i] = 1;
    }
    *out = c;
    return GC_OK;
}

void gc_cuda_close(gc_cuda *c)
{
    if (!c) return;
    for (int i = 0; i < c->ndev; i++)
        if (c->retained[i]) c->cuDevicePrimaryCtxRelease(c->devs[i]);
    if (c->lib) dlclose(c->lib);
    free(c);
}

static int map_state(CUprocessState s)
{
    switch (s) {
    case CU_PROCESS_STATE_RUNNING: return GC_PS_RUNNING;
    case CU_PROCESS_STATE_LOCKED: return GC_PS_LOCKED;
    case CU_PROCESS_STATE_CHECKPOINTING: return GC_PS_CHECKPOINTING;
    case CU_PROCESS_STATE_CHECKPOINTED: return GC_PS_CHECKPOINTED;
    case CU_PROCESS_STATE_RESTORING: return GC_PS_RESTORING;
    case CU_PROCESS_STATE_FAILED: return GC_PS_FAILED;
    default: return GC_PS_UNKNOWN;
    }
}

int gc_cuda_get_state(gc_cuda *c, int pid, int *state)
{
    CUprocessState s;
    CUresult r = c->cuCheckpointProcessGetState(pid, &s);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuCheckpointProcessGetState", r);
    *state = map_state(s);
    return GC_OK;
}

int gc_cuda_lock(gc_cuda *c, int pid, unsigned timeout_ms)
{
    CUcheckpointLockArgs a;
    memset(&a, 0, sizeof a);
    a.timeoutMs = timeout_ms;
    CUresult r = c->cuCheckpointProcessLock(pid, &a);
    if (r != CUDA_SUCCESS) {
        int st = GC_PS_UNKNOWN;
        gc_cuda_get_state(c, pid, &st);
        cu_fail(c, "cuCheckpointProcessLock", r);
        if (st == GC_PS_RUNNING && timeout_ms)
            gc_set_error("cuCheckpointProcessLock: not locked within %u ms (target still RUNNING); CUresult %d", timeout_ms, (int)r);
        return GC_EBACKEND;
    }
    return GC_OK;
}

int gc_cuda_unlock(gc_cuda *c, int pid)
{
    CUcheckpointUnlockArgs a;
    memset(&a, 0, sizeof a);
    CUresult r = c->cuCheckpointProcessUnlock(pid, &a);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuCheckpointProcessUnlock", r);
    return GC_OK;
}

int gc_cuda_restore_thread_id(gc_cuda *c, int pid, int *tid)
{
    CUresult r = c->cuCheckpointProcessGetRestoreThreadId(pid, tid);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuCheckpointProcessGetRestoreThreadId", r);
    return GC_OK;
}

/* ------------------------------------------------------------ image ops */

static int cu_attach(gc_image *img)
{
    cuda_priv *p = img->priv;
    if (p->c->ndev == 0) return GC_OK;
    CUresult r = p->c->cuCtxSetCurrent(p->c->ctx[0]);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuCtxSetCurrent", r);
    return GC_OK;
}

static int cu_read(gc_image *img, uint32_t d, uint64_t off, void *dst, size_t len)
{
    cuda_priv *p = img->priv;
    if (p->completed) { gc_set_error("mapping already released by cuCheckpointOperationComplete"); return GC_ESTATE; }
    if (off + len > img->dev[d].size) return GC_EINVAL;
    if (img->dev[d].ordinal >= 0 && img->dev[d].ordinal < p->c->ndev)
        p->c->cuCtxSetCurrent(p->c->ctx[img->dev[d].ordinal]);
    CUresult r = p->c->cuMemcpyDtoH(dst, p->ptr[d] + off, len);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyDtoH", r);
    return GC_OK;
}

static int cu_write(gc_image *img, uint32_t d, uint64_t off, const void *src, size_t len)
{
    cuda_priv *p = img->priv;
    if (p->completed) { gc_set_error("mapping already released by cuCheckpointOperationComplete"); return GC_ESTATE; }
    if (off + len > img->dev[d].size) return GC_EINVAL;
    if (img->dev[d].ordinal >= 0 && img->dev[d].ordinal < p->c->ndev)
        p->c->cuCtxSetCurrent(p->c->ctx[img->dev[d].ordinal]);
    CUresult r = p->c->cuMemcpyHtoD(p->ptr[d] + off, src, len);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyHtoD", r);
    return GC_OK;
}

static void *cu_buf_alloc(gc_image *img, size_t len)
{
    cuda_priv *p = img->priv;
    void *b = NULL;
    if (p->c->cuMemHostAlloc && p->c->cuMemHostAlloc(&b, len, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS && b)
        return b;
    return malloc(len);
}

static void cu_buf_free(gc_image *img, void *b, size_t len)
{
    (void)len;
    cuda_priv *p = img->priv;
    if (p->c->cuMemFreeHost && p->c->cuMemFreeHost(b) == CUDA_SUCCESS) return;
    free(b);
}

static void fill_uuid(gc_cuda *c, int ordinal, char *out, size_t n)
{
    CUuuid u;
    if (ordinal < 0 || ordinal >= c->ndev || c->cuDeviceGetUuid(&u, c->devs[ordinal]) != CUDA_SUCCESS) {
        snprintf(out, n, "-");
        return;
    }
    const unsigned char *b = (const unsigned char *)u.bytes;
    snprintf(out, n, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
}

/* Shared tail of checkpoint_begin / restore_begin: read the per-device
 * table the driver handed back and describe it as a gc_image. */
static int adopt_info(gc_cuda *c, cuda_priv *p, gc_image *img, const char *what)
{
    CUcheckpointCustomStorageInfo *info = p->info;
    if (!info) { gc_set_error("%s: driver returned no custom storage info", what); return GC_EBACKEND; }
    p->handle = info->handle;
    if (info->deviceCount == 0 || !info->perDeviceData) {
        gc_set_error("%s: custom storage info has deviceCount=%u perDeviceData=%p", what,
                     info->deviceCount, (void *)info->perDeviceData);
        return GC_EBACKEND;
    }
    if (info->deviceCount > GC_MAX_DEVICES) {
        gc_set_error("%s: %u devices exceeds GC_MAX_DEVICES=%d", what, info->deviceCount, GC_MAX_DEVICES);
        return GC_EBACKEND;
    }
    img->device_count = info->deviceCount;
    for (unsigned i = 0; i < info->deviceCount; i++) {
        const CUcheckpointCustomStoragePerDeviceData *e = &info->perDeviceData[i];
        p->ptr[i] = e->devPtr;
        p->stream[i] = e->stream;
        img->dev[i].size = e->size;
        img->dev[i].ordinal = -1;
        if (c->cuPointerGetAttribute) {
            int ord = -1;
            if (c->cuPointerGetAttribute(&ord, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, e->devPtr) == CUDA_SUCCESS)
                img->dev[i].ordinal = ord;
        }
        fill_uuid(c, img->dev[i].ordinal, img->dev[i].uuid, sizeof img->dev[i].uuid);
    }
    snprintf(img->backend, sizeof img->backend, "cuda");
    img->thread_attach = cu_attach;
    img->buf_alloc = cu_buf_alloc;
    img->buf_free = cu_buf_free;
    img->priv = p;
    return GC_OK;
}

int gc_cuda_checkpoint_begin(gc_cuda *c, int pid, gc_image *img)
{
    memset(img, 0, sizeof *img);
    cuda_priv *p = calloc(1, sizeof *p);
    if (!p) return GC_ENOMEM;
    p->c = c;
    p->info = &p->info_storage;              /* zeroed: handle 0, perDeviceData NULL, deviceCount 0 */
    CUcheckpointCheckpointArgs a;
    memset(&a, 0, sizeof a);
    a.customStorageInfo_out = &p->info;
    CUresult r = c->cuCheckpointProcessCheckpoint(pid, &a);
    if (r != CUDA_SUCCESS) { free(p); return cu_fail(c, "cuCheckpointProcessCheckpoint", r); }
    int rc = adopt_info(c, p, img, "cuCheckpointProcessCheckpoint");
    if (rc) { free(p); img->priv = NULL; return rc; }
    img->read = cu_read;
    return GC_OK;
}

int gc_cuda_restore_begin(gc_cuda *c, int pid, gc_image *img)
{
    memset(img, 0, sizeof *img);
    cuda_priv *p = calloc(1, sizeof *p);
    if (!p) return GC_ENOMEM;
    p->c = c;
    p->is_restore = 1;
    p->info = &p->info_storage;
    CUcheckpointRestoreArgs a;
    memset(&a, 0, sizeof a);
    a.customStorageInfo_out = &p->info;
    CUresult r = c->cuCheckpointProcessRestore(pid, &a);
    if (r != CUDA_SUCCESS) { free(p); return cu_fail(c, "cuCheckpointProcessRestore", r); }
    int rc = adopt_info(c, p, img, "cuCheckpointProcessRestore");
    if (rc) { free(p); img->priv = NULL; return rc; }
    img->write = cu_write;
    return GC_OK;
}

int gc_cuda_op_complete(gc_cuda *c, gc_image *img)
{
    cuda_priv *p = img->priv;
    if (!p) return GC_EINVAL;
    if (p->completed) return GC_OK;
    /* our copies were synchronous, but honour the documented contract anyway */
    for (uint32_t i = 0; i < img->device_count; i++)
        if (p->stream[i]) c->cuStreamSynchronize(p->stream[i]);
    CUresult r = c->cuCheckpointOperationComplete(p->handle);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuCheckpointOperationComplete", r);
    p->completed = 1;
    return GC_OK;
}

void gc_cuda_image_close(gc_cuda *c, gc_image *img)
{
    (void)c;
    free(img->priv);
    img->priv = NULL;
}
