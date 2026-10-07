/* CUDA driver checkpoint adapter.
 *
 * Runtime binding: libcuda, libnvrtc and libcufile are dlopen'ed and every
 * symbol is resolved by name, so (a) the binary runs on hosts without a
 * driver for file-backed work, (b) a missing symbol is a clear, reportable
 * error rather than a load failure, and (c) the test suite can substitute
 * mock libraries through GPUCKPT_LIBCUDA / GPUCKPT_LIBNVRTC /
 * GPUCKPT_LIBCUFILE. Compile-time binding: struct layouts come from
 * <cuda.h> and <cufile.h>; see docs/design.md "Build modes".
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
 *
 * Data paths, fastest last (docs/performance.md):
 *   - synchronous cuMemcpyDtoH into pageable memory (fallback)
 *   - async cuMemcpyDtoHAsync into a page-locked arena, two slots per
 *     worker on separate streams, so copies overlap CPU hashing
 *   - device hashing: a SHA-256 kernel (NVRTC-compiled at runtime) hashes
 *     every chunk in place; only 32 bytes per chunk cross PCIe, and only
 *     chunks the store lacks are copied at all
 *   - GPUDirect Storage: new chunks go from the mapping to their file via
 *     cuFileWrite with no host staging
 */
#include "internal.h"
#include <cuda.h>
#include <cufile.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

extern const char gc_kernel_sha256_src[];

#define MAX_SLOTS 4

typedef CUresult (*fn_cuInit)(unsigned);
typedef CUresult (*fn_cuGetErrorString)(CUresult, const char **);
typedef CUresult (*fn_cuDeviceGetCount)(int *);
typedef CUresult (*fn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*fn_cuDeviceGetUuid)(CUuuid *, CUdevice);
typedef CUresult (*fn_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*fn_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
typedef CUresult (*fn_cuDevicePrimaryCtxRelease)(CUdevice);
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuCtxSynchronize)(void);
typedef CUresult (*fn_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
typedef CUresult (*fn_cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream);
typedef CUresult (*fn_cuMemHostAlloc)(void **, size_t, unsigned);
typedef CUresult (*fn_cuMemFreeHost)(void *);
typedef CUresult (*fn_cuMemHostRegister)(void *, size_t, unsigned);
typedef CUresult (*fn_cuMemHostUnregister)(void *);
typedef CUresult (*fn_cuMemAlloc)(CUdeviceptr *, size_t);
typedef CUresult (*fn_cuMemFree)(CUdeviceptr);
typedef CUresult (*fn_cuPointerGetAttribute)(void *, CUpointer_attribute, CUdeviceptr);
typedef CUresult (*fn_cuStreamCreate)(CUstream *, unsigned);
typedef CUresult (*fn_cuStreamDestroy)(CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuModuleLoadData)(CUmodule *, const void *);
typedef CUresult (*fn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
typedef CUresult (*fn_cuModuleUnload)(CUmodule);
typedef CUresult (*fn_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                      unsigned, CUstream, void **, void **);
typedef CUresult (*fn_cuCheckpointProcessGetState)(int, CUprocessState *);
typedef CUresult (*fn_cuCheckpointProcessLock)(int, CUcheckpointLockArgs *);
typedef CUresult (*fn_cuCheckpointProcessUnlock)(int, CUcheckpointUnlockArgs *);
typedef CUresult (*fn_cuCheckpointProcessCheckpoint)(int, CUcheckpointCheckpointArgs *);
typedef CUresult (*fn_cuCheckpointProcessRestore)(int, CUcheckpointRestoreArgs *);
typedef CUresult (*fn_cuCheckpointOperationComplete)(CUcheckpointOperationHandle);
typedef CUresult (*fn_cuCheckpointProcessGetRestoreThreadId)(int, int *);

/* NVRTC: declared here rather than included, so no toolkit header is
 * needed; nvrtcResult is an int-sized enum with NVRTC_SUCCESS == 0. */
typedef int nvrtcResult_t;
typedef void *nvrtcProgram_t;
typedef nvrtcResult_t (*fn_nvrtcCreateProgram)(nvrtcProgram_t *, const char *, const char *, int,
                                               const char *const *, const char *const *);
typedef nvrtcResult_t (*fn_nvrtcCompileProgram)(nvrtcProgram_t, int, const char *const *);
typedef nvrtcResult_t (*fn_nvrtcGetPTXSize)(nvrtcProgram_t, size_t *);
typedef nvrtcResult_t (*fn_nvrtcGetPTX)(nvrtcProgram_t, char *);
typedef nvrtcResult_t (*fn_nvrtcGetProgramLogSize)(nvrtcProgram_t, size_t *);
typedef nvrtcResult_t (*fn_nvrtcGetProgramLog)(nvrtcProgram_t, char *);
typedef nvrtcResult_t (*fn_nvrtcDestroyProgram)(nvrtcProgram_t *);
typedef const char *(*fn_nvrtcGetErrorString)(nvrtcResult_t);

typedef CUfileError_t (*fn_cuFileDriverOpen)(void);
typedef CUfileError_t (*fn_cuFileDriverClose)(void);
typedef CUfileError_t (*fn_cuFileHandleRegister)(CUfileHandle_t *, CUfileDescr_t *);
typedef void (*fn_cuFileHandleDeregister)(CUfileHandle_t);
typedef CUfileError_t (*fn_cuFileBufRegister)(const void *, size_t, int);
typedef CUfileError_t (*fn_cuFileBufDeregister)(const void *);
typedef ssize_t (*fn_cuFileRead)(CUfileHandle_t, void *, size_t, off_t, off_t);
typedef ssize_t (*fn_cuFileWrite)(CUfileHandle_t, const void *, size_t, off_t, off_t);

struct gc_cuda {
    void *lib;
    char  libpath[PATH_MAX];
    fn_cuInit cuInit;
    fn_cuGetErrorString cuGetErrorString;
    fn_cuDeviceGetCount cuDeviceGetCount;
    fn_cuDeviceGet cuDeviceGet;
    fn_cuDeviceGetUuid cuDeviceGetUuid;
    fn_cuDeviceGetAttribute cuDeviceGetAttribute;
    fn_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain;
    fn_cuDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease;
    fn_cuCtxSetCurrent cuCtxSetCurrent;
    fn_cuCtxSynchronize cuCtxSynchronize;
    fn_cuMemcpyDtoH cuMemcpyDtoH;
    fn_cuMemcpyHtoD cuMemcpyHtoD;
    fn_cuMemcpyDtoHAsync cuMemcpyDtoHAsync;
    fn_cuMemcpyHtoDAsync cuMemcpyHtoDAsync;
    fn_cuMemHostAlloc cuMemHostAlloc;
    fn_cuMemFreeHost cuMemFreeHost;
    fn_cuMemHostRegister cuMemHostRegister;
    fn_cuMemHostUnregister cuMemHostUnregister;
    fn_cuMemAlloc cuMemAlloc;
    fn_cuMemFree cuMemFree;
    fn_cuPointerGetAttribute cuPointerGetAttribute;
    fn_cuStreamCreate cuStreamCreate;
    fn_cuStreamDestroy cuStreamDestroy;
    fn_cuStreamSynchronize cuStreamSynchronize;
    fn_cuModuleLoadData cuModuleLoadData;
    fn_cuModuleGetFunction cuModuleGetFunction;
    fn_cuModuleUnload cuModuleUnload;
    fn_cuLaunchKernel cuLaunchKernel;
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

    /* device hashing */
    int        gpu_hash;
    CUmodule   mod[GC_MAX_DEVICES];
    CUfunction fn_hash[GC_MAX_DEVICES];

    /* GPUDirect Storage */
    int   gds;
    void *cufile_lib;
    fn_cuFileDriverOpen cuFileDriverOpen;
    fn_cuFileDriverClose cuFileDriverClose;
    fn_cuFileHandleRegister cuFileHandleRegister;
    fn_cuFileHandleDeregister cuFileHandleDeregister;
    fn_cuFileBufRegister cuFileBufRegister;
    fn_cuFileBufDeregister cuFileBufDeregister;
    fn_cuFileRead cuFileRead;
    fn_cuFileWrite cuFileWrite;
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
    int         buf_registered[GC_MAX_DEVICES];   /* cuFileBufRegister succeeded */
} cuda_priv;

/* per-worker: streams created lazily per (device ordinal, slot) */
typedef struct {
    CUstream s[GC_MAX_DEVICES][MAX_SLOTS];
    int      slot_ctx[MAX_SLOTS];                 /* context index the slot's last copy used */
} cuda_tctx;

const char *gc_cuda_header_mode(void)
{
#if GPUCKPT_CUDA_HEADER_REAL
    return "real";
#else
    return "mock (UNVERIFIED)";
#endif
}

const char *gc_cufile_header_mode(void);
const char *gc_cufile_header_mode(void)
{
#if GPUCKPT_CUFILE_HEADER_REAL
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

static const char *cu_errstr(gc_cuda *c, CUresult r)
{
    const char *s = NULL;
    if (c->cuGetErrorString && c->cuGetErrorString(r, &s) != CUDA_SUCCESS) s = NULL;
    return s ? s : "no description";
}

static int cu_fail(gc_cuda *c, const char *what, CUresult r)
{
    gc_set_error("%s: CUresult %d (%s)", what, (int)r, cu_errstr(c, r));
    return GC_EBACKEND;
}

static void *sym(void *lib, const char *libpath, const char *name, const char *alt, int required, int *ok)
{
    void *p = dlsym(lib, name);
    if (!p && alt) p = dlsym(lib, alt);
    if (!p && required) {
        gc_set_error("%s: symbol %s not found (driver too old for the checkpoint API?)", libpath, name);
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
    void *L = c->lib;
    const char *P = c->libpath;
#define REQ(f) c->f = (fn_##f)sym(L, P, #f, NULL, 1, &ok)
#define REQ2(f, a, b) c->f = (fn_##f)sym(L, P, a, b, 1, &ok)
#define OPT(f) c->f = (fn_##f)sym(L, P, #f, NULL, 0, &ok)
#define OPT2(f, a, b) c->f = (fn_##f)sym(L, P, a, b, 0, &ok)
    REQ(cuInit); REQ(cuGetErrorString); REQ(cuDeviceGetCount); REQ(cuDeviceGet);
    REQ2(cuDeviceGetUuid, "cuDeviceGetUuid_v2", "cuDeviceGetUuid");
    REQ(cuDevicePrimaryCtxRetain);
    REQ2(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease");
    REQ(cuCtxSetCurrent);
    REQ2(cuMemcpyDtoH, "cuMemcpyDtoH_v2", NULL);
    REQ2(cuMemcpyHtoD, "cuMemcpyHtoD_v2", NULL);
    REQ(cuStreamSynchronize);
    REQ(cuCheckpointProcessGetState); REQ(cuCheckpointProcessLock); REQ(cuCheckpointProcessUnlock);
    REQ(cuCheckpointProcessCheckpoint); REQ(cuCheckpointProcessRestore);
    REQ(cuCheckpointOperationComplete); REQ(cuCheckpointProcessGetRestoreThreadId);
    /* optional: each one only gates an acceleration */
    OPT(cuDeviceGetAttribute); OPT(cuCtxSynchronize);
    OPT2(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", NULL);
    OPT2(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", NULL);
    OPT(cuMemHostAlloc); OPT(cuMemFreeHost);
    OPT2(cuMemHostRegister, "cuMemHostRegister_v2", "cuMemHostRegister");
    OPT(cuMemHostUnregister);
    OPT2(cuMemAlloc, "cuMemAlloc_v2", NULL);
    OPT2(cuMemFree, "cuMemFree_v2", NULL);
    OPT(cuPointerGetAttribute); OPT(cuStreamCreate);
    OPT2(cuStreamDestroy, "cuStreamDestroy_v2", "cuStreamDestroy");
    OPT(cuModuleLoadData); OPT(cuModuleGetFunction); OPT(cuModuleUnload); OPT(cuLaunchKernel);
#undef REQ
#undef REQ2
#undef OPT
#undef OPT2
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
    for (int i = 0; i < c->ndev; i++) {
        if (c->mod[i] && c->cuModuleUnload) { c->cuCtxSetCurrent(c->ctx[i]); c->cuModuleUnload(c->mod[i]); }
        if (c->retained[i]) c->cuDevicePrimaryCtxRelease(c->devs[i]);
    }
    if (c->gds && c->cuFileDriverClose) c->cuFileDriverClose();
    if (c->cufile_lib) dlclose(c->cufile_lib);
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

/* ------------------------------------------------------ acceleration setup */

static int load_file(const char *path, char **out, size_t *len)
{
    void *d = NULL;
    size_t n = 0;
    int rc = gc_read_file(path, &d, &n);
    if (rc) return rc;
    char *s = realloc(d, n + 1);
    if (!s) { free(d); return GC_ENOMEM; }
    s[n] = 0;
    *out = s;
    *len = n;
    return GC_OK;
}

/* Compile the kernel source to PTX with NVRTC for the oldest compute
 * capability present, so the driver can JIT it for every device. */
static int nvrtc_compile(gc_cuda *c, char **ptx_out, char *why, size_t whylen)
{
    const char *cands[] = {getenv("GPUCKPT_LIBNVRTC"), "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"};
    void *lib = NULL;
    const char *used = NULL;
    char tried[512] = "";
    for (size_t i = 0; i < sizeof cands / sizeof cands[0]; i++) {
        if (!cands[i] || !*cands[i]) continue;
        lib = dlopen(cands[i], RTLD_NOW | RTLD_LOCAL);
        if (lib) { used = cands[i]; break; }
        size_t l = strlen(tried);
        snprintf(tried + l, sizeof tried - l, "%s%s", l ? ", " : "", cands[i]);
    }
    if (!lib) {
        snprintf(why, whylen, "NVRTC not found (tried %s); install the CUDA runtime compiler or set GPUCKPT_KERNEL_PTX", tried);
        return GC_EBACKEND;
    }
    fn_nvrtcCreateProgram create = (fn_nvrtcCreateProgram)dlsym(lib, "nvrtcCreateProgram");
    fn_nvrtcCompileProgram compile = (fn_nvrtcCompileProgram)dlsym(lib, "nvrtcCompileProgram");
    fn_nvrtcGetPTXSize ptxsize = (fn_nvrtcGetPTXSize)dlsym(lib, "nvrtcGetPTXSize");
    fn_nvrtcGetPTX getptx = (fn_nvrtcGetPTX)dlsym(lib, "nvrtcGetPTX");
    fn_nvrtcGetProgramLogSize logsize = (fn_nvrtcGetProgramLogSize)dlsym(lib, "nvrtcGetProgramLogSize");
    fn_nvrtcGetProgramLog getlog = (fn_nvrtcGetProgramLog)dlsym(lib, "nvrtcGetProgramLog");
    fn_nvrtcDestroyProgram destroy = (fn_nvrtcDestroyProgram)dlsym(lib, "nvrtcDestroyProgram");
    fn_nvrtcGetErrorString errstr = (fn_nvrtcGetErrorString)dlsym(lib, "nvrtcGetErrorString");
    if (!create || !compile || !ptxsize || !getptx || !destroy) {
        snprintf(why, whylen, "%s: missing NVRTC symbols", used);
        dlclose(lib);
        return GC_EBACKEND;
    }
    int major = 99, minor = 99;
    for (int i = 0; i < c->ndev; i++) {
        int ma = 0, mi = 0;
        if (c->cuDeviceGetAttribute(&ma, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, c->devs[i]) != CUDA_SUCCESS ||
            c->cuDeviceGetAttribute(&mi, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, c->devs[i]) != CUDA_SUCCESS) {
            snprintf(why, whylen, "cuDeviceGetAttribute(compute capability) failed for device %d", i);
            dlclose(lib);
            return GC_EBACKEND;
        }
        if (ma < major || (ma == major && mi < minor)) { major = ma; minor = mi; }
    }
    char arch[64];
    snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", major, minor);
    const char *opts[] = {arch, "--std=c++11"};
    nvrtcProgram_t prog = NULL;
    int rc = GC_EBACKEND;
    nvrtcResult_t nr = create(&prog, gc_kernel_sha256_src, "sha256_chunks.cu", 0, NULL, NULL);
    if (nr != 0) { snprintf(why, whylen, "nvrtcCreateProgram: %s", errstr ? errstr(nr) : "error"); goto done; }
    nr = compile(prog, 2, opts);
    if (nr != 0) {
        char logbuf[512] = "";
        size_t ls = 0;
        if (logsize && getlog && logsize(prog, &ls) == 0 && ls > 0) {
            char *lg = malloc(ls + 1);
            if (lg && getlog(prog, lg) == 0) { lg[ls] = 0; snprintf(logbuf, sizeof logbuf, "%s", lg); }
            free(lg);
        }
        snprintf(why, whylen, "nvrtcCompileProgram(%s): %s: %s", arch, errstr ? errstr(nr) : "error", logbuf);
        goto done;
    }
    size_t n = 0;
    if (ptxsize(prog, &n) != 0 || n == 0) { snprintf(why, whylen, "nvrtcGetPTXSize failed"); goto done; }
    char *ptx = malloc(n + 1);
    if (!ptx || getptx(prog, ptx) != 0) { free(ptx); snprintf(why, whylen, "nvrtcGetPTX failed"); goto done; }
    ptx[n] = 0;
    *ptx_out = ptx;
    rc = GC_OK;
done:
    if (prog) destroy(&prog);
    dlclose(lib);
    return rc;
}

int gc_cuda_enable_gpu_hash(gc_cuda *c, char *why, size_t whylen)
{
    if (c->gpu_hash) return GC_OK;
    if (!c->cuModuleLoadData || !c->cuModuleGetFunction || !c->cuLaunchKernel || !c->cuMemAlloc ||
        !c->cuMemFree || !c->cuCtxSynchronize || !c->cuDeviceGetAttribute) {
        snprintf(why, whylen, "%s lacks module/launch entry points", c->libpath);
        return GC_EBACKEND;
    }
    char *ptx = NULL;
    size_t n = 0;
    const char *pre = getenv("GPUCKPT_KERNEL_PTX");
    if (pre && *pre) {
        if (load_file(pre, &ptx, &n)) { snprintf(why, whylen, "GPUCKPT_KERNEL_PTX=%s: unreadable", pre); return GC_EBACKEND; }
    } else if (nvrtc_compile(c, &ptx, why, whylen)) {
        return GC_EBACKEND;
    }
    for (int i = 0; i < c->ndev; i++) {
        c->cuCtxSetCurrent(c->ctx[i]);
        CUresult r = c->cuModuleLoadData(&c->mod[i], ptx);
        if (r == CUDA_SUCCESS) r = c->cuModuleGetFunction(&c->fn_hash[i], c->mod[i], "gc_sha256_chunks");
        if (r != CUDA_SUCCESS) {
            snprintf(why, whylen, "loading kernel on device %d: CUresult %d (%s)", i, (int)r, cu_errstr(c, r));
            for (int j = 0; j <= i; j++)
                if (c->mod[j]) { c->cuCtxSetCurrent(c->ctx[j]); c->cuModuleUnload(c->mod[j]); c->mod[j] = NULL; }
            free(ptx);
            return GC_EBACKEND;
        }
    }
    free(ptx);
    c->gpu_hash = 1;
    return GC_OK;
}

int gc_cuda_enable_gds(gc_cuda *c, char *why, size_t whylen)
{
    if (c->gds) return GC_OK;
    const char *path = getenv("GPUCKPT_LIBCUFILE");
    const char *cands[] = {path, "libcufile.so.0", "libcufile.so"};
    for (size_t i = 0; i < 3 && !c->cufile_lib; i++)
        if (cands[i] && *cands[i]) c->cufile_lib = dlopen(cands[i], RTLD_NOW | RTLD_LOCAL);
    if (!c->cufile_lib) { snprintf(why, whylen, "libcufile not found (install GPUDirect Storage or set GPUCKPT_LIBCUFILE): %s", dlerror()); return GC_EBACKEND; }
    void *L = c->cufile_lib;
    c->cuFileDriverOpen = (fn_cuFileDriverOpen)dlsym(L, "cuFileDriverOpen");
    c->cuFileDriverClose = (fn_cuFileDriverClose)dlsym(L, "cuFileDriverClose_v2");
    if (!c->cuFileDriverClose) c->cuFileDriverClose = (fn_cuFileDriverClose)dlsym(L, "cuFileDriverClose");
    c->cuFileHandleRegister = (fn_cuFileHandleRegister)dlsym(L, "cuFileHandleRegister");
    c->cuFileHandleDeregister = (fn_cuFileHandleDeregister)dlsym(L, "cuFileHandleDeregister");
    c->cuFileBufRegister = (fn_cuFileBufRegister)dlsym(L, "cuFileBufRegister");
    c->cuFileBufDeregister = (fn_cuFileBufDeregister)dlsym(L, "cuFileBufDeregister");
    c->cuFileRead = (fn_cuFileRead)dlsym(L, "cuFileRead");
    c->cuFileWrite = (fn_cuFileWrite)dlsym(L, "cuFileWrite");
    if (!c->cuFileDriverOpen || !c->cuFileHandleRegister || !c->cuFileHandleDeregister || !c->cuFileRead || !c->cuFileWrite) {
        snprintf(why, whylen, "libcufile lacks required symbols");
        dlclose(L); c->cufile_lib = NULL;
        return GC_EBACKEND;
    }
    c->cuCtxSetCurrent(c->ctx[0]);
    CUfileError_t e = c->cuFileDriverOpen();
    if (e.err != CU_FILE_SUCCESS) {
        snprintf(why, whylen, "cuFileDriverOpen: cuFile error %d, CUresult %d (nvidia-fs loaded? see cufile.json)", (int)e.err, (int)e.cu_err);
        dlclose(L); c->cufile_lib = NULL;
        return GC_EBACKEND;
    }
    c->gds = 1;
    return GC_OK;
}

/* ------------------------------------------------------------ image ops */

static int ctx_index(cuda_priv *p, const gc_image *img, uint32_t d)
{
    int o = img->dev[d].ordinal;
    return (o >= 0 && o < p->c->ndev) ? o : 0;
}

static int check_live(cuda_priv *p, const gc_image *img, uint32_t d, uint64_t off, size_t len)
{
    if (p->completed) { gc_set_error("mapping already released by cuCheckpointOperationComplete"); return GC_ESTATE; }
    if (off + len > img->dev[d].size) return GC_EINVAL;
    return GC_OK;
}

static int cu_read(gc_image *img, uint32_t d, uint64_t off, void *dst, size_t len)
{
    cuda_priv *p = img->priv;
    int rc = check_live(p, img, d, off, len);
    if (rc) return rc;
    p->c->cuCtxSetCurrent(p->c->ctx[ctx_index(p, img, d)]);
    CUresult r = p->c->cuMemcpyDtoH(dst, p->ptr[d] + off, len);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyDtoH", r);
    return GC_OK;
}

static int cu_write(gc_image *img, uint32_t d, uint64_t off, const void *src, size_t len)
{
    cuda_priv *p = img->priv;
    int rc = check_live(p, img, d, off, len);
    if (rc) return rc;
    p->c->cuCtxSetCurrent(p->c->ctx[ctx_index(p, img, d)]);
    CUresult r = p->c->cuMemcpyHtoD(p->ptr[d] + off, src, len);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyHtoD", r);
    return GC_OK;
}

static int cu_attach(gc_image *img, void **tctx)
{
    cuda_priv *p = img->priv;
    if (p->c->ndev > 0) {
        CUresult r = p->c->cuCtxSetCurrent(p->c->ctx[0]);
        if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuCtxSetCurrent", r);
    }
    *tctx = calloc(1, sizeof(cuda_tctx));
    return *tctx ? GC_OK : GC_ENOMEM;
}

static void cu_detach(gc_image *img, void *tctx)
{
    cuda_priv *p = img->priv;
    cuda_tctx *t = tctx;
    if (!t) return;
    for (int i = 0; i < GC_MAX_DEVICES; i++)
        for (int s = 0; s < MAX_SLOTS; s++)
            if (t->s[i][s]) { p->c->cuCtxSetCurrent(p->c->ctx[i]); p->c->cuStreamDestroy(t->s[i][s]); }
    free(t);
}

static int get_stream(cuda_priv *p, cuda_tctx *t, int ci, int slot, CUstream *out)
{
    if (!t->s[ci][slot]) {
        CUresult r = p->c->cuStreamCreate(&t->s[ci][slot], CU_STREAM_NON_BLOCKING);
        if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuStreamCreate", r);
    }
    *out = t->s[ci][slot];
    return GC_OK;
}

static int cu_read_start(gc_image *img, void *tctx, int slot, uint32_t d, uint64_t off, void *dst, size_t len)
{
    cuda_priv *p = img->priv;
    cuda_tctx *t = tctx;
    int rc = check_live(p, img, d, off, len);
    if (rc) return rc;
    int ci = ctx_index(p, img, d);
    p->c->cuCtxSetCurrent(p->c->ctx[ci]);
    CUstream s;
    if ((rc = get_stream(p, t, ci, slot, &s))) return rc;
    t->slot_ctx[slot] = ci;
    CUresult r = p->c->cuMemcpyDtoHAsync(dst, p->ptr[d] + off, len, s);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyDtoHAsync", r);
    return GC_OK;
}

static int cu_write_start(gc_image *img, void *tctx, int slot, uint32_t d, uint64_t off, const void *src, size_t len)
{
    cuda_priv *p = img->priv;
    cuda_tctx *t = tctx;
    int rc = check_live(p, img, d, off, len);
    if (rc) return rc;
    int ci = ctx_index(p, img, d);
    p->c->cuCtxSetCurrent(p->c->ctx[ci]);
    CUstream s;
    if ((rc = get_stream(p, t, ci, slot, &s))) return rc;
    t->slot_ctx[slot] = ci;
    CUresult r = p->c->cuMemcpyHtoDAsync(p->ptr[d] + off, src, len, s);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuMemcpyHtoDAsync", r);
    return GC_OK;
}

static int cu_wait(gc_image *img, void *tctx, int slot)
{
    cuda_priv *p = img->priv;
    cuda_tctx *t = tctx;
    int ci = t->slot_ctx[slot];
    if (!t->s[ci][slot]) return GC_OK;
    p->c->cuCtxSetCurrent(p->c->ctx[ci]);
    CUresult r = p->c->cuStreamSynchronize(t->s[ci][slot]);
    if (r != CUDA_SUCCESS) return cu_fail(p->c, "cuStreamSynchronize", r);
    return GC_OK;
}

static int cu_host_register(gc_image *img, void *ptr, size_t len, char *why, size_t whylen)
{
    cuda_priv *p = img->priv;
    if (p->c->ndev) p->c->cuCtxSetCurrent(p->c->ctx[0]);
    CUresult r = p->c->cuMemHostRegister(ptr, len, CU_MEMHOSTREGISTER_PORTABLE);
    if (r != CUDA_SUCCESS) { snprintf(why, whylen, "cuMemHostRegister: CUresult %d (%s)", (int)r, cu_errstr(p->c, r)); return -1; }
    return 0;
}

static void cu_host_unregister(gc_image *img, void *ptr)
{
    cuda_priv *p = img->priv;
    if (p->c->ndev) p->c->cuCtxSetCurrent(p->c->ctx[0]);
    p->c->cuMemHostUnregister(ptr);
}

static void *cu_host_alloc(gc_image *img, size_t len, char *why, size_t whylen)
{
    cuda_priv *p = img->priv;
    void *b = NULL;
    if (p->c->ndev) p->c->cuCtxSetCurrent(p->c->ctx[0]);
    CUresult r = p->c->cuMemHostAlloc(&b, len, CU_MEMHOSTALLOC_PORTABLE);
    if (r != CUDA_SUCCESS || !b) { snprintf(why, whylen, "cuMemHostAlloc: CUresult %d (%s)", (int)r, cu_errstr(p->c, r)); return NULL; }
    return b;
}

static void cu_host_free(gc_image *img, void *ptr)
{
    cuda_priv *p = img->priv;
    if (p->c->ndev) p->c->cuCtxSetCurrent(p->c->ctx[0]);
    p->c->cuMemFreeHost(ptr);
}

/* SHA-256 of every chunk of device d, computed by a kernel reading the
 * custom-storage mapping in place. Only the digests come back to the host.
 * A fault inside the kernel (e.g. if the mapping is not kernel-readable)
 * surfaces at cuCtxSynchronize; such errors are sticky for the context,
 * which is why --gpu-hash defaults to off until gate 7 is verified. */
static int cu_hash_chunks(gc_image *img, uint32_t d, uint64_t chunk_size, uint8_t (*out)[GC_HASH_LEN])
{
    cuda_priv *p = img->priv;
    gc_cuda *c = p->c;
    int rc = check_live(p, img, d, 0, 0);
    if (rc) return rc;
    uint64_t size = img->dev[d].size;
    uint64_t n = (size + chunk_size - 1) / chunk_size;
    if (n == 0) return GC_OK;
    if ((n + 127) / 128 > 0x7fffffffull) { gc_set_error("too many chunks for one launch"); return GC_EINVAL; }
    int ci = ctx_index(p, img, d);
    c->cuCtxSetCurrent(c->ctx[ci]);
    CUdeviceptr dout = 0;
    CUresult r = c->cuMemAlloc(&dout, (size_t)n * GC_HASH_LEN);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuMemAlloc(digests)", r);
    CUdeviceptr base = p->ptr[d];
    uint64_t sz = size, cs = chunk_size, nn = n;
    void *args[] = {&base, &sz, &cs, &nn, &dout};
    unsigned grid = (unsigned)((n + 127) / 128);
    r = c->cuLaunchKernel(c->fn_hash[ci], grid, 1, 1, 128, 1, 1, 0, NULL, args, NULL);
    if (r == CUDA_SUCCESS) r = c->cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        cu_fail(c, "gc_sha256_chunks kernel", r);
        c->cuMemFree(dout);
        /* Never launch again in this process: a kernel fault is sticky for
         * the context, and later images must not be offered the kernel. */
        c->gpu_hash = 0;
        img->hash_chunks = NULL;
        return GC_EBACKEND;
    }
    r = c->cuMemcpyDtoH(out, dout, (size_t)n * GC_HASH_LEN);
    c->cuMemFree(dout);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuMemcpyDtoH(digests)", r);
    return GC_OK;
}

/* GPUDirect Storage: one registered handle per chunk file. */
static int gds_xfer(gc_image *img, uint32_t d, uint64_t off, size_t len, int fd, int to_file)
{
    cuda_priv *p = img->priv;
    gc_cuda *c = p->c;
    int rc = check_live(p, img, d, off, len);
    if (rc) return rc;
    c->cuCtxSetCurrent(c->ctx[ctx_index(p, img, d)]);
    CUfileDescr_t desc;
    memset(&desc, 0, sizeof desc);
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    desc.handle.fd = fd;
    CUfileHandle_t fh;
    CUfileError_t e = c->cuFileHandleRegister(&fh, &desc);
    if (e.err != CU_FILE_SUCCESS) {
        gc_set_error("cuFileHandleRegister: cuFile error %d, CUresult %d", (int)e.err, (int)e.cu_err);
        return GC_EBACKEND;
    }
    void *base = (void *)(uintptr_t)p->ptr[d];
    ssize_t n = to_file ? c->cuFileWrite(fh, base, len, 0, (off_t)off)
                        : c->cuFileRead(fh, base, len, 0, (off_t)off);
    c->cuFileHandleDeregister(fh);
    if (n != (ssize_t)len) {
        if (n == -1) gc_set_error("%s: %s", to_file ? "cuFileWrite" : "cuFileRead", strerror(errno));
        else if (n < 0) gc_set_error("%s: cuFile error %zd", to_file ? "cuFileWrite" : "cuFileRead", -n);
        else gc_set_error("%s: %zd of %zu bytes", to_file ? "cuFileWrite" : "cuFileRead", n, len);
        return GC_EIO;
    }
    return GC_OK;
}

static int cu_dev_to_file(gc_image *img, void *tctx, uint32_t d, uint64_t off, size_t len, int fd)
{
    (void)tctx;
    return gds_xfer(img, d, off, len, fd, 1);
}

static int cu_file_to_dev(gc_image *img, void *tctx, uint32_t d, uint64_t off, size_t len, int fd)
{
    (void)tctx;
    return gds_xfer(img, d, off, len, fd, 0);
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
 * table the driver handed back and describe it as a gc_image, exposing the
 * accelerations this gc_cuda has enabled. */
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
    img->priv = p;
    img->thread_attach = cu_attach;
    img->thread_detach = cu_detach;
    if (c->cuMemcpyDtoHAsync && c->cuStreamCreate && c->cuStreamDestroy) {
        if (!p->is_restore) img->read_start = cu_read_start;
        img->io_wait = cu_wait;
    }
    if (p->is_restore && c->cuMemcpyHtoDAsync && c->cuStreamCreate && c->cuStreamDestroy) img->write_start = cu_write_start;
    if (c->cuMemHostRegister && c->cuMemHostUnregister) { img->host_register = cu_host_register; img->host_unregister = cu_host_unregister; }
    if (c->cuMemHostAlloc && c->cuMemFreeHost) { img->host_alloc_pinned = cu_host_alloc; img->host_free_pinned = cu_host_free; }
    if (c->gpu_hash) img->hash_chunks = cu_hash_chunks;
    if (c->gds) {
        img->dev_to_file = cu_dev_to_file;
        img->file_to_dev = cu_file_to_dev;
        img->direct_odirect = 1;
        /* Registering the mapping with cuFile lets GDS DMA straight into it;
         * if registration fails (e.g. BAR1 space), cuFile still works through
         * its own GPU-side bounce buffers. Deregistered before complete. */
        if (c->cuFileBufRegister)
            for (unsigned i = 0; i < img->device_count; i++) {
                c->cuCtxSetCurrent(c->ctx[ctx_index(p, img, i)]);
                CUfileError_t e = c->cuFileBufRegister((const void *)(uintptr_t)p->ptr[i], (size_t)img->dev[i].size, 0);
                p->buf_registered[i] = e.err == CU_FILE_SUCCESS;
            }
    }
    return GC_OK;
}

int gc_cuda_pinning_image(gc_cuda *c, gc_image *img)
{
    memset(img, 0, sizeof *img);
    cuda_priv *p = calloc(1, sizeof *p);
    if (!p) return GC_ENOMEM;
    p->c = c;
    p->completed = 1;               /* no mapping: data-path entry points refuse */
    snprintf(img->backend, sizeof img->backend, "cuda");
    img->priv = p;
    if (c->cuMemHostRegister && c->cuMemHostUnregister) { img->host_register = cu_host_register; img->host_unregister = cu_host_unregister; }
    if (c->cuMemHostAlloc && c->cuMemFreeHost) { img->host_alloc_pinned = cu_host_alloc; img->host_free_pinned = cu_host_free; }
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
    img->read = cu_read;        /* read-back, used to re-check chunks device verification flags */
    return GC_OK;
}

static void gds_deregister(gc_cuda *c, cuda_priv *p, gc_image *img)
{
    for (uint32_t i = 0; i < img->device_count; i++)
        if (p->buf_registered[i] && c->cuFileBufDeregister) {
            c->cuCtxSetCurrent(c->ctx[ctx_index(p, img, i)]);
            c->cuFileBufDeregister((const void *)(uintptr_t)p->ptr[i]);
            p->buf_registered[i] = 0;
        }
}

int gc_cuda_op_complete(gc_cuda *c, gc_image *img)
{
    cuda_priv *p = img->priv;
    if (!p) return GC_EINVAL;
    if (p->completed) return GC_OK;
    gds_deregister(c, p, img);               /* complete unmaps the memory cuFile pinned */
    for (uint32_t i = 0; i < img->device_count; i++)
        if (p->stream[i]) c->cuStreamSynchronize(p->stream[i]);
    CUresult r = c->cuCheckpointOperationComplete(p->handle);
    if (r != CUDA_SUCCESS) return cu_fail(c, "cuCheckpointOperationComplete", r);
    p->completed = 1;
    return GC_OK;
}

void gc_cuda_image_close(gc_cuda *c, gc_image *img)
{
    cuda_priv *p = img->priv;
    if (p && !p->completed) gds_deregister(c, p, img);
    if (!p) return;
    free(img->priv);
    img->priv = NULL;
}
