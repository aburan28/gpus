/* Real-driver graph replay workload. No mock CUDA header or nvcc required.
 * The executable graph, module, stream and allocation are created once and
 * remain alive across every checkpoint. stdin/stdout form a barrier protocol
 * for scripts/graph-replay.py; diagnostics go to stderr. */
#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define API_LIST(X) \
    X(cuInit) X(cuGetErrorString) X(cuDriverGetVersion) \
    X(cuDeviceGet) X(cuDeviceGetName) X(cuDeviceGetUuid) \
    X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent) \
    X(cuModuleLoadData) X(cuModuleGetFunction) X(cuModuleUnload) \
    X(cuMemAlloc) X(cuMemFree) X(cuMemAllocHost) X(cuMemFreeHost) \
    X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuMemcpyDtoHAsync) \
    X(cuStreamCreate) X(cuStreamDestroy) X(cuStreamSynchronize) \
    X(cuStreamBeginCapture) X(cuStreamEndCapture) X(cuLaunchKernel) \
    X(cuGraphCreate) X(cuGraphAddKernelNode) X(cuGraphAddMemcpyNode) \
    X(cuGraphInstantiateWithFlags) X(cuGraphUpload) X(cuGraphLaunch) \
    X(cuGraphExecDestroy) X(cuGraphDestroy)
#define DECLARE(name) static __typeof__(&name) p_##name;
API_LIST(DECLARE)
#undef DECLARE
#define STR_INNER(x) #x
#define STR(x) STR_INNER(x)

static void fail(const char *what, CUresult result)
{
    const char *detail = NULL;
    if (p_cuGetErrorString) p_cuGetErrorString(result, &detail);
    fprintf(stderr, "%s: CUresult=%d %s\n", what, result, detail ? detail : "");
    exit(2);
}
#define CUDA(call) do { CUresult r_ = (call); if (r_ != CUDA_SUCCESS) fail(#call, r_); } while (0)

/* A plain, deterministic kernel keeps the fixture independent of nvcc,
 * libraries such as cuBLAS, and any particular research application. */
static const char *ptx =
    ".version 6.0\n.target sm_50\n.address_size 64\n"
    ".visible .entry graph_step(.param .u64 data, .param .u32 count) {\n"
    " .reg .pred pred; .reg .b32 idx, block, size, lane, n, value;\n"
    " .reg .b64 ptr, offset, address;\n"
    " ld.param.u64 ptr, [data]; ld.param.u32 n, [count];\n"
    " mov.u32 block, %ctaid.x; mov.u32 size, %ntid.x; mov.u32 lane, %tid.x;\n"
    " mad.lo.u32 idx, block, size, lane; setp.ge.u32 pred, idx, n;\n"
    " @pred bra done; mul.wide.u32 offset, idx, 4; add.u64 address, ptr, offset;\n"
    " ld.global.u32 value, [address]; add.u32 value, value, 3;\n"
    " st.global.u32 [address], value; done: ret; }\n";

enum { ELEMENTS = 4096 };
static uint32_t *host;
static CUdeviceptr device_ptr;
static CUgraphExec executable;
static CUstream stream;
static uint64_t launches;
static int driver_version;
static char device_name[256];
static CUuuid device_uuid;
static const char *mode;

static void verify(void)
{
    for (unsigned i = 0; i < ELEMENTS; ++i) {
        uint32_t expected = (uint32_t)(i * 17u + 11u + launches * 3u);
        if (host[i] != expected) {
            fprintf(stderr, "graph output mismatch at %u: got=%u expected=%u launches=%llu\n",
                    i, host[i], expected, (unsigned long long)launches);
            exit(2);
        }
    }
}

static void emit(const char *event)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    /* Hash words, with explicitly little-endian byte order. */
    for (unsigned i = 0; i < ELEMENTS; ++i)
        for (unsigned b = 0; b < 4; ++b) {
            hash ^= (host[i] >> (8 * b)) & 255u;
            hash *= UINT64_C(1099511628211);
        }
    printf("{\"event\":\"%s\",\"pid\":%ld,\"mode\":\"%s\","
           "\"launches\":%llu,\"elements\":%u,\"checksum\":\"%016llx\","
           "\"graph_exec\":\"%llx\",\"device_ptr\":\"%llx\","
           "\"graph_instantiations\":1,\"driver_version\":%d,\"device_uuid\":\"",
           event, (long)getpid(), mode, (unsigned long long)launches, ELEMENTS,
           (unsigned long long)hash, (unsigned long long)(uintptr_t)executable,
           (unsigned long long)device_ptr, driver_version);
    for (unsigned i = 0; i < sizeof device_uuid.bytes; ++i)
        printf("%02x", (unsigned char)device_uuid.bytes[i]);
    /* GPU names need not be escaped because they are printed to stderr. */
    puts("\"}");
    fflush(stdout);
}

static void replay(uint64_t count)
{
    for (uint64_t i = 0; i < count; ++i) CUDA(p_cuGraphLaunch(executable, stream));
    CUDA(p_cuStreamSynchronize(stream));
    launches += count;
    /* This checks the DtoH node inside the graph, before any independent copy. */
    verify();
}

int main(int argc, char **argv)
{
    mode = argc == 2 ? argv[1] : "";
    if (strcmp(mode, "capture") && strcmp(mode, "explicit")) {
        fprintf(stderr, "usage: graph_replay capture|explicit\n");
        return 1;
    }
    void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "libcuda.so.1: %s\n", dlerror()); return 77; }
#define LOAD(name) do { \
        *(void **)(&p_##name) = dlsym(library, STR(name)); \
        if (!p_##name) { fprintf(stderr, "missing driver symbol %s\n", STR(name)); return 77; } \
    } while (0);
    API_LIST(LOAD)
#undef LOAD
    CUDA(p_cuInit(0));
    CUDA(p_cuDriverGetVersion(&driver_version));
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUgraph graph;
    CUDA(p_cuDeviceGet(&device, 0));
    CUDA(p_cuDeviceGetName(device_name, sizeof device_name, device));
    CUDA(p_cuDeviceGetUuid(&device_uuid, device));
    fprintf(stderr, "device=%s driver_version=%d CUDA_VERSION=%d\n", device_name, driver_version, CUDA_VERSION);
    CUDA(p_cuDevicePrimaryCtxRetain(&context, device));
    CUDA(p_cuCtxSetCurrent(context));
    CUDA(p_cuModuleLoadData(&module, ptx));
    CUDA(p_cuModuleGetFunction(&function, module, "graph_step"));
    CUDA(p_cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUDA(p_cuMemAlloc(&device_ptr, ELEMENTS * sizeof *host));
    CUDA(p_cuMemAllocHost((void **)&host, ELEMENTS * sizeof *host));
    for (unsigned i = 0; i < ELEMENTS; ++i) host[i] = i * 17u + 11u;
    CUDA(p_cuMemcpyHtoD(device_ptr, host, ELEMENTS * sizeof *host));
    unsigned elements = ELEMENTS;
    void *arguments[] = { &device_ptr, &elements };
    if (!strcmp(mode, "capture")) {
        CUDA(p_cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
        CUDA(p_cuLaunchKernel(function, ELEMENTS / 128, 1, 1, 128, 1, 1, 0, stream, arguments, NULL));
        CUDA(p_cuMemcpyDtoHAsync(host, device_ptr, ELEMENTS * sizeof *host, stream));
        CUDA(p_cuStreamEndCapture(stream, &graph));
    } else {
        CUDA(p_cuGraphCreate(&graph, 0));
        CUDA_KERNEL_NODE_PARAMS kernel = {0};
        kernel.func = function;
        kernel.gridDimX = ELEMENTS / 128; kernel.gridDimY = kernel.gridDimZ = 1;
        kernel.blockDimX = 128; kernel.blockDimY = kernel.blockDimZ = 1;
        kernel.kernelParams = arguments;
        CUgraphNode kernel_node, copy_node;
        CUDA(p_cuGraphAddKernelNode(&kernel_node, graph, NULL, 0, &kernel));
        CUDA_MEMCPY3D copy = {0};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = device_ptr;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = host;
        copy.WidthInBytes = ELEMENTS * sizeof *host;
        copy.srcPitch = copy.dstPitch = copy.WidthInBytes;
        copy.Height = copy.Depth = 1;
        CUDA(p_cuGraphAddMemcpyNode(&copy_node, graph, &kernel_node, 1, &copy, context));
    }
    CUDA(p_cuGraphInstantiateWithFlags(&executable, graph, 0));
    CUDA(p_cuGraphUpload(executable, stream));
    CUDA(p_cuStreamSynchronize(stream));
    replay(2);
    emit("ready");
    char line[128];
    while (fgets(line, sizeof line, stdin)) {
        unsigned long long count;
        char extra;
        if (!strcmp(line, "check\n")) {
            CUDA(p_cuMemcpyDtoH(host, device_ptr, ELEMENTS * sizeof *host));
            verify(); emit("checked");
        } else if (sscanf(line, "replay %llu %c", &count, &extra) == 1 && count && count <= 1000000) {
            replay(count); emit("replayed");
        } else if (!strcmp(line, "quit\n")) {
            emit("done"); break;
        } else { fprintf(stderr, "invalid command\n"); return 1; }
    }
    CUDA(p_cuGraphExecDestroy(executable));
    CUDA(p_cuGraphDestroy(graph));
    CUDA(p_cuMemFreeHost(host));
    CUDA(p_cuMemFree(device_ptr));
    CUDA(p_cuStreamDestroy(stream));
    CUDA(p_cuModuleUnload(module));
    CUDA(p_cuDevicePrimaryCtxRelease(device));
    dlclose(library);
    return 0;
}
