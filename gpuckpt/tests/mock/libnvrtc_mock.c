/* Mock NVRTC. "Compiles" by checking that the source is the gpuckpt kernel
 * and that a --gpu-architecture option was passed, then returns a PTX stub
 * tagged with that architecture; the mock driver's cuModuleLoadData accepts
 * only the stub for its own device (compute_90). The kernel's arithmetic is
 * exercised separately, compiled as C, by the mock driver and kernel_test.
 *   GPUCKPT_MOCK_NVRTC_FAIL=1   compilation fails with a log message */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *src; char ptx[128]; char log[128]; } prog;

int nvrtcCreateProgram(void **p, const char *src, const char *name, int nh, const char *const *h, const char *const *hn)
{
    (void)name; (void)nh; (void)h; (void)hn;
    prog *g = calloc(1, sizeof *g);
    g->src = strdup(src);
    *p = g;
    return 0;
}

int nvrtcCompileProgram(void *p, int n, const char *const *opts)
{
    prog *g = p;
    if (getenv("GPUCKPT_MOCK_NVRTC_FAIL")) { snprintf(g->log, sizeof g->log, "mock: injected compile failure"); return 6; }
    if (!strstr(g->src, "extern \"C\" __global__ void gc_sha256_chunks")) { snprintf(g->log, sizeof g->log, "mock: not the gpuckpt kernel"); return 6; }
    const char *arch = NULL;
    for (int i = 0; i < n; i++) if (!strncmp(opts[i], "--gpu-architecture=", 19)) arch = opts[i] + 19;
    if (!arch) { snprintf(g->log, sizeof g->log, "mock: no --gpu-architecture"); return 6; }
    snprintf(g->ptx, sizeof g->ptx, "// MOCKPTX arch=%s\n", arch);
    return 0;
}

int nvrtcGetPTXSize(void *p, size_t *n) { *n = strlen(((prog *)p)->ptx) + 1; return 0; }
int nvrtcGetPTX(void *p, char *out) { strcpy(out, ((prog *)p)->ptx); return 0; }
int nvrtcGetProgramLogSize(void *p, size_t *n) { *n = strlen(((prog *)p)->log) + 1; return 0; }
int nvrtcGetProgramLog(void *p, char *out) { strcpy(out, ((prog *)p)->log); return 0; }
int nvrtcDestroyProgram(void **p) { prog *g = *p; free(g->src); free(g); *p = NULL; return 0; }
const char *nvrtcGetErrorString(int r) { return r == 0 ? "NVRTC_SUCCESS" : r == 6 ? "NVRTC_ERROR_COMPILATION" : "NVRTC_ERROR"; }
