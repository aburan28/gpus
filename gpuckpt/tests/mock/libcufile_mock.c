/* Mock libcufile (GPUDirect Storage). Device memory in the mock driver is
 * host memory, so a "direct" transfer is a pwrite/pread from the device
 * pointer; what matters for the tests is that the library never staged the
 * bytes itself, which the mock driver's copy counters show.
 * Writes cufile.txt into $GPUCKPT_MOCK_COUNTERS at exit.
 *   GPUCKPT_MOCK_CUFILE_FAIL=1   cuFileDriverOpen fails (no nvidia-fs) */
#define _GNU_SOURCE
#include <cuda.h>
#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned long long w_bytes, r_bytes, handles, bufreg, opened;
/* called from concurrent worker threads: atomic, so the tests' exact byte
 * counts cannot lose updates */
#define CADD(var, n) __atomic_fetch_add(&(var), (unsigned long long)(n), __ATOMIC_RELAXED)

__attribute__((destructor)) static void dump(void)
{
    const char *dir = getenv("GPUCKPT_MOCK_COUNTERS");
    if (!dir || !*dir) return;
    char p[4096];
    snprintf(p, sizeof p, "%s/cufile.txt", dir);
    FILE *f = fopen(p, "w");
    if (!f) return;
    fprintf(f, "write_bytes %llu\nread_bytes %llu\nhandles %llu\nbuf_registered %llu\ndriver_opened %llu\n",
            w_bytes, r_bytes, handles, bufreg, opened);
    fclose(f);
}

static CUfileError_t ok(void) { CUfileError_t e = {CU_FILE_SUCCESS, CUDA_SUCCESS}; return e; }

CUfileError_t cuFileDriverOpen(void)
{
    if (getenv("GPUCKPT_MOCK_CUFILE_FAIL")) { CUfileError_t e = {CU_FILE_DRIVER_NOT_INITIALIZED, CUDA_SUCCESS}; return e; }
    CADD(opened, 1);
    return ok();
}
CUfileError_t cuFileDriverClose_v2(void) { return ok(); }

typedef struct { int fd; } mh;
CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *d)
{
    if (d->type != CU_FILE_HANDLE_TYPE_OPAQUE_FD || d->handle.fd < 0) { CUfileError_t e = {CU_FILE_IO_NOT_SUPPORTED, CUDA_SUCCESS}; return e; }
    mh *h = malloc(sizeof *h);
    h->fd = d->handle.fd;
    *fh = h;
    CADD(handles, 1);
    return ok();
}
void cuFileHandleDeregister(CUfileHandle_t fh) { free(fh); }
CUfileError_t cuFileBufRegister(const void *p, size_t n, int flags) { (void)p; (void)n; (void)flags; CADD(bufreg, 1); return ok(); }
CUfileError_t cuFileBufDeregister(const void *p) { (void)p; return ok(); }

/* O_DIRECT needs aligned host buffers, which the mock's "device" memory
 * is not; drop the flag for the mock's own pwrite/pread. */
static void plain(int fd) { int fl = fcntl(fd, F_GETFL); if (fl >= 0 && (fl & O_DIRECT)) fcntl(fd, F_SETFL, fl & ~O_DIRECT); }

ssize_t cuFileWrite(CUfileHandle_t fh, const void *base, size_t n, off_t foff, off_t boff)
{
    int fd = ((mh *)fh)->fd;
    plain(fd);
    size_t done = 0;
    while (done < n) {
        ssize_t w = pwrite(fd, (const char *)base + boff + done, n - done, foff + (off_t)done);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        done += (size_t)w;
    }
    CADD(w_bytes, n);
    return (ssize_t)n;
}

ssize_t cuFileRead(CUfileHandle_t fh, void *base, size_t n, off_t foff, off_t boff)
{
    int fd = ((mh *)fh)->fd;
    plain(fd);
    size_t done = 0;
    while (done < n) {
        ssize_t r = pread(fd, (char *)base + boff + done, n - done, foff + (off_t)done);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) break;
        done += (size_t)r;
    }
    CADD(r_bytes, done);
    return (ssize_t)done;
}
