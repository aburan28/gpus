#include "internal.h"
#include <stdarg.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

static __thread char g_err[512];

const char *gc_last_error(void) { return g_err; }

void gc_set_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

const char *gc_strerror(int rc)
{
    switch (rc) {
    case GC_OK: return "ok";
    case GC_EINVAL: return "invalid argument or format";
    case GC_EIO: return "I/O error";
    case GC_ENOENT: return "not found";
    case GC_EEXIST: return "already exists (records are immutable)";
    case GC_ECORRUPT: return "stored data does not match its hash";
    case GC_EMISMATCH: return "image shape does not match manifest";
    case GC_EBACKEND: return "backend error";
    case GC_ESTATE: return "target process in wrong checkpoint state";
    case GC_ENOMEM: return "out of memory";
    default: return "unknown error";
    }
}

uint64_t gc_mono_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t gc_wall_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void gc_hex(const uint8_t *h, size_t n, char *out)
{
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = d[h[i] >> 4];
        out[2 * i + 1] = d[h[i] & 15];
    }
    out[2 * n] = 0;
}

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int gc_unhex(const char *hex, uint8_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        int a = hexval(hex[2 * i]), b = hexval(hex[2 * i + 1]);
        if (a < 0 || b < 0) return GC_EINVAL;
        out[i] = (uint8_t)(a << 4 | b);
    }
    if (hex[2 * n] != 0) return GC_EINVAL;
    return GC_OK;
}

int gc_mkdir_p(const char *path)
{
    char buf[PATH_MAX];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof buf) return GC_EINVAL;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (buf[i] == '/' || buf[i] == 0) {
            char c = buf[i];
            buf[i] = 0;
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) {
                gc_set_error("mkdir %s: %s", buf, strerror(errno));
                return GC_EIO;
            }
            buf[i] = c;
        }
    }
    return GC_OK;
}

int gc_file_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static int fsync_dir(const char *file_path)
{
    char dir[PATH_MAX];
    size_t n = strlen(file_path);
    if (n >= sizeof dir) return -1;
    memcpy(dir, file_path, n + 1);
    char *slash = strrchr(dir, '/');
    if (!slash) strcpy(dir, ".");
    else if (slash == dir) dir[1] = 0;
    else *slash = 0;
    int fd = open(dir, O_RDONLY | O_DIRECTORY);
    if (fd < 0) return -1;
    int rc = fsync(fd);
    close(fd);
    return rc;
}

/* Write data to a unique temp file in tmp_dir, fsync, then publish it at
 * final_path. Publication uses link(2), which fails with EEXIST if the
 * target already exists, so exactly one of N concurrent writers of the same
 * content observes success (GC_OK) and the others observe GC_EEXIST. Readers
 * never see a partial file because the name appears only after the data and
 * its fsync. With allow_exists the GC_EEXIST is a benign dedup signal; without
 * it the caller treats it as an immutability violation. */
int gc_write_file_atomic(const char *final_path, const char *tmp_dir,
                         const void *data, size_t len, int allow_exists)
{
    (void)allow_exists;
    if (gc_file_exists(final_path)) return GC_EEXIST;
    char tmp[PATH_MAX];
    static __thread unsigned seq;
    int n = snprintf(tmp, sizeof tmp, "%s/.tmp.%ld.%lu.%u", tmp_dir, (long)getpid(),
                     (unsigned long)pthread_self(), seq++);
    if (n < 0 || (size_t)n >= sizeof tmp) return GC_EINVAL;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (fd < 0) { gc_set_error("open %s: %s", tmp, strerror(errno)); return GC_EIO; }
    const uint8_t *p = data;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            gc_set_error("write %s: %s", tmp, strerror(errno));
            close(fd); unlink(tmp); return GC_EIO;
        }
        off += (size_t)w;
    }
    if (fsync(fd) != 0) { gc_set_error("fsync %s: %s", tmp, strerror(errno)); close(fd); unlink(tmp); return GC_EIO; }
    close(fd);
    int rc = GC_OK;
    if (link(tmp, final_path) != 0) {
        if (errno == EEXIST) rc = GC_EEXIST;
        else { gc_set_error("link %s -> %s: %s", tmp, final_path, strerror(errno)); rc = GC_EIO; }
    }
    unlink(tmp);
    if (rc == GC_OK) fsync_dir(final_path);
    return rc;
}

int gc_read_file(const char *path, void **data, size_t *len)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) { gc_set_error("open %s: %s", path, strerror(errno)); return errno == ENOENT ? GC_ENOENT : GC_EIO; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return GC_EIO; }
    size_t n = (size_t)st.st_size;
    uint8_t *buf = malloc(n ? n : 1);
    if (!buf) { close(fd); return GC_ENOMEM; }
    size_t off = 0;
    while (off < n) {
        ssize_t r = read(fd, buf + off, n - off);
        if (r < 0) { if (errno == EINTR) continue; free(buf); close(fd); return GC_EIO; }
        if (r == 0) break;
        off += (size_t)r;
    }
    close(fd);
    *data = buf;
    *len = off;
    return GC_OK;
}

void gc_gen_id(char out[GC_ID_LEN])
{
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    uint8_t rnd[4];
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0 || read(fd, rnd, 4) != 4) {
        uint64_t x = gc_mono_ns() ^ ((uint64_t)getpid() << 32);
        memcpy(rnd, &x, 4);
    }
    if (fd >= 0) close(fd);
    char hex[9];
    gc_hex(rnd, 4, hex);
    snprintf(out, GC_ID_LEN, "%04d%02d%02dT%02d%02d%02dZ-%s", tm.tm_year + 1900,
             tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, hex);
}

/* ids are path components: [A-Za-z0-9._-], no leading dot, <= 63 chars. */
int gc_valid_id(const char *id)
{
    size_t n = strlen(id);
    if (n == 0 || n >= GC_ID_LEN || id[0] == '.') return 0;
    for (size_t i = 0; i < n; i++) {
        char c = id[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return 1;
}

int gc_nprocs(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return (int)n;
}
