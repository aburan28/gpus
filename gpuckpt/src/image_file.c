/* Host-file image backend: device d is the file paths[d]. Used by the test
 * suite, by the CPU-only benchmark, and for any memory image that already
 * sits in a file (for instance a host-storage checkpoint dumped by other
 * tooling). pread/pwrite keep it thread-safe without shared offsets. */
#include "internal.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { int fds[GC_MAX_DEVICES]; } file_priv;

static int f_read(gc_image *img, uint32_t d, uint64_t off, void *dst, size_t len)
{
    file_priv *p = img->priv;
    uint8_t *b = dst;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(p->fds[d], b + done, len - done, (off_t)(off + done));
        if (r < 0) { if (errno == EINTR) continue; gc_set_error("pread: %s", strerror(errno)); return GC_EIO; }
        if (r == 0) { gc_set_error("pread: short read at %llu", (unsigned long long)(off + done)); return GC_EIO; }
        done += (size_t)r;
    }
    return GC_OK;
}

static int f_write(gc_image *img, uint32_t d, uint64_t off, const void *src, size_t len)
{
    file_priv *p = img->priv;
    const uint8_t *b = src;
    size_t done = 0;
    while (done < len) {
        ssize_t w = pwrite(p->fds[d], b + done, len - done, (off_t)(off + done));
        if (w < 0) { if (errno == EINTR) continue; gc_set_error("pwrite: %s", strerror(errno)); return GC_EIO; }
        done += (size_t)w;
    }
    return GC_OK;
}

int gc_image_file_open(gc_image *img, const char *mode, uint32_t n,
                       const char *const *paths, const uint64_t *sizes)
{
    if (n == 0 || n > GC_MAX_DEVICES) return GC_EINVAL;
    memset(img, 0, sizeof *img);
    file_priv *p = calloc(1, sizeof *p);
    if (!p) return GC_ENOMEM;
    for (uint32_t i = 0; i < GC_MAX_DEVICES; i++) p->fds[i] = -1;
    img->priv = p;
    snprintf(img->backend, sizeof img->backend, "file");
    int writing = mode[0] == 'w';
    for (uint32_t d = 0; d < n; d++) {
        int fd = writing ? open(paths[d], O_RDWR | O_CREAT | O_TRUNC, 0644) : open(paths[d], O_RDONLY);
        if (fd < 0) {
            gc_set_error("open %s: %s", paths[d], strerror(errno));
            gc_image_file_close(img);
            return GC_EIO;
        }
        p->fds[d] = fd;
        if (writing) {
            if (ftruncate(fd, (off_t)sizes[d]) != 0) { gc_set_error("ftruncate: %s", strerror(errno)); gc_image_file_close(img); return GC_EIO; }
            img->dev[d].size = sizes[d];
        } else {
            struct stat st;
            if (fstat(fd, &st) != 0) { gc_image_file_close(img); return GC_EIO; }
            img->dev[d].size = (uint64_t)st.st_size;
        }
        snprintf(img->dev[d].uuid, sizeof img->dev[d].uuid, "-");
        img->dev[d].ordinal = (int)d;
    }
    img->device_count = n;
    img->read = f_read;
    img->write = f_write;
    return GC_OK;
}

void gc_image_file_close(gc_image *img)
{
    file_priv *p = img->priv;
    if (!p) return;
    for (uint32_t d = 0; d < GC_MAX_DEVICES; d++)
        if (p->fds[d] >= 0) { fsync(p->fds[d]); close(p->fds[d]); }
    free(p);
    img->priv = NULL;
}
