/* Staging memory for copies between the image and the store.
 *
 * One arena is allocated per operation and sliced into per-worker slots, so
 * page-locking and huge-page setup are paid once, before the copy phase,
 * instead of per chunk. Options, each with a reported fallback:
 *
 *   malloc     page-aligned heap memory
 *   thp        anonymous mmap, 2 MiB aligned, madvise(MADV_HUGEPAGE)
 *   hugetlb    MAP_HUGETLB 2 MiB pages (needs vm.nr_hugepages); falls back to thp
 *   hugetlb1g  MAP_HUGETLB 1 GiB pages; falls back to hugetlb, then thp
 *   mlock      mlock(2) the arena (no swap-out, no faults mid-copy)
 *   pin        register:  backend page-locks our arena for DMA (keeps huge pages)
 *              alloc:     backend allocates pinned memory itself (no huge-page control)
 *
 * The arena is pre-faulted, so first-touch page faults happen here and not
 * inside the checkpoint pause. Huge-page coverage is measured from
 * /proc/self/smaps and reported, because THP is a request, not a guarantee.
 */
#include "hostmem.h"
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

#define MB2 (2ull << 20)
#define GB1 (1ull << 30)

void gc_io_opts_default(gc_io_opts *o)
{
    memset(o, 0, sizeof *o);
    o->hostmem = GC_HOSTMEM_THP;
    o->pin = GC_PIN_REGISTER;
    o->pipeline = 2;
    o->gpu_hash = GC_OFF;
    o->gpu_hash_check = 8;
    o->gds = GC_OFF;
}

static size_t round_up(size_t n, size_t a) { return (n + a - 1) / a * a; }

const char *gc_hostmem_name(int kind)
{
    switch (kind) {
    case GC_HOSTMEM_MALLOC: return "malloc";
    case GC_HOSTMEM_THP: return "thp";
    case GC_HOSTMEM_HUGETLB: return "hugetlb";
    case GC_HOSTMEM_HUGETLB_1G: return "hugetlb1g";
    default: return "?";
    }
}

/* kB of [base, base+len) backed by huge pages, from /proc/self/smaps. */
uint64_t gc_hostmem_huge_kb(const void *base, size_t len)
{
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return 0;
    uintptr_t lo = (uintptr_t)base, hi = lo + len;
    char line[512];
    int in = 0;
    uint64_t kb = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long a, b;
        if (sscanf(line, "%lx-%lx ", &a, &b) == 2 && strchr(line, '-') < strchr(line, ' ')) {
            in = (uintptr_t)a < hi && (uintptr_t)b > lo;
            continue;
        }
        unsigned long v;
        if (in && (sscanf(line, "AnonHugePages: %lu kB", &v) == 1 ||
                   sscanf(line, "Private_Hugetlb: %lu kB", &v) == 1 ||
                   sscanf(line, "Shared_Hugetlb: %lu kB", &v) == 1))
            kb += v;
    }
    fclose(f);
    return kb;
}

static int thp_mode(char *out, size_t n)
{
    FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (!f) { snprintf(out, n, "unknown"); return -1; }
    char buf[128] = "";
    if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
    fclose(f);
    char *a = strchr(buf, '['), *b = a ? strchr(a, ']') : NULL;
    if (!a || !b) { snprintf(out, n, "unknown"); return -1; }
    snprintf(out, n, "%.*s", (int)(b - a - 1), a + 1);
    return 0;
}

static void *map_thp(size_t size, size_t *map_len, void **map_base, char *why, size_t whylen)
{
    size_t len = round_up(size, MB2);
    size_t over = len + MB2;
    void *p = mmap(NULL, over, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { snprintf(why, whylen, "mmap: %s", strerror(errno)); return NULL; }
    uintptr_t start = ((uintptr_t)p + MB2 - 1) & ~(uintptr_t)(MB2 - 1);
    size_t head = start - (uintptr_t)p, tail = over - head - len;
    if (head) munmap(p, head);
    if (tail) munmap((void *)(start + len), tail);
    if (madvise((void *)start, len, MADV_HUGEPAGE) != 0)
        snprintf(why, whylen, "madvise(MADV_HUGEPAGE): %s", strerror(errno));
    *map_len = len;
    *map_base = (void *)start;
    return (void *)start;
}

static void *map_hugetlb(size_t size, int gb, size_t *map_len, void **map_base, char *why, size_t whylen)
{
    size_t page = gb ? GB1 : MB2;
    size_t len = round_up(size, page);
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE | (gb ? MAP_HUGE_1GB : MAP_HUGE_2MB);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        snprintf(why, whylen, "MAP_HUGETLB %s: %s (reserve pages: %s)", gb ? "1 GiB" : "2 MiB", strerror(errno),
                 gb ? "hugepagesz=1G hugepages=N on the kernel command line"
                    : "sysctl vm.nr_hugepages=N");
        return NULL;
    }
    *map_len = len;
    *map_base = p;
    return p;
}

int gc_hostmem_create(gc_hostmem *hm, size_t size, const gc_io_opts *io, gc_image *img, gc_io_report *rep)
{
    memset(hm, 0, sizeof *hm);
    hm->size = size;
    gc_io_report dummy;
    if (!rep) rep = &dummy;
    if (size == 0) size = 4096;
    char why[256] = "";

    /* 1. backend-allocated pinned memory */
    if (io->pin == GC_PIN_ALLOC) {
        if (img && img->host_alloc_pinned) {
            hm->base = img->host_alloc_pinned(img, size, why, sizeof why);
            if (hm->base) {
                hm->owner = img;
                hm->kind = -1;
                memset(hm->base, 0, size);
                snprintf(rep->hostmem, sizeof rep->hostmem, "backend-allocated (requested %s ignored with --pin alloc)",
                         gc_hostmem_name(io->hostmem));
                snprintf(rep->pin, sizeof rep->pin, "alloc: backend-allocated pinned memory, %zu bytes", size);
                snprintf(rep->mlock, sizeof rep->mlock, "n/a (pinned by backend)");
                hm->pinned = 1;
                return GC_OK;
            }
        } else {
            snprintf(why, sizeof why, "backend cannot allocate pinned memory");
        }
        snprintf(rep->pin, sizeof rep->pin, "none (alloc failed: %s)", why);
        why[0] = 0;
    }

    /* 2. our own mapping */
    int kind = io->hostmem;
    char fallback[512] = "";
    if (kind == GC_HOSTMEM_HUGETLB_1G) {
        hm->base = map_hugetlb(size, 1, &hm->map_len, &hm->map_base, why, sizeof why);
        if (!hm->base) { snprintf(fallback, sizeof fallback, "%s; ", why); kind = GC_HOSTMEM_HUGETLB; }
    }
    if (!hm->base && kind == GC_HOSTMEM_HUGETLB) {
        hm->base = map_hugetlb(size, 0, &hm->map_len, &hm->map_base, why, sizeof why);
        if (!hm->base) { size_t l = strlen(fallback); snprintf(fallback + l, sizeof fallback - l, "%s; ", why); kind = GC_HOSTMEM_THP; }
    }
    char thp_note[160] = "";
    if (!hm->base && kind == GC_HOSTMEM_THP) {
        why[0] = 0;
        hm->base = map_thp(size, &hm->map_len, &hm->map_base, why, sizeof why);
        char mode[32];
        thp_mode(mode, sizeof mode);
        if (!strcmp(mode, "never")) snprintf(thp_note, sizeof thp_note, "; THP disabled system-wide (enabled=never)");
        else if (why[0]) snprintf(thp_note, sizeof thp_note, "; %s", why);
        if (!hm->base) kind = GC_HOSTMEM_MALLOC;
    }
    if (!hm->base) {
        kind = GC_HOSTMEM_MALLOC;
        if (posix_memalign(&hm->base, 4096, size) != 0) return GC_ENOMEM;
        hm->malloced = 1;
    }
    hm->kind = kind;
    size_t fl = strlen(fallback);
    while (fl && (fallback[fl - 1] == ' ' || fallback[fl - 1] == ';')) fallback[--fl] = 0;
    /* pre-fault now, outside the checkpoint pause */
    memset(hm->base, 0, size);
    hm->huge_kb = gc_hostmem_huge_kb(hm->base, size);
    snprintf(rep->hostmem, sizeof rep->hostmem, "%s%s: %llu of %zu kB in huge pages%s%s", gc_hostmem_name(kind),
             io->hostmem != kind ? " (fallback)" : "", (unsigned long long)hm->huge_kb, size / 1024,
             fallback[0] ? " after " : "", fallback);
    if (thp_note[0]) {
        size_t l = strlen(rep->hostmem);
        snprintf(rep->hostmem + l, sizeof rep->hostmem - l, "%s", thp_note);
    }

    /* 3. mlock */
    if (io->mlock) {
        if (mlock(hm->base, size) == 0) { hm->locked = 1; snprintf(rep->mlock, sizeof rep->mlock, "locked %zu kB", size / 1024); }
        else snprintf(rep->mlock, sizeof rep->mlock, "failed: %s (raise RLIMIT_MEMLOCK / ulimit -l, or run with CAP_IPC_LOCK)", strerror(errno));
    } else {
        snprintf(rep->mlock, sizeof rep->mlock, "off");
    }

    /* 4. DMA registration */
    if (io->pin == GC_PIN_REGISTER) {
        if (img && img->host_register) {
            why[0] = 0;
            if (img->host_register(img, hm->base, size, why, sizeof why) == 0) {
                hm->pinned = 1;
                hm->owner = img;
                snprintf(rep->pin, sizeof rep->pin, "register: %zu kB page-locked for DMA", size / 1024);
            } else {
                snprintf(rep->pin, sizeof rep->pin, "none (register failed: %s; copies use pageable memory)", why);
            }
        } else {
            snprintf(rep->pin, sizeof rep->pin, "n/a (%s backend has no DMA engine)", img ? img->backend : "no");
        }
    } else if (io->pin == GC_PIN_NONE) {
        snprintf(rep->pin, sizeof rep->pin, "off");
    }
    return GC_OK;
}

void gc_hostmem_destroy(gc_hostmem *hm)
{
    if (!hm->base) return;
    if (hm->kind == -1) {
        hm->owner->host_free_pinned(hm->owner, hm->base);
    } else {
        if (hm->pinned && hm->owner && hm->owner->host_unregister) hm->owner->host_unregister(hm->owner, hm->base);
        if (hm->locked) munlock(hm->base, hm->size ? hm->size : 4096);
        if (hm->malloced) free(hm->base);
        else munmap(hm->map_base, hm->map_len);
    }
    memset(hm, 0, sizeof *hm);
}

/* ----------------------------------------------------------- gc_staging */

int gc_staging_create(gc_staging **out, size_t bytes, const gc_io_opts *io, gc_image *pin_with,
                      gc_io_report *rep, uint64_t *ns_setup)
{
    gc_io_opts dflt;
    if (!io) { gc_io_opts_default(&dflt); io = &dflt; }
    gc_staging *s = calloc(1, sizeof *s);
    if (!s) return GC_ENOMEM;
    uint64_t t0 = gc_mono_ns();
    int rc = gc_hostmem_create(&s->hm, bytes, io, pin_with, &s->rep);
    if (ns_setup) *ns_setup = gc_mono_ns() - t0;
    if (rc) { free(s); return rc; }
    if (rep) *rep = s->rep;
    *out = s;
    return GC_OK;
}

void gc_staging_destroy(gc_staging *s)
{
    if (!s) return;
    gc_hostmem_destroy(&s->hm);
    free(s);
}
