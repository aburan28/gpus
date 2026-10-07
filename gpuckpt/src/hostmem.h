#ifndef GC_HOSTMEM_H
#define GC_HOSTMEM_H
#include "internal.h"

typedef struct gc_hostmem {
    void    *base;
    size_t   size;
    void    *map_base;
    size_t   map_len;
    int      kind;       /* GC_HOSTMEM_* obtained, -1 = backend-allocated */
    int      malloced;
    int      locked;
    int      pinned;
    uint64_t huge_kb;
    gc_image *owner;     /* backend that pinned / allocated it */
} gc_hostmem;

struct gc_staging {
    gc_hostmem   hm;
    gc_io_report rep;
};

int      gc_hostmem_create(gc_hostmem *hm, size_t size, const gc_io_opts *io, gc_image *img, gc_io_report *rep);
void     gc_hostmem_destroy(gc_hostmem *hm);
uint64_t gc_hostmem_huge_kb(const void *base, size_t len);
const char *gc_hostmem_name(int kind);
#endif
