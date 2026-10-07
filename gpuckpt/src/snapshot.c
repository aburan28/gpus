/* Repository lifecycle and the four operations over an image:
 * create, restore, verify, gc. All heavy loops are chunk-parallel: workers
 * pull (device, index) pairs from one atomic counter, so the only shared
 * mutable state is that counter, the first-error slot, and the manifest's
 * per-chunk hash slots (each written by exactly one worker).
 *
 * Timing: ns_read / ns_hash / ns_write are summed across worker threads;
 * ns_wall is the elapsed time of the whole operation. Dividing a summed
 * phase by ns_wall gives the average number of threads busy in that phase.
 */
#include "objstore.h"
#include "hostmem.h"
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#define MIN_CHUNK (4096ull)
#define MAX_CHUNK (1ull << 30)

/* ------------------------------------------------------------------ repo */

static int is_s3(const char *path) { return strncmp(path, "s3://", 5) == 0; }

int gc_repo_init(const char *path, uint64_t chunk_size)
{
    if (chunk_size < MIN_CHUNK || chunk_size > MAX_CHUNK || (chunk_size & (chunk_size - 1))) {
        gc_set_error("chunk_size must be a power of two in [%llu, %llu]",
                     (unsigned long long)MIN_CHUNK, (unsigned long long)MAX_CHUNK);
        return GC_EINVAL;
    }
    gc_repo tmp;
    memset(&tmp, 0, sizeof tmp);
    tmp.lock_fd = -1;
    int rc;
    if (is_s3(path)) {
        tmp.kind = GC_REPO_S3;
        snprintf(tmp.path, sizeof tmp.path, "%s", path);
        rc = gc_s3_open(path, &tmp.s3);
        if (rc) return rc;
    } else {
        char p[PATH_MAX];
        const char *subs[] = {"", "/chunks", "/snapshots", "/tmp"};
        for (size_t i = 0; i < 4; i++) {
            snprintf(p, sizeof p, "%s%s", path, subs[i]);
            rc = gc_mkdir_p(p);
            if (rc) return rc;
        }
        if (!realpath(path, tmp.path)) { gc_set_error("repo %s: %s", path, strerror(errno)); return GC_EIO; }
    }
    if (gc_obj_head(&tmp, "config", NULL) == GC_OK) {
        gc_set_error("repository already initialised: %s/config", path);
        gc_s3_close(tmp.s3);
        return GC_EEXIST;
    }
    char cfg[256];
    int n = snprintf(cfg, sizeof cfg, "gpuckpt-repo %d\nchunk_size %llu\nhash sha256\n",
                     GC_FORMAT_VERSION, (unsigned long long)chunk_size);
    rc = gc_obj_put(&tmp, "config", cfg, (size_t)n);
    gc_s3_close(tmp.s3);
    if (rc == GC_EEXIST) gc_set_error("repository already initialised: %s/config", path);
    return rc;
}

int gc_repo_open(const char *path, gc_repo **out)
{
    gc_repo *r = calloc(1, sizeof *r);
    if (!r) return GC_ENOMEM;
    r->lock_fd = -1;
    int rc;
    if (is_s3(path)) {
        r->kind = GC_REPO_S3;
        snprintf(r->path, sizeof r->path, "%s", path);
        rc = gc_s3_open(path, &r->s3);
        if (rc) { free(r); return rc; }
        r->gc_grace_seconds = 3600;
    } else {
        if (!realpath(path, r->path)) {
            gc_set_error("repo %s: %s", path, strerror(errno));
            free(r);
            return GC_ENOENT;
        }
    }
    void *data = NULL; size_t n = 0;
    rc = gc_obj_get(r, "config", &data, &n);
    if (rc) { if (rc == GC_ENOENT) gc_set_error("%s: not a gpuckpt repository (no config)", path); gc_repo_close(r); return rc; }
    char *text = data;
    if (n < 16 || strncmp(text, "gpuckpt-repo 1\n", 15) != 0) { free(data); gc_repo_close(r); gc_set_error("%s: not a gpuckpt repository", path); return GC_EINVAL; }
    char *cs = strstr(text, "chunk_size ");
    if (!cs) { free(data); gc_repo_close(r); return GC_EINVAL; }
    r->chunk_size = strtoull(cs + 11, NULL, 10);
    free(data);
    if (r->chunk_size < MIN_CHUNK || r->chunk_size > MAX_CHUNK) { gc_repo_close(r); return GC_EINVAL; }
    if (r->kind == GC_REPO_LOCAL) {
        char lp[PATH_MAX];
        snprintf(lp, sizeof lp, "%s/lock", r->path);
        r->lock_fd = open(lp, O_RDONLY | O_CREAT, 0644);
        if (r->lock_fd < 0) { gc_set_error("open %s: %s", lp, strerror(errno)); gc_repo_close(r); return GC_EIO; }
    }
    *out = r;
    return GC_OK;
}

void gc_repo_close(gc_repo *r)
{
    if (!r) return;
    if (r->lock_fd >= 0) close(r->lock_fd);
    gc_s3_close(r->s3);
    free(r);
}

const char *gc_repo_kind(const gc_repo *r) { return r->kind == GC_REPO_S3 ? "s3" : "local"; }
void gc_repo_set_gc_grace(gc_repo *r, int seconds) { r->gc_grace_seconds = seconds; }
const char *gc_s3_backend(void) { return gc_s3_impl(); }

uint64_t gc_repo_chunk_size(const gc_repo *r) { return r->chunk_size; }
const char *gc_repo_path(const gc_repo *r) { return r->path; }

int gc_repo_lock(gc_repo *r, int exclusive)
{
    if (r->lock_fd < 0) return GC_OK;   /* s3: no lock; gc relies on its grace period */
    if (flock(r->lock_fd, exclusive ? LOCK_EX : LOCK_SH) != 0) {
        gc_set_error("flock: %s", strerror(errno));
        return GC_EIO;
    }
    return GC_OK;
}
void gc_repo_unlock(gc_repo *r) { if (r->lock_fd >= 0) flock(r->lock_fd, LOCK_UN); }

/* ----------------------------------------------------------- work pool */

/* Modes:
 *   M_HASH_STORE  read each chunk through staging, hash on the CPU, store
 *   M_COPY_NEW    digests already known (GPU): skip chunks the store has,
 *                 write the rest directly (GDS) or through staging
 *   M_CPU_CHECK   read a sample through staging and compare CPU digests
 *                 with the GPU's
 *   M_RESTORE     fetch each chunk from the store into the image
 *   M_VERIFY      re-hash every stored chunk of a manifest
 */
enum { M_HASH_STORE = 0, M_RESTORE, M_VERIFY, M_COPY_NEW, M_CPU_CHECK };
#define MAX_PIPELINE 4

typedef struct work {
    gc_repo *r;
    gc_image *img;
    gc_manifest *m;
    int mode;
    uint64_t total;                     /* flattened chunk count */
    uint64_t prefix[GC_MAX_DEVICES + 1];
    const uint64_t *subset;             /* optional flattened indices to visit */
    uint64_t nsubset;
    atomic_uint_fast64_t next;
    atomic_int err;
    atomic_int next_tid;
    pthread_mutex_t mu;
    gc_stats st;
    uint8_t *arena;                     /* threads * pipeline * chunk_size, or NULL */
    int pipeline;
    int direct;                         /* GDS: dev_to_file / file_to_dev */
    int cpu_verify;                     /* restore: re-hash each chunk on the CPU */
    atomic_int odirect_used;
    atomic_uint_fast64_t first_mismatch;/* M_CPU_CHECK: flattened index + 1, 0 = none */
} work;

typedef struct { int valid; uint64_t k; uint32_t d; uint64_t idx, off; size_t len; } pend;

static void locate(const work *w, uint64_t k, uint32_t *d, uint64_t *idx)
{
    uint32_t dd = 0;
    while (dd + 1 < w->m->device_count && k >= w->prefix[dd + 1]) dd++;
    *d = dd;
    *idx = k - w->prefix[dd];
}

static void set_err(work *w, int rc)
{
    int expected = 0;
    atomic_compare_exchange_strong(&w->err, &expected, rc);
}

static int claim(work *w, uint64_t *k)
{
    uint64_t n = w->subset ? w->nsubset : w->total;
    uint64_t i = atomic_fetch_add(&w->next, 1);
    if (i >= n) return 0;
    *k = w->subset ? w->subset[i] : i;
    return 1;
}

static void geom(const work *w, uint64_t k, pend *p)
{
    locate(w, k, &p->d, &p->idx);
    uint64_t cs = w->m->chunk_size, sz = w->m->dev[p->d].size;
    p->off = p->idx * cs;
    p->len = (size_t)(sz - p->off < cs ? sz - p->off : cs);
    p->k = k;
    p->valid = 1;
}

static int start_read(work *w, void *tctx, int slot, const pend *p, void *buf)
{
    if (w->img->read_start) return w->img->read_start(w->img, tctx, slot, p->d, p->off, buf, p->len);
    return w->img->read(w->img, p->d, p->off, buf, p->len);
}

static int start_write(work *w, void *tctx, int slot, const pend *p, const void *buf)
{
    if (w->img->write_start) return w->img->write_start(w->img, tctx, slot, p->d, p->off, buf, p->len);
    return w->img->write(w->img, p->d, p->off, buf, p->len);
}

static int wait_slot(work *w, void *tctx, int slot)
{
    return w->img && w->img->io_wait ? w->img->io_wait(w->img, tctx, slot) : GC_OK;
}

typedef struct { gc_image *img; void *tctx; uint32_t d; uint64_t off; size_t len; } dfill;
static int dfill_cb(int fd, void *u)
{
    dfill *f = u;
    return f->img->dev_to_file(f->img, f->tctx, f->d, f->off, f->len, fd);
}

/* Claim the next chunk whose bytes must come through a staging buffer.
 * In M_COPY_NEW, chunks the store already holds are counted and skipped,
 * and with direct I/O new chunks go straight from the device to their file.
 * Returns 0 when the work is exhausted or an error was recorded. */
static int next_read(work *w, void *tctx, pend *p, gc_stats *L)
{
    uint64_t k;
    while (!atomic_load(&w->err) && claim(w, &k)) {
        geom(w, k, p);
        if (w->mode != M_COPY_NEW) return 1;
        const uint8_t *h = w->m->dev[p->d].hashes[p->idx];
        L->chunks_total++;
        L->bytes_total += p->len;
        uint64_t t0 = gc_mono_ns();
        if (gc_store_has(w->r, h)) { L->ns_write += gc_mono_ns() - t0; continue; }
        if (w->direct) {
            dfill f = { w->img, tctx, p->d, p->off, p->len };
            int was_new = 0, used = 0;
            int rc = gc_store_put_direct(w->r, h, p->len, w->img->direct_odirect, dfill_cb, &f, &was_new, &used);
            L->ns_write += gc_mono_ns() - t0;
            if (rc) { set_err(w, rc); break; }
            if (used) atomic_store(&w->odirect_used, 1);
            L->bytes_direct += p->len;
            if (was_new) { L->chunks_new++; L->bytes_new += p->len; }
            continue;
        }
        L->ns_write += gc_mono_ns() - t0;
        return 1;
    }
    p->valid = 0;
    return 0;
}

static int process_read(work *w, const pend *p, const uint8_t *buf, gc_stats *L)
{
    uint8_t *slot = w->m->dev[p->d].hashes[p->idx];
    uint64_t t0 = gc_mono_ns(), t1;
    L->bytes_staged += p->len;
    if (w->mode == M_CPU_CHECK) {
        uint8_t h[GC_HASH_LEN];
        gc_sha256(buf, p->len, h);
        L->ns_hash += gc_mono_ns() - t0;
        L->chunks_cpu_checked++;
        if (memcmp(h, slot, GC_HASH_LEN) != 0) {
            uint_fast64_t want = p->k + 1, cur = 0;
            while (!atomic_compare_exchange_weak(&w->first_mismatch, &cur, want))
                if (cur && cur <= want) break;
        }
        return GC_OK;
    }
    if (w->mode == M_HASH_STORE) {
        gc_sha256(buf, p->len, slot);
        t1 = gc_mono_ns();
        L->ns_hash += t1 - t0;
        t0 = t1;
        L->chunks_total++;
        L->bytes_total += p->len;
    }
    int was_new = 0;
    int rc = gc_store_put(w->r, slot, buf, p->len, &was_new);
    L->ns_write += gc_mono_ns() - t0;
    if (rc) return rc;
    if (was_new) { L->chunks_new++; L->bytes_new += p->len; }
    return GC_OK;
}

/* Read modes: keep up to `pipeline` copies in flight per worker, so the
 * next chunk's device-to-host copy runs while this one is hashed/stored. */
static void run_reads(work *w, void *tctx, uint8_t **bufs, int P, gc_stats *L)
{
    pend pq[MAX_PIPELINE];
    memset(pq, 0, sizeof pq);
    int rc;
    for (int s = 0; s < P; s++) {
        if (!next_read(w, tctx, &pq[s], L)) break;
        uint64_t t0 = gc_mono_ns();
        rc = start_read(w, tctx, s, &pq[s], bufs[s]);
        L->ns_read += gc_mono_ns() - t0;
        if (rc) { pq[s].valid = 0; set_err(w, rc); goto drain; }
    }
    for (int s = 0;; s = (s + 1) % P) {
        int found = -1;
        for (int i = 0; i < P; i++) if (pq[(s + i) % P].valid) { found = (s + i) % P; break; }
        if (found < 0) break;
        s = found;
        uint64_t t0 = gc_mono_ns();
        rc = wait_slot(w, tctx, s);
        L->ns_read += gc_mono_ns() - t0;
        pend cur = pq[s];
        pq[s].valid = 0;
        if (rc) { set_err(w, rc); goto drain; }
        rc = process_read(w, &cur, bufs[s], L);
        if (rc) { set_err(w, rc); goto drain; }
        if (next_read(w, tctx, &pq[s], L)) {
            t0 = gc_mono_ns();
            rc = start_read(w, tctx, s, &pq[s], bufs[s]);
            L->ns_read += gc_mono_ns() - t0;
            if (rc) { pq[s].valid = 0; set_err(w, rc); goto drain; }
        }
    }
drain:
    for (int s = 0; s < P; s++) if (pq[s].valid) { wait_slot(w, tctx, s); pq[s].valid = 0; }
}

static void run_restore(work *w, void *tctx, uint8_t **bufs, int P, gc_stats *L)
{
    pend pq[MAX_PIPELINE];
    memset(pq, 0, sizeof pq);
    const uint64_t cs = w->m->chunk_size;
    uint64_t k;
    int s = 0, rc;
    while (!atomic_load(&w->err) && claim(w, &k)) {
        pend p;
        geom(w, k, &p);
        const uint8_t *h = w->m->dev[p.d].hashes[p.idx];
        L->chunks_total++;
        L->bytes_total += p.len;
        uint64_t t0 = gc_mono_ns();
        if (w->direct) {
            int fd = -1, used = 0;
            uint64_t fsz = 0;
            rc = gc_store_open_chunk(w->r, h, w->img->direct_odirect, &fd, &fsz, &used);
            L->ns_read += gc_mono_ns() - t0;
            if (rc == GC_ENOENT) { L->chunks_missing++; set_err(w, GC_ENOENT); break; }
            if (rc) { set_err(w, rc); break; }
            if (fsz != p.len) {
                close(fd);
                L->chunks_corrupt++;
                char hex[GC_HEX_LEN + 1]; gc_hex(h, GC_HASH_LEN, hex);
                gc_set_error("chunk %s: %llu bytes stored, %zu expected", hex, (unsigned long long)fsz, p.len);
                set_err(w, GC_ECORRUPT);
                break;
            }
            t0 = gc_mono_ns();
            rc = w->img->file_to_dev(w->img, tctx, p.d, p.off, p.len, fd);
            L->ns_write += gc_mono_ns() - t0;
            close(fd);
            if (rc) { set_err(w, rc); break; }
            if (used) atomic_store(&w->odirect_used, 1);
            L->bytes_direct += p.len;
            continue;
        }
        if (pq[s].valid) {                       /* slot buffer still being copied to the device */
            rc = wait_slot(w, tctx, s);
            L->ns_write += gc_mono_ns() - t0;
            pq[s].valid = 0;
            if (rc) { set_err(w, rc); break; }
            t0 = gc_mono_ns();
        }
        size_t got = 0;
        rc = gc_store_get(w->r, h, bufs[s], (size_t)cs, &got, 0);
        uint64_t t1 = gc_mono_ns();
        L->ns_read += t1 - t0;
        if (rc == GC_ENOENT) { L->chunks_missing++; set_err(w, GC_ENOENT); break; }
        if (rc) { set_err(w, rc); break; }
        int bad = got != p.len;
        if (!bad && w->cpu_verify) {
            uint8_t hh[GC_HASH_LEN];
            gc_sha256(bufs[s], got, hh);
            bad = memcmp(hh, h, GC_HASH_LEN) != 0;
            L->ns_hash += gc_mono_ns() - t1;
        }
        if (bad) {
            L->chunks_corrupt++;
            char hex[GC_HEX_LEN + 1]; gc_hex(h, GC_HASH_LEN, hex);
            gc_set_error("chunk %s: stored bytes do not match hash", hex);
            set_err(w, GC_ECORRUPT);
            break;
        }
        t0 = gc_mono_ns();
        rc = start_write(w, tctx, s, &p, bufs[s]);
        L->ns_write += gc_mono_ns() - t0;
        if (rc) { set_err(w, rc); break; }
        pq[s] = p;
        L->bytes_staged += p.len;
        s = (s + 1) % P;
    }
    for (int i = 0; i < P; i++)
        if (pq[i].valid) {
            uint64_t t0 = gc_mono_ns();
            rc = wait_slot(w, tctx, i);
            L->ns_write += gc_mono_ns() - t0;
            if (rc) set_err(w, rc);
        }
}

static void run_verify(work *w, uint8_t *buf, gc_stats *L)
{
    const uint64_t cs = w->m->chunk_size;
    uint64_t k;
    while (!atomic_load(&w->err) && claim(w, &k)) {
        pend p;
        geom(w, k, &p);
        const uint8_t *h = w->m->dev[p.d].hashes[p.idx];
        size_t got = 0;
        uint64_t t0 = gc_mono_ns();
        int rc = gc_store_get(w->r, h, buf, (size_t)cs, &got, 0);
        uint64_t t1 = gc_mono_ns();
        L->chunks_total++;
        L->bytes_total += p.len;
        L->ns_read += t1 - t0;
        if (rc == GC_ENOENT) { L->chunks_missing++; continue; }
        if (rc) { set_err(w, rc); break; }
        uint8_t hh[GC_HASH_LEN];
        gc_sha256(buf, got, hh);
        L->ns_hash += gc_mono_ns() - t1;
        if (got != p.len || memcmp(hh, h, GC_HASH_LEN) != 0) L->chunks_corrupt++;
    }
}

static void *worker(void *arg)
{
    work *w = arg;
    gc_image *img = w->img;
    const uint64_t cs = w->m->chunk_size;
    gc_stats L = {0};
    int tid = atomic_fetch_add(&w->next_tid, 1);
    int P = w->pipeline < 1 ? 1 : w->pipeline;
    uint8_t *bufs[MAX_PIPELINE] = {0};
    uint8_t *own = NULL;
    void *tctx = NULL;
    if (w->arena) {
        for (int s = 0; s < P; s++) bufs[s] = w->arena + ((size_t)tid * P + s) * cs;
    } else if (!w->direct) {
        own = malloc((size_t)cs);
        if (!own) { set_err(w, GC_ENOMEM); return NULL; }
        bufs[0] = own;
        P = 1;
    }
    if (img && img->thread_attach) {
        int rc = img->thread_attach(img, &tctx);
        if (rc) { set_err(w, rc); goto out; }
    }
    switch (w->mode) {
    case M_HASH_STORE: case M_COPY_NEW: case M_CPU_CHECK: run_reads(w, tctx, bufs, P, &L); break;
    case M_RESTORE: run_restore(w, tctx, bufs, P, &L); break;
    case M_VERIFY: run_verify(w, bufs[0], &L); break;
    }
    if (img && img->thread_detach) img->thread_detach(img, tctx);
out:
    free(own);
    pthread_mutex_lock(&w->mu);
    gc_stats *S = &w->st;
    S->bytes_total += L.bytes_total; S->chunks_total += L.chunks_total;
    S->chunks_new += L.chunks_new;   S->bytes_new += L.bytes_new;
    S->chunks_missing += L.chunks_missing; S->chunks_corrupt += L.chunks_corrupt;
    S->ns_read += L.ns_read; S->ns_hash += L.ns_hash; S->ns_write += L.ns_write;
    S->bytes_staged += L.bytes_staged; S->bytes_direct += L.bytes_direct;
    S->chunks_cpu_checked += L.chunks_cpu_checked;
    pthread_mutex_unlock(&w->mu);
    return NULL;
}

static int resolve_threads(const gc_repo *r, uint64_t total, int req)
{
    int t = req;
    if (t <= 0) {
        t = gc_nprocs();
        if (t > 8) t = 8;
        if (r->kind == GC_REPO_S3 && t < 16) t = 16;   /* latency-bound */
    }
    if (total > 0 && (uint64_t)t > total) t = (int)total;
    return t < 1 ? 1 : t;
}

size_t gc_staging_bytes(const gc_repo *r, int threads, const gc_io_opts *io)
{
    gc_io_opts dflt;
    if (!io) { gc_io_opts_default(&dflt); io = &dflt; }
    int t = resolve_threads(r, 0, threads);
    int p = io->pipeline < 1 ? 1 : io->pipeline > MAX_PIPELINE ? MAX_PIPELINE : io->pipeline;
    return (size_t)t * (size_t)p * (size_t)r->chunk_size;
}

/* Use the caller's pre-built arena when it is large enough, else build one.
 * Returns the arena to use; *own is set when the caller must destroy it. */
static int get_arena(gc_staging *pre, size_t need, const gc_io_opts *io, gc_image *img, gc_io_report *rep,
                     gc_hostmem *own_hm, gc_hostmem **out, int *own)
{
    if (pre && pre->hm.base && pre->hm.size >= need) {
        snprintf(rep->hostmem, sizeof rep->hostmem, "%s [prepared before the pause]", pre->rep.hostmem);
        snprintf(rep->mlock, sizeof rep->mlock, "%s", pre->rep.mlock);
        snprintf(rep->pin, sizeof rep->pin, "%s", pre->rep.pin);
        *out = &pre->hm;
        *own = 0;
        return GC_OK;
    }
    int rc = gc_hostmem_create(own_hm, need, io, img, rep);
    if (rc) return rc;
    if (pre) {
        size_t l = strlen(rep->hostmem);
        snprintf(rep->hostmem + l, sizeof rep->hostmem - l, " [pre-built arena too small: %zu < %zu bytes]", pre->hm.size, need);
    }
    *out = own_hm;
    *own = 1;
    return GC_OK;
}

/* Runs the pool; stats accumulate into w->st across calls. */
static int run_pool(work *w, int threads)
{
    uint64_t n = w->subset ? w->nsubset : w->total;
    if ((uint64_t)threads > n && n > 0) threads = (int)n;
    if (threads < 1) threads = 1;
    atomic_store(&w->next, 0);
    atomic_store(&w->err, 0);
    atomic_store(&w->next_tid, 0);
    pthread_mutex_init(&w->mu, NULL);
    pthread_t *th = calloc((size_t)threads, sizeof *th);
    if (!th) return GC_ENOMEM;
    int started = 0;
    for (int i = 0; i < threads; i++) {
        if (pthread_create(&th[i], NULL, worker, w) != 0) { set_err(w, GC_EIO); break; }
        started++;
    }
    for (int i = 0; i < started; i++) pthread_join(th[i], NULL);
    free(th);
    pthread_mutex_destroy(&w->mu);
    return atomic_load(&w->err);
}

static void build_prefix(work *w)
{
    w->prefix[0] = 0;
    for (uint32_t d = 0; d < w->m->device_count; d++)
        w->prefix[d + 1] = w->prefix[d] + w->m->dev[d].nchunks;
    w->total = w->prefix[w->m->device_count];
}

static int clamp_pipeline(const gc_io_opts *io, const gc_image *img, int writing, gc_io_report *rep)
{
    int p = io->pipeline < 1 ? 1 : io->pipeline > MAX_PIPELINE ? MAX_PIPELINE : io->pipeline;
    int async = img && (writing ? img->write_start != NULL : img->read_start != NULL);
    if (p > 1 && !async) {
        snprintf(rep->pipeline, sizeof rep->pipeline, "1 (requested %d; %s backend copies are synchronous)", p,
                 img ? img->backend : "this");
        return 1;
    }
    snprintf(rep->pipeline, sizeof rep->pipeline, "%d%s", p, p > 1 ? " slots per worker (copy overlaps hash/store)" : "");
    return p;
}

/* ---------------------------------------------------------------- create */

int gc_snapshot_create(gc_repo *r, gc_image *img, const gc_snapshot_opts *o,
                       gc_stats *st, char id_out[GC_ID_LEN])
{
    if (!img || img->device_count == 0 || img->device_count > GC_MAX_DEVICES || !img->read) return GC_EINVAL;
    gc_io_opts dflt;
    gc_io_opts_default(&dflt);
    const gc_io_opts *io = o && o->io ? o->io : &dflt;
    gc_io_report rep_local;
    gc_io_report *rep = o && o->report ? o->report : &rep_local;
    memset(rep, 0, sizeof *rep);

    gc_manifest *m = calloc(1, sizeof *m);
    if (!m) return GC_ENOMEM;
    int rc;
    if (o && o->id) {
        if (!gc_valid_id(o->id)) { gc_set_error("invalid snapshot id %s", o->id); rc = GC_EINVAL; goto out; }
        snprintf(m->id, sizeof m->id, "%s", o->id);
    } else {
        gc_gen_id(m->id);
    }
    {
        char key[160];
        gc_manifest_key(m->id, key, sizeof key);
        if (gc_obj_head(r, key, NULL) == GC_OK) { gc_set_error("snapshot %s already exists", m->id); rc = GC_EEXIST; goto out; }
    }
    if (o && o->parent) snprintf(m->parent, sizeof m->parent, "%s", o->parent);
    if (o && o->note) snprintf(m->note, sizeof m->note, "%s", o->note);
    m->pid = o ? o->pid : 0;
    m->created_unix_ns = gc_wall_ns();
    m->chunk_size = r->chunk_size;
    snprintf(m->backend, sizeof m->backend, "%s", img->backend[0] ? img->backend : "unknown");
    m->device_count = img->device_count;
    for (uint32_t d = 0; d < m->device_count; d++) {
        rc = gc_manifest_alloc_dev(m, d, img->dev[d].size, m->chunk_size);
        if (rc) goto out;
        snprintf(m->dev[d].uuid, sizeof m->dev[d].uuid, "%s", img->dev[d].uuid);
        m->dev[d].ordinal = img->dev[d].ordinal;
    }
    gc_manifest *parent = NULL;
    if (m->parent[0]) {
        rc = gc_manifest_load(r, m->parent, &parent);
        if (rc) { gc_set_error("parent %s: %s", m->parent, gc_strerror(rc)); goto out; }
    }
    rc = gc_repo_lock(r, 0);
    if (rc) { gc_manifest_free(parent); goto out; }

    work w;
    memset(&w, 0, sizeof w);
    w.r = r; w.img = img; w.m = m;
    build_prefix(&w);
    int threads = resolve_threads(r, w.total, o ? o->threads : 0);
    w.pipeline = clamp_pipeline(io, img, 0, rep);
    const uint64_t cs = m->chunk_size;

    gc_hostmem hm_own, *hm = NULL;
    int own_arena = 0;
    uint64_t t_start = gc_mono_ns();
    rc = get_arena(o ? o->staging : NULL, (size_t)threads * (size_t)w.pipeline * (size_t)cs, io, img, rep,
                   &hm_own, &hm, &own_arena);
    if (rc) { gc_repo_unlock(r); gc_manifest_free(parent); goto out; }
    w.arena = hm->base;
    w.st.hostmem_huge_kb = hm->huge_kb;

    /* device-side hashing */
    int gpu = io->gpu_hash == GC_ON && img->hash_chunks;
    if (io->gpu_hash != GC_ON) snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off");
    else if (!img->hash_chunks) snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off (requested; %s backend did not enable device hashing)", img->backend);
    if (gpu) {
        uint64_t tg = gc_mono_ns();
        for (uint32_t d = 0; d < m->device_count && gpu; d++) {
            if (img->hash_chunks(img, d, cs, m->dev[d].hashes) != GC_OK) {
                snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off (device hashing failed: %s; hashed on the CPU)", gc_last_error());
                gpu = 0;
            }
        }
        w.st.ns_gpu_hash = gc_mono_ns() - tg;
    }
    if (gpu && io->gpu_hash_check > 0 && w.total > 0) {
        uint64_t n = (uint64_t)io->gpu_hash_check < w.total ? (uint64_t)io->gpu_hash_check : w.total;
        uint64_t *sample = malloc(n * sizeof *sample);
        if (!sample) { rc = GC_ENOMEM; goto fail; }
        for (uint64_t i = 0; i < n; i++) sample[i] = n == 1 ? 0 : i * (w.total - 1) / (n - 1);
        w.mode = M_CPU_CHECK;
        w.subset = sample;
        w.nsubset = n;
        rc = run_pool(&w, threads);
        w.subset = NULL;
        free(sample);
        if (rc) goto fail;
        uint64_t bad = atomic_load(&w.first_mismatch);
        if (bad) {
            snprintf(rep->gpu_hash, sizeof rep->gpu_hash,
                     "off (GPU digest of chunk %llu differed from the CPU's in the %llu-chunk cross-check; all chunks re-hashed on the CPU)",
                     (unsigned long long)(bad - 1), (unsigned long long)n);
            gpu = 0;
        }
    }
    if (gpu)
        snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "on: %llu chunks hashed on the device in %.1f ms, %llu cross-checked on the CPU",
                 (unsigned long long)w.total, (double)w.st.ns_gpu_hash / 1e6, (unsigned long long)w.st.chunks_cpu_checked);

    /* direct storage I/O */
    w.direct = 0;
    if (io->gds != GC_ON) snprintf(rep->gds, sizeof rep->gds, "off");
    else if (!gpu) snprintf(rep->gds, sizeof rep->gds, "off (needs device hashing: a chunk must be hashed before it is named, and without a host copy only the GPU can hash it)");
    else if (r->kind != GC_REPO_LOCAL) snprintf(rep->gds, sizeof rep->gds, "off (repo is %s; direct I/O writes local files)", gc_repo_kind(r));
    else if (!img->dev_to_file) snprintf(rep->gds, sizeof rep->gds, "off (requested; %s backend did not enable direct I/O)", img->backend);
    else w.direct = 1;

    w.mode = gpu ? M_COPY_NEW : M_HASH_STORE;
    rc = run_pool(&w, threads);
    if (rc) goto fail;
    if (w.direct)
        snprintf(rep->gds, sizeof rep->gds, "on: %llu bytes written device->file, files opened %s",
                 (unsigned long long)w.st.bytes_direct, atomic_load(&w.odirect_used) ? "O_DIRECT" : "buffered (filesystem has no O_DIRECT)");

    if (parent)
        for (uint32_t d = 0; d < m->device_count && d < parent->device_count; d++)
            for (uint64_t i = 0; i < m->dev[d].nchunks && i < parent->dev[d].nchunks; i++)
                if (!memcmp(m->dev[d].hashes[i], parent->dev[d].hashes[i], GC_HASH_LEN)) w.st.chunks_same_as_parent++;
    w.st.ns_wall = gc_mono_ns() - t_start;
    w.st.threads = threads;
    m->stats = w.st;
    rc = gc_manifest_save(r, m);
fail:
    if (own_arena) gc_hostmem_destroy(hm);
    gc_repo_unlock(r);
    gc_manifest_free(parent);
    if (st) *st = w.st;
    if (rc == GC_OK && id_out) snprintf(id_out, GC_ID_LEN, "%s", m->id);
out:
    gc_manifest_free(m);
    return rc;
}

/* --------------------------------------------------------------- restore */

int gc_snapshot_restore(gc_repo *r, const char *id, gc_image *img, const gc_restore_opts *o, gc_stats *st)
{
    if (!img || !img->write) return GC_EINVAL;
    gc_io_opts dflt;
    gc_io_opts_default(&dflt);
    const gc_io_opts *io = o && o->io ? o->io : &dflt;
    gc_io_report rep_local;
    gc_io_report *rep = o && o->report ? o->report : &rep_local;
    memset(rep, 0, sizeof *rep);
    gc_manifest *m = NULL;
    int rc = gc_manifest_load(r, id, &m);
    if (rc) return rc;
    if (img->device_count != m->device_count) {
        gc_set_error("image has %u devices, snapshot %s has %u", img->device_count, id, m->device_count);
        gc_manifest_free(m); return GC_EMISMATCH;
    }
    for (uint32_t d = 0; d < m->device_count; d++)
        if (img->dev[d].size != m->dev[d].size) {
            gc_set_error("device %u: image size %llu, snapshot size %llu", d,
                         (unsigned long long)img->dev[d].size, (unsigned long long)m->dev[d].size);
            gc_manifest_free(m); return GC_EMISMATCH;
        }
    rc = gc_repo_lock(r, 0);
    if (rc) { gc_manifest_free(m); return rc; }
    work w;
    memset(&w, 0, sizeof w);
    w.r = r; w.img = img; w.m = m; w.mode = M_RESTORE;
    build_prefix(&w);
    int threads = resolve_threads(r, w.total, o ? o->threads : 0);
    const uint64_t cs = m->chunk_size;

    int gpu_verify = io->gpu_hash == GC_ON && img->hash_chunks;
    if (io->gpu_hash != GC_ON) snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off (chunks verified on the CPU)");
    else if (!img->hash_chunks) snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off (requested; %s backend did not enable device hashing; chunks verified on the CPU)", img->backend);
    if (io->gds != GC_ON) snprintf(rep->gds, sizeof rep->gds, "off");
    else if (!gpu_verify) snprintf(rep->gds, sizeof rep->gds, "off (needs device hashing to verify chunks that never pass through host memory)");
    else if (r->kind != GC_REPO_LOCAL) snprintf(rep->gds, sizeof rep->gds, "off (repo is %s; direct I/O reads local files)", gc_repo_kind(r));
    else if (!img->file_to_dev) snprintf(rep->gds, sizeof rep->gds, "off (requested; %s backend did not enable direct I/O)", img->backend);
    else w.direct = 1;
    w.cpu_verify = !gpu_verify;

    gc_hostmem hm_own, *hm = NULL;
    int own_arena = 0;
    uint64_t t_start = gc_mono_ns();
    if (!w.direct) {
        w.pipeline = clamp_pipeline(io, img, 1, rep);
        rc = get_arena(o ? o->staging : NULL, (size_t)threads * (size_t)w.pipeline * (size_t)cs, io, img, rep,
                       &hm_own, &hm, &own_arena);
        if (rc) goto done;
        w.arena = hm->base;
        w.st.hostmem_huge_kb = hm->huge_kb;
    } else {
        snprintf(rep->hostmem, sizeof rep->hostmem, "not used (direct I/O)");
        snprintf(rep->pin, sizeof rep->pin, "not used (direct I/O)");
        snprintf(rep->mlock, sizeof rep->mlock, "not used (direct I/O)");
        snprintf(rep->pipeline, sizeof rep->pipeline, "n/a (direct I/O)");
    }
    rc = run_pool(&w, threads);
    if (rc == GC_OK && w.direct)
        snprintf(rep->gds, sizeof rep->gds, "on: %llu bytes read file->device, files opened %s",
                 (unsigned long long)w.st.bytes_direct, atomic_load(&w.odirect_used) ? "O_DIRECT" : "buffered (filesystem has no O_DIRECT)");
    if (rc == GC_OK && gpu_verify) {
        /* Hash the restored image in place. A chunk the device flags is
         * copied back and re-hashed on the CPU before it is called corrupt,
         * so a faulty kernel cannot fail a correct restore; only a CPU-
         * confirmed mismatch fails it. */
        uint64_t tg = gc_mono_ns(), cpu_rechecked = 0, cpu_cleared = 0;
        uint8_t *cbuf = NULL;
        for (uint32_t d = 0; d < m->device_count && rc == GC_OK; d++) {
            uint64_t n = m->dev[d].nchunks;
            uint8_t (*got)[GC_HASH_LEN] = calloc(n ? n : 1, GC_HASH_LEN);
            if (!got) { rc = GC_ENOMEM; break; }
            int all_cpu = 0;
            if (img->hash_chunks(img, d, cs, got) != GC_OK) {
                /* Device hashing failed: verify every chunk on the CPU by
                 * reading it back, or fail if the image cannot be read. */
                if (!img->read) {
                    gc_set_error("restored image could not be verified on the device: %s", gc_last_error());
                    rc = GC_EBACKEND;
                    free(got);
                    break;
                }
                snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "off (device verification failed: %s; verified on the CPU by reading back)",
                         gc_last_error());
                all_cpu = 1;
            }
            for (uint64_t i = 0; i < n && rc == GC_OK; i++) {
                if (!all_cpu && !memcmp(got[i], m->dev[d].hashes[i], GC_HASH_LEN)) continue;
                uint64_t off = i * cs, len = m->dev[d].size - off < cs ? m->dev[d].size - off : cs;
                int confirmed = 1;
                if (img->read) {
                    if (!cbuf && !(cbuf = malloc((size_t)cs))) { rc = GC_ENOMEM; break; }
                    if (img->read(img, d, off, cbuf, (size_t)len) == GC_OK) {
                        uint8_t h[GC_HASH_LEN];
                        gc_sha256(cbuf, (size_t)len, h);
                        confirmed = memcmp(h, m->dev[d].hashes[i], GC_HASH_LEN) != 0;
                        cpu_rechecked++;
                        w.st.bytes_staged += len;
                    }
                }
                if (!confirmed) { if (!all_cpu) cpu_cleared++; continue; }
                if (!w.st.chunks_corrupt)
                    gc_set_error("device %u chunk %llu: restored bytes do not match the manifest", d, (unsigned long long)i);
                w.st.chunks_corrupt++;
            }
            free(got);
        }
        free(cbuf);
        if (rc == GC_OK && w.st.chunks_corrupt) rc = GC_ECORRUPT;
        w.st.ns_gpu_hash = gc_mono_ns() - tg;
        w.st.chunks_cpu_checked += cpu_rechecked;
        if (rc == GC_OK && !strncmp(rep->gpu_hash, "off (device verification failed", 31))
            ; /* keep the fallback report */
        else if (rc == GC_OK && cpu_cleared)
            snprintf(rep->gpu_hash, sizeof rep->gpu_hash,
                     "on, unreliable: the device flagged %llu chunk(s) that re-hashed correctly on the CPU; %llu restored chunks verified",
                     (unsigned long long)cpu_cleared, (unsigned long long)w.total);
        else if (rc == GC_OK)
            snprintf(rep->gpu_hash, sizeof rep->gpu_hash, "on: %llu restored chunks verified on the device in %.1f ms",
                     (unsigned long long)w.total, (double)w.st.ns_gpu_hash / 1e6);
    }
done:
    if (own_arena) gc_hostmem_destroy(hm);
    gc_repo_unlock(r);
    w.st.ns_wall = gc_mono_ns() - t_start;
    w.st.threads = threads;
    if (st) *st = w.st;
    gc_manifest_free(m);
    return rc;
}

/* ---------------------------------------------------------------- verify */

int gc_snapshot_verify(gc_repo *r, const char *id, gc_stats *st)
{
    gc_manifest *m = NULL;
    int rc = gc_manifest_load(r, id, &m);
    if (rc) return rc;
    rc = gc_repo_lock(r, 0);
    if (rc) { gc_manifest_free(m); return rc; }
    work w;
    memset(&w, 0, sizeof w);
    w.r = r; w.img = NULL; w.m = m; w.mode = M_VERIFY; w.pipeline = 1;
    build_prefix(&w);
    uint64_t t0 = gc_mono_ns();
    int threads = resolve_threads(r, w.total, 0);
    rc = run_pool(&w, threads);
    w.st.ns_wall = gc_mono_ns() - t0;
    w.st.threads = threads;
    gc_repo_unlock(r);
    if (st) *st = w.st;
    gc_manifest_free(m);
    if (rc) return rc;
    if (w.st.chunks_corrupt) return GC_ECORRUPT;
    if (w.st.chunks_missing) return GC_ENOENT;
    return GC_OK;
}

/* ---------------------------------------------------------- forget / gc */

int gc_snapshot_forget(gc_repo *r, const char *id)
{
    char key[160];
    int rc = gc_manifest_key(id, key, sizeof key);
    if (rc) return rc;
    rc = gc_repo_lock(r, 0);
    if (rc) return rc;
    rc = gc_obj_delete(r, key, NULL);
    gc_repo_unlock(r);
    return rc;
}

typedef struct { char (*ids)[GC_ID_LEN]; size_t n, cap; } idlist;

static int collect_cb(const gc_obj_info *info, void *u)
{
    idlist *l = u;
    const char *k = info->key;
    if (strncmp(k, "snapshots/", 10) != 0) return GC_OK;
    k += 10;
    size_t n = strlen(k);
    if (n <= 9 || strcmp(k + n - 9, ".manifest") != 0 || strchr(k, '/')) return GC_OK;
    if (n - 9 >= GC_ID_LEN) return GC_OK;
    if (l->n == l->cap) {
        size_t nc = l->cap ? l->cap * 2 : 64;
        void *np = realloc(l->ids, nc * GC_ID_LEN);
        if (!np) return GC_ENOMEM;
        l->ids = np; l->cap = nc;
    }
    memcpy(l->ids[l->n], k, n - 9);
    l->ids[l->n][n - 9] = 0;
    l->n++;
    return GC_OK;
}

static int collect_ids(gc_repo *r, idlist *l)
{
    return gc_obj_list(r, "snapshots/", collect_cb, l);
}

static int cmp_id(const void *a, const void *b) { return strcmp(a, b); }
static int cmp_hash(const void *a, const void *b) { return memcmp(a, b, GC_HASH_LEN); }

int gc_repo_list(gc_repo *r, int (*cb)(const char *id, void *u), void *u)
{
    idlist l = {0};
    int rc = collect_ids(r, &l);
    if (rc) { free(l.ids); return rc; }
    qsort(l.ids, l.n, GC_ID_LEN, cmp_id);
    for (size_t i = 0; i < l.n && rc == GC_OK; i++) rc = cb(l.ids[i], u);
    free(l.ids);
    return rc;
}

typedef struct {
    gc_repo *r;
    uint8_t (*live)[GC_HASH_LEN];
    size_t nlive;
    uint64_t deleted, bytes, skipped_young;
    int64_t now;
} gcwalk;

static int gc_cb(const gc_obj_info *info, void *u)
{
    gcwalk *g = u;
    uint8_t hash[GC_HASH_LEN];
    if (gc_store_key_parse(info->key, hash) != GC_OK) return GC_OK;   /* foreign object: leave it */
    if (g->nlive && bsearch(hash, g->live, g->nlive, GC_HASH_LEN, cmp_hash)) return GC_OK;
    if (g->r->gc_grace_seconds > 0 && info->mtime >= 0 && g->now - info->mtime < g->r->gc_grace_seconds) {
        g->skipped_young++;
        return GC_OK;
    }
    int rc = gc_obj_delete(g->r, info->key, NULL);
    if (rc == GC_OK) { g->deleted++; g->bytes += info->size; }
    else if (rc != GC_ENOENT) return rc;
    return GC_OK;
}

/* Delete every chunk not referenced by any manifest. Takes the exclusive
 * repo lock where one exists (local), so no snapshot or restore is in
 * flight; on S3 there is no lock, so unreferenced chunks younger than the
 * grace period are left alone because a concurrent snapshot may have just
 * deduplicated against them. If any manifest fails to load, nothing is
 * deleted: an unknown live set is not an empty one. */
int gc_repo_gc(gc_repo *r, uint64_t *chunks_deleted, uint64_t *bytes_freed)
{
    int rc = gc_repo_lock(r, 1);
    if (rc) return rc;
    idlist l = {0};
    rc = collect_ids(r, &l);
    if (rc) { gc_repo_unlock(r); free(l.ids); return rc; }
    uint8_t (*live)[GC_HASH_LEN] = NULL;
    size_t nlive = 0, cap = 0;
    for (size_t i = 0; i < l.n; i++) {
        gc_manifest *m = NULL;
        rc = gc_manifest_load(r, l.ids[i], &m);
        if (rc) { gc_set_error("gc aborted: manifest %s unreadable (%s)", l.ids[i], gc_strerror(rc)); goto out; }
        for (uint32_t d = 0; d < m->device_count; d++) {
            uint64_t n = m->dev[d].nchunks;
            if (nlive + n > cap) {
                size_t nc = cap ? cap : 1024;
                while (nc < nlive + n) nc *= 2;
                void *np = realloc(live, nc * GC_HASH_LEN);
                if (!np) { gc_manifest_free(m); rc = GC_ENOMEM; goto out; }
                live = np; cap = nc;
            }
            memcpy(live[nlive], m->dev[d].hashes, (size_t)n * GC_HASH_LEN);
            nlive += n;
        }
        gc_manifest_free(m);
    }
    qsort(live, nlive, GC_HASH_LEN, cmp_hash);
    gcwalk g = { .r = r, .live = live, .nlive = nlive, .now = (int64_t)time(NULL) };
    rc = gc_obj_list(r, "chunks/", gc_cb, &g);
    if (chunks_deleted) *chunks_deleted = g.deleted;
    if (bytes_freed) *bytes_freed = g.bytes;
    /* stale staging files from crashed writers (local only) */
    if (r->kind == GC_REPO_LOCAL) {
        char tp[PATH_MAX];
        snprintf(tp, sizeof tp, "%s/tmp", r->path);
        DIR *d = opendir(tp);
        if (d) {
            struct dirent *e;
            while ((e = readdir(d))) {
                if (strncmp(e->d_name, ".tmp.", 5) != 0) continue;
                char fp[PATH_MAX];
                snprintf(fp, sizeof fp, "%s/%s", tp, e->d_name);
                unlink(fp);
            }
            closedir(d);
        }
    }
out:
    free(live);
    free(l.ids);
    gc_repo_unlock(r);
    return rc;
}
