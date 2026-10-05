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
#include "internal.h"
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

int gc_repo_init(const char *path, uint64_t chunk_size)
{
    if (chunk_size < MIN_CHUNK || chunk_size > MAX_CHUNK || (chunk_size & (chunk_size - 1))) {
        gc_set_error("chunk_size must be a power of two in [%llu, %llu]",
                     (unsigned long long)MIN_CHUNK, (unsigned long long)MAX_CHUNK);
        return GC_EINVAL;
    }
    char p[PATH_MAX];
    const char *subs[] = {"", "/chunks", "/snapshots", "/tmp"};
    for (size_t i = 0; i < 4; i++) {
        snprintf(p, sizeof p, "%s%s", path, subs[i]);
        int rc = gc_mkdir_p(p);
        if (rc) return rc;
    }
    char cfg[256];
    int n = snprintf(cfg, sizeof cfg, "gpuckpt-repo %d\nchunk_size %llu\nhash sha256\n",
                     GC_FORMAT_VERSION, (unsigned long long)chunk_size);
    char cfgp[PATH_MAX], tmp[PATH_MAX];
    snprintf(cfgp, sizeof cfgp, "%s/config", path);
    snprintf(tmp, sizeof tmp, "%s/tmp", path);
    int rc = gc_write_file_atomic(cfgp, tmp, cfg, (size_t)n, 0);
    if (rc == GC_EEXIST) gc_set_error("repository already initialised: %s", cfgp);
    return rc;
}

int gc_repo_open(const char *path, gc_repo **out)
{
    gc_repo *r = calloc(1, sizeof *r);
    if (!r) return GC_ENOMEM;
    r->lock_fd = -1;
    if (!realpath(path, r->path)) {
        gc_set_error("repo %s: %s", path, strerror(errno));
        free(r);
        return GC_ENOENT;
    }
    char cfgp[PATH_MAX];
    snprintf(cfgp, sizeof cfgp, "%s/config", r->path);
    void *data = NULL; size_t n = 0;
    int rc = gc_read_file(cfgp, &data, &n);
    if (rc) { free(r); return rc; }
    char *text = data;
    if (n < 16 || strncmp(text, "gpuckpt-repo 1\n", 15) != 0) { free(data); free(r); gc_set_error("%s: not a gpuckpt repository", path); return GC_EINVAL; }
    char *cs = strstr(text, "chunk_size ");
    if (!cs) { free(data); free(r); return GC_EINVAL; }
    r->chunk_size = strtoull(cs + 11, NULL, 10);
    free(data);
    if (r->chunk_size < MIN_CHUNK || r->chunk_size > MAX_CHUNK) { free(r); return GC_EINVAL; }
    char lp[PATH_MAX];
    snprintf(lp, sizeof lp, "%s/lock", r->path);
    r->lock_fd = open(lp, O_RDONLY | O_CREAT, 0644);
    if (r->lock_fd < 0) { gc_set_error("open %s: %s", lp, strerror(errno)); free(r); return GC_EIO; }
    *out = r;
    return GC_OK;
}

void gc_repo_close(gc_repo *r)
{
    if (!r) return;
    if (r->lock_fd >= 0) close(r->lock_fd);
    free(r);
}

uint64_t gc_repo_chunk_size(const gc_repo *r) { return r->chunk_size; }
const char *gc_repo_path(const gc_repo *r) { return r->path; }

int gc_repo_lock(gc_repo *r, int exclusive)
{
    if (flock(r->lock_fd, exclusive ? LOCK_EX : LOCK_SH) != 0) {
        gc_set_error("flock: %s", strerror(errno));
        return GC_EIO;
    }
    return GC_OK;
}
void gc_repo_unlock(gc_repo *r) { flock(r->lock_fd, LOCK_UN); }

/* ----------------------------------------------------------- work pool */

typedef struct work {
    gc_repo *r;
    gc_image *img;
    gc_manifest *m;
    const gc_manifest *parent;
    int mode;                         /* 0 create, 1 restore, 2 verify */
    uint64_t total;                   /* flattened chunk count */
    uint64_t prefix[GC_MAX_DEVICES + 1];
    atomic_uint_fast64_t next;
    atomic_int err;
    pthread_mutex_t mu;
    gc_stats st;
} work;

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

static void *worker(void *arg)
{
    work *w = arg;
    gc_image *img = w->img;
    const uint64_t cs = w->m->chunk_size;
    gc_stats local = {0};
    uint8_t *buf;
    if (img && img->buf_alloc) buf = img->buf_alloc(img, (size_t)cs);
    else buf = malloc((size_t)cs);
    if (!buf) { set_err(w, GC_ENOMEM); return NULL; }
    if (img && img->thread_attach) {
        int rc = img->thread_attach(img);
        if (rc) { set_err(w, rc); goto out; }
    }
    for (;;) {
        if (atomic_load(&w->err)) break;
        uint64_t k = atomic_fetch_add(&w->next, 1);
        if (k >= w->total) break;
        uint32_t d; uint64_t idx;
        locate(w, k, &d, &idx);
        gc_manifest_dev *dv = &w->m->dev[d];
        uint64_t off = idx * cs;
        size_t len = (size_t)((dv->size - off) < cs ? (dv->size - off) : cs);
        uint8_t *slot = dv->hashes[idx];
        uint64_t t0, t1, t2, t3;
        int rc;

        if (w->mode == 0) {
            t0 = gc_mono_ns();
            rc = img->read(img, d, off, buf, len);
            if (rc) { set_err(w, rc); break; }
            t1 = gc_mono_ns();
            gc_sha256(buf, len, slot);
            t2 = gc_mono_ns();
            int was_new = 0;
            rc = gc_store_put(w->r, slot, buf, len, &was_new);
            if (rc) { set_err(w, rc); break; }
            t3 = gc_mono_ns();
            local.chunks_total++;
            local.bytes_total += len;
            if (was_new) { local.chunks_new++; local.bytes_new += len; }
            if (w->parent && d < w->parent->device_count && idx < w->parent->dev[d].nchunks &&
                memcmp(slot, w->parent->dev[d].hashes[idx], GC_HASH_LEN) == 0)
                local.chunks_same_as_parent++;
            local.ns_read += t1 - t0; local.ns_hash += t2 - t1; local.ns_write += t3 - t2;
        } else {
            size_t got = 0;
            t0 = gc_mono_ns();
            rc = gc_store_get(w->r, slot, buf, (size_t)cs, &got, 0);
            t1 = gc_mono_ns();
            local.chunks_total++;
            local.bytes_total += len;
            local.ns_read += t1 - t0;
            if (rc == GC_ENOENT) {
                local.chunks_missing++;
                if (w->mode == 1) { set_err(w, GC_ENOENT); break; }
                continue;
            }
            if (rc) { set_err(w, rc); break; }
            uint8_t h[GC_HASH_LEN];
            gc_sha256(buf, got, h);
            t2 = gc_mono_ns();
            local.ns_hash += t2 - t1;
            if (got != len || memcmp(h, slot, GC_HASH_LEN) != 0) {
                local.chunks_corrupt++;
                if (w->mode == 1) {
                    char hex[GC_HEX_LEN + 1]; gc_hex(slot, GC_HASH_LEN, hex);
                    gc_set_error("chunk %s: stored bytes do not match hash", hex);
                    set_err(w, GC_ECORRUPT); break;
                }
                continue;
            }
            if (w->mode == 1) {
                rc = img->write(img, d, off, buf, len);
                t3 = gc_mono_ns();
                if (rc) { set_err(w, rc); break; }
                local.ns_write += t3 - t2;
            }
        }
    }
out:
    if (img && img->thread_detach) img->thread_detach(img);
    if (img && img->buf_free) img->buf_free(img, buf, (size_t)cs); else free(buf);
    pthread_mutex_lock(&w->mu);
    w->st.bytes_total += local.bytes_total; w->st.chunks_total += local.chunks_total;
    w->st.chunks_new += local.chunks_new;   w->st.bytes_new += local.bytes_new;
    w->st.chunks_same_as_parent += local.chunks_same_as_parent;
    w->st.chunks_missing += local.chunks_missing; w->st.chunks_corrupt += local.chunks_corrupt;
    w->st.ns_read += local.ns_read; w->st.ns_hash += local.ns_hash; w->st.ns_write += local.ns_write;
    pthread_mutex_unlock(&w->mu);
    return NULL;
}

static int run_pool(work *w, int threads)
{
    if (threads <= 0) { threads = gc_nprocs(); if (threads > 8) threads = 8; }
    if ((uint64_t)threads > w->total && w->total > 0) threads = (int)w->total;
    if (threads < 1) threads = 1;
    w->total = w->prefix[w->m->device_count];
    atomic_store(&w->next, 0);
    atomic_store(&w->err, 0);
    pthread_mutex_init(&w->mu, NULL);
    w->st.threads = threads;
    uint64_t t0 = gc_mono_ns();
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
    w->st.ns_wall = gc_mono_ns() - t0;
    return atomic_load(&w->err);
}

static void build_prefix(work *w)
{
    w->prefix[0] = 0;
    for (uint32_t d = 0; d < w->m->device_count; d++)
        w->prefix[d + 1] = w->prefix[d] + w->m->dev[d].nchunks;
    w->total = w->prefix[w->m->device_count];
}

/* ---------------------------------------------------------------- create */

int gc_snapshot_create(gc_repo *r, gc_image *img, const gc_snapshot_opts *o,
                       gc_stats *st, char id_out[GC_ID_LEN])
{
    if (!img || img->device_count == 0 || img->device_count > GC_MAX_DEVICES || !img->read) return GC_EINVAL;
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
        char p[PATH_MAX];
        gc_manifest_path(r, m->id, p, sizeof p);
        if (gc_file_exists(p)) { gc_set_error("snapshot %s already exists", m->id); rc = GC_EEXIST; goto out; }
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
    work w = {0};
    w.r = r; w.img = img; w.m = m; w.parent = parent; w.mode = 0;
    build_prefix(&w);
    rc = run_pool(&w, o ? o->threads : 0);
    gc_manifest_free(parent);
    if (rc == GC_OK) {
        m->stats = w.st;
        rc = gc_manifest_save(r, m);
    }
    gc_repo_unlock(r);
    if (st) *st = w.st;
    if (rc == GC_OK && id_out) snprintf(id_out, GC_ID_LEN, "%s", m->id);
out:
    gc_manifest_free(m);
    return rc;
}

/* --------------------------------------------------------------- restore */

int gc_snapshot_restore(gc_repo *r, const char *id, gc_image *img, int threads, gc_stats *st)
{
    if (!img || !img->write) return GC_EINVAL;
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
    work w = {0};
    w.r = r; w.img = img; w.m = m; w.mode = 1;
    build_prefix(&w);
    rc = run_pool(&w, threads);
    gc_repo_unlock(r);
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
    work w = {0};
    w.r = r; w.img = NULL; w.m = m; w.mode = 2;
    build_prefix(&w);
    rc = run_pool(&w, 0);
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
    char p[PATH_MAX];
    int rc = gc_manifest_path(r, id, p, sizeof p);
    if (rc) return rc;
    rc = gc_repo_lock(r, 0);
    if (rc) return rc;
    if (unlink(p) != 0) {
        rc = errno == ENOENT ? GC_ENOENT : GC_EIO;
        gc_set_error("unlink %s: %s", p, strerror(errno));
    }
    gc_repo_unlock(r);
    return rc;
}

typedef struct { char (*ids)[GC_ID_LEN]; size_t n, cap; } idlist;

static int collect_ids(gc_repo *r, idlist *l)
{
    char p[PATH_MAX];
    snprintf(p, sizeof p, "%s/snapshots", r->path);
    DIR *d = opendir(p);
    if (!d) return GC_EIO;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n <= 9 || strcmp(e->d_name + n - 9, ".manifest") != 0) continue;
        if (n - 9 >= GC_ID_LEN) continue;
        if (l->n == l->cap) {
            size_t nc = l->cap ? l->cap * 2 : 64;
            void *np = realloc(l->ids, nc * GC_ID_LEN);
            if (!np) { closedir(d); return GC_ENOMEM; }
            l->ids = np; l->cap = nc;
        }
        memcpy(l->ids[l->n], e->d_name, n - 9);
        l->ids[l->n][n - 9] = 0;
        l->n++;
    }
    closedir(d);
    return GC_OK;
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
    uint64_t deleted, bytes;
    int rc;
} gcwalk;

static int gc_cb(const uint8_t *hash, const char *path, void *u)
{
    gcwalk *g = u;
    (void)path;
    if (g->nlive && bsearch(hash, g->live, g->nlive, GC_HASH_LEN, cmp_hash)) return GC_OK;
    uint64_t b = 0;
    int rc = gc_store_delete(g->r, hash, &b);
    if (rc == GC_OK) { g->deleted++; g->bytes += b; }
    else if (rc != GC_ENOENT) return rc;
    return GC_OK;
}

/* Delete every chunk not referenced by any manifest. Takes the exclusive
 * repo lock, so no snapshot or restore is in flight. If any manifest fails
 * to load, nothing is deleted: an unknown live set is not an empty one. */
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
    gcwalk g = { .r = r, .live = live, .nlive = nlive };
    rc = gc_store_walk(r, gc_cb, &g);
    if (chunks_deleted) *chunks_deleted = g.deleted;
    if (bytes_freed) *bytes_freed = g.bytes;
    /* stale staging files from crashed writers */
    {
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
