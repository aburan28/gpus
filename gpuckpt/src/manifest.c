/* Snapshot manifest: a self-describing, self-checked text record.
 *
 *   gpuckpt-manifest 1
 *   id <id>
 *   parent <id|->
 *   created_unix_ns <n>
 *   pid <n>
 *   chunk_size <n>
 *   hash sha256
 *   backend <name>
 *   note <single line>
 *   device_count <n>
 *   device <d> size <bytes> ordinal <n> uuid <str|->
 *   c <d> <64 hex>                 one per chunk, in offset order
 *   stat <name> <n>
 *   end <64 hex>                   SHA-256 of every byte above this line
 *
 * The chunk list is complete: restoring needs this file and the chunks it
 * names, nothing else. The trailer makes truncation or bit-rot detectable
 * before any chunk is touched. Manifests are write-once; gc_manifest_save
 * refuses to overwrite an existing id.
 */
#include "objstore.h"
#include <stdarg.h>
#include <dirent.h>
#include <unistd.h>

int gc_manifest_key(const char *id, char *out, size_t outlen)
{
    if (!gc_valid_id(id)) return GC_EINVAL;
    int n = snprintf(out, outlen, "snapshots/%s.manifest", id);
    return (n < 0 || (size_t)n >= outlen) ? GC_EINVAL : GC_OK;
}

int gc_manifest_alloc_dev(gc_manifest *m, uint32_t d, uint64_t size, uint64_t chunk_size)
{
    if (d >= GC_MAX_DEVICES || chunk_size == 0) return GC_EINVAL;
    gc_manifest_dev *dv = &m->dev[d];
    dv->size = size;
    dv->nchunks = (size + chunk_size - 1) / chunk_size;
    dv->hashes = calloc(dv->nchunks ? dv->nchunks : 1, GC_HASH_LEN);
    if (!dv->hashes) return GC_ENOMEM;
    return GC_OK;
}

void gc_manifest_free(gc_manifest *m)
{
    if (!m) return;
    for (uint32_t d = 0; d < GC_MAX_DEVICES; d++) free(m->dev[d].hashes);
    free(m);
}

/* growable text buffer */
typedef struct { char *p; size_t n, cap; } sbuf;
static int sb_put(sbuf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->n + n + 1) nc *= 2;
        char *np = realloc(b->p, nc);
        if (!np) return GC_ENOMEM;
        b->p = np; b->cap = nc;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
    return GC_OK;
}
static int sb_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static int sb_printf(sbuf *b, const char *fmt, ...)
{
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof tmp) return GC_EINVAL;
    return sb_put(b, tmp, (size_t)n);
}

#define STAT_FIELDS(X) \
    X(bytes_total) X(chunks_total) X(chunks_new) X(bytes_new) X(chunks_same_as_parent) \
    X(chunks_missing) X(chunks_corrupt) X(ns_read) X(ns_hash) X(ns_write) X(ns_wall)

static int serialize(const gc_manifest *m, sbuf *b)
{
    int rc = 0;
    rc |= sb_printf(b, "gpuckpt-manifest %d\n", GC_FORMAT_VERSION);
    rc |= sb_printf(b, "id %s\n", m->id);
    rc |= sb_printf(b, "parent %s\n", m->parent[0] ? m->parent : "-");
    rc |= sb_printf(b, "created_unix_ns %llu\n", (unsigned long long)m->created_unix_ns);
    rc |= sb_printf(b, "pid %d\n", m->pid);
    rc |= sb_printf(b, "chunk_size %llu\n", (unsigned long long)m->chunk_size);
    rc |= sb_printf(b, "hash sha256\n");
    rc |= sb_printf(b, "backend %s\n", m->backend[0] ? m->backend : "-");
    rc |= sb_printf(b, "note %s\n", m->note[0] ? m->note : "-");
    rc |= sb_printf(b, "device_count %u\n", m->device_count);
    for (uint32_t d = 0; d < m->device_count; d++)
        rc |= sb_printf(b, "device %u size %llu ordinal %d uuid %s\n", d,
                        (unsigned long long)m->dev[d].size, m->dev[d].ordinal,
                        m->dev[d].uuid[0] ? m->dev[d].uuid : "-");
    for (uint32_t d = 0; d < m->device_count; d++) {
        char hex[GC_HEX_LEN + 1];
        for (uint64_t i = 0; i < m->dev[d].nchunks; i++) {
            gc_hex(m->dev[d].hashes[i], GC_HASH_LEN, hex);
            rc |= sb_printf(b, "c %u %s\n", d, hex);
        }
    }
#define X(f) rc |= sb_printf(b, "stat %s %llu\n", #f, (unsigned long long)m->stats.f);
    STAT_FIELDS(X)
#undef X
    rc |= sb_printf(b, "stat threads %d\n", m->stats.threads);
    if (rc) return GC_ENOMEM;
    uint8_t h[GC_HASH_LEN];
    gc_sha256(b->p, b->n, h);
    char hex[GC_HEX_LEN + 1];
    gc_hex(h, GC_HASH_LEN, hex);
    return sb_printf(b, "end %s\n", hex);
}

int gc_manifest_save(gc_repo *r, const gc_manifest *m)
{
    char key[160];
    int rc = gc_manifest_key(m->id, key, sizeof key);
    if (rc) return rc;
    if (gc_obj_head(r, key, NULL) == GC_OK) return GC_EEXIST;
    sbuf b = {0};
    rc = serialize(m, &b);
    if (rc) { free(b.p); return rc; }
    rc = gc_obj_put(r, key, b.p, b.n);
    free(b.p);
    return rc;
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s || (*end && *end != '\n' && *end != ' ')) return GC_EINVAL;
    *out = v;
    return GC_OK;
}

int gc_manifest_load(gc_repo *r, const char *id, gc_manifest **out)
{
    char key[160];
    int rc = gc_manifest_key(id, key, sizeof key);
    if (rc) return rc;
    void *data = NULL;
    size_t n = 0;
    rc = gc_obj_get(r, key, &data, &n);
    if (rc) return rc;
    char *text = data;

    /* trailer check first: the manifest must be intact before we trust a line */
    const char *end_line = NULL;
    if (n >= 70 && text[n - 1] == '\n') {
        const char *p = text + n - 2;
        while (p > text && *p != '\n') p--;
        if (*p == '\n') p++;
        if (strncmp(p, "end ", 4) == 0) end_line = p;
    }
    if (!end_line) { free(data); gc_set_error("manifest %s: missing trailer", id); return GC_ECORRUPT; }
    {
        uint8_t want[GC_HASH_LEN], got[GC_HASH_LEN];
        char hex[GC_HEX_LEN + 1];
        memcpy(hex, end_line + 4, GC_HEX_LEN); hex[GC_HEX_LEN] = 0;
        if (gc_unhex(hex, want, GC_HASH_LEN) != GC_OK) { free(data); return GC_ECORRUPT; }
        gc_sha256(text, (size_t)(end_line - text), got);
        if (memcmp(want, got, GC_HASH_LEN) != 0) {
            free(data);
            gc_set_error("manifest %s: trailer hash mismatch (truncated or corrupted)", id);
            return GC_ECORRUPT;
        }
    }

    gc_manifest *m = calloc(1, sizeof *m);
    if (!m) { free(data); return GC_ENOMEM; }
    for (uint32_t d = 0; d < GC_MAX_DEVICES; d++) m->dev[d].ordinal = -1;
    uint64_t next_chunk[GC_MAX_DEVICES] = {0};
    int saw_header = 0;
    char *save = NULL;
    for (char *line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
        if (!saw_header) {
            if (strncmp(line, "gpuckpt-manifest 1", 18) != 0) { rc = GC_EINVAL; goto fail; }
            saw_header = 1;
            continue;
        }
        char *sp = strchr(line, ' ');
        if (!sp) { rc = GC_EINVAL; goto fail; }
        *sp = 0;
        const char *k = line, *val = sp + 1;
        uint64_t u;
        if (!strcmp(k, "id")) { snprintf(m->id, sizeof m->id, "%s", val); }
        else if (!strcmp(k, "parent")) { if (strcmp(val, "-")) snprintf(m->parent, sizeof m->parent, "%s", val); }
        else if (!strcmp(k, "created_unix_ns")) { if (parse_u64(val, &u)) { rc = GC_EINVAL; goto fail; } m->created_unix_ns = u; }
        else if (!strcmp(k, "pid")) { m->pid = atoi(val); }
        else if (!strcmp(k, "chunk_size")) { if (parse_u64(val, &u) || !u) { rc = GC_EINVAL; goto fail; } m->chunk_size = u; }
        else if (!strcmp(k, "hash")) { if (strcmp(val, "sha256")) { rc = GC_EINVAL; goto fail; } }
        else if (!strcmp(k, "backend")) { snprintf(m->backend, sizeof m->backend, "%s", val); }
        else if (!strcmp(k, "note")) { if (strcmp(val, "-")) snprintf(m->note, sizeof m->note, "%s", val); }
        else if (!strcmp(k, "device_count")) {
            if (parse_u64(val, &u) || u > GC_MAX_DEVICES) { rc = GC_EINVAL; goto fail; }
            m->device_count = (uint32_t)u;
        }
        else if (!strcmp(k, "device")) {
            unsigned d; unsigned long long sz; int ord; char uuid[GC_UUID_LEN];
            if (sscanf(val, "%u size %llu ordinal %d uuid %71s", &d, &sz, &ord, uuid) != 4 ||
                d >= m->device_count || !m->chunk_size) { rc = GC_EINVAL; goto fail; }
            rc = gc_manifest_alloc_dev(m, d, sz, m->chunk_size);
            if (rc) goto fail;
            m->dev[d].ordinal = ord;
            if (strcmp(uuid, "-")) snprintf(m->dev[d].uuid, sizeof m->dev[d].uuid, "%s", uuid);
        }
        else if (!strcmp(k, "c")) {
            unsigned d; char hex[GC_HEX_LEN + 1];
            if (sscanf(val, "%u %64s", &d, hex) != 2 || d >= m->device_count || !m->dev[d].hashes ||
                next_chunk[d] >= m->dev[d].nchunks) { rc = GC_EINVAL; goto fail; }
            if (gc_unhex(hex, m->dev[d].hashes[next_chunk[d]], GC_HASH_LEN)) { rc = GC_EINVAL; goto fail; }
            next_chunk[d]++;
        }
        else if (!strcmp(k, "stat")) {
            char name[64]; unsigned long long v;
            if (sscanf(val, "%63s %llu", name, &v) != 2) { rc = GC_EINVAL; goto fail; }
#define X(f) if (!strcmp(name, #f)) m->stats.f = v; else
            STAT_FIELDS(X)
#undef X
            if (!strcmp(name, "threads")) m->stats.threads = (int)v;
        }
        else if (!strcmp(k, "end")) { break; }
        else { /* unknown keys are ignored for forward compatibility */ }
    }
    for (uint32_t d = 0; d < m->device_count; d++)
        if (!m->dev[d].hashes || next_chunk[d] != m->dev[d].nchunks) {
            gc_set_error("manifest %s: device %u lists %llu of %llu chunks", id, d,
                         (unsigned long long)next_chunk[d], (unsigned long long)m->dev[d].nchunks);
            rc = GC_EINVAL; goto fail;
        }
    if (strcmp(m->id, id) != 0) { gc_set_error("manifest %s: id field says %s", id, m->id); rc = GC_EINVAL; goto fail; }
    free(data);
    *out = m;
    return GC_OK;
fail:
    if (!gc_last_error()[0]) gc_set_error("manifest %s: malformed", id);
    free(data);
    gc_manifest_free(m);
    return rc;
}

void gc_manifest_print(const gc_manifest *m, int with_chunks, void *fpv)
{
    FILE *fp = fpv;
    fprintf(fp, "id              %s\n", m->id);
    fprintf(fp, "parent          %s\n", m->parent[0] ? m->parent : "-");
    time_t t = (time_t)(m->created_unix_ns / 1000000000ull);
    char ts[64]; struct tm tm; gmtime_r(&t, &tm);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", &tm);
    fprintf(fp, "created         %s\n", ts);
    fprintf(fp, "pid             %d\n", m->pid);
    fprintf(fp, "backend         %s\n", m->backend);
    fprintf(fp, "chunk_size      %llu\n", (unsigned long long)m->chunk_size);
    if (m->note[0]) fprintf(fp, "note            %s\n", m->note);
    fprintf(fp, "devices         %u\n", m->device_count);
    for (uint32_t d = 0; d < m->device_count; d++)
        fprintf(fp, "  device %u     size=%llu chunks=%llu ordinal=%d uuid=%s\n", d,
                (unsigned long long)m->dev[d].size, (unsigned long long)m->dev[d].nchunks,
                m->dev[d].ordinal, m->dev[d].uuid[0] ? m->dev[d].uuid : "-");
    const gc_stats *s = &m->stats;
    fprintf(fp, "bytes_total     %llu\n", (unsigned long long)s->bytes_total);
    fprintf(fp, "chunks_total    %llu\n", (unsigned long long)s->chunks_total);
    fprintf(fp, "chunks_new      %llu\n", (unsigned long long)s->chunks_new);
    fprintf(fp, "bytes_new       %llu\n", (unsigned long long)s->bytes_new);
    fprintf(fp, "same_as_parent  %llu\n", (unsigned long long)s->chunks_same_as_parent);
    fprintf(fp, "ns_read         %llu\n", (unsigned long long)s->ns_read);
    fprintf(fp, "ns_hash         %llu\n", (unsigned long long)s->ns_hash);
    fprintf(fp, "ns_write        %llu\n", (unsigned long long)s->ns_write);
    fprintf(fp, "ns_wall         %llu\n", (unsigned long long)s->ns_wall);
    fprintf(fp, "threads         %d\n", s->threads);
    if (with_chunks) {
        char hex[GC_HEX_LEN + 1];
        for (uint32_t d = 0; d < m->device_count; d++)
            for (uint64_t i = 0; i < m->dev[d].nchunks; i++) {
                gc_hex(m->dev[d].hashes[i], GC_HASH_LEN, hex);
                fprintf(fp, "c %u %llu %s\n", d, (unsigned long long)i, hex);
            }
    }
}
