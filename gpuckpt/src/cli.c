/* gpuckpt command-line front end. See `gpuckpt help`. */
#include "internal.h"
#include <unistd.h>

static const char *USAGE =
"gpuckpt " GC_VERSION " -- incremental, deduplicated GPU checkpoint storage\n"
"\n"
"Repository (R is a directory or s3://bucket[/prefix]; see docs/s3.md)\n"
"  init     --repo R [--chunk-size BYTES]        create a chunk store (default 1 MiB chunks)\n"
"  list     --repo R                             list snapshot ids\n"
"  show     --repo R --snapshot ID [--chunks]    print a manifest\n"
"  verify   --repo R [--snapshot ID]             re-hash every chunk of one or all snapshots\n"
"  forget   --repo R --snapshot ID               drop a manifest (chunks stay until gc)\n"
"  gc       --repo R [--grace-seconds N]         delete chunks no manifest references\n"
"           (s3: unreferenced chunks younger than N seconds are kept, default 3600)\n"
"\n"
"Host-file images (tests, benchmarks, images already on disk)\n"
"  snapshot-file --repo R --input F[,F...] [--id ID] [--parent ID] [--threads N] [--note TEXT]\n"
"  restore-file  --repo R --snapshot ID --output F[,F...] [--threads N]\n"
"\n"
"CUDA custom-storage checkpoint of a running process (helper runs as root\n"
"or with the same privileges cuda-checkpoint needs; never on its own pid)\n"
"  state    --pid P                              print checkpoint state\n"
"  lock     --pid P [--timeout-ms N]             RUNNING -> LOCKED\n"
"  unlock   --pid P                              LOCKED  -> RUNNING\n"
"  snapshot --repo R --pid P [--resume] [--timeout-ms N] [--id ID] [--parent ID]\n"
"           [--threads N] [--note TEXT] [--complete-on-error]\n"
"           lock if RUNNING, checkpoint GPU memory to the store, complete.\n"
"           Without --resume the target is left CHECKPOINTED (GPU released).\n"
"           With --resume it is restored from the snapshot just taken and unlocked.\n"
"  restore  --repo R --pid P --snapshot ID [--threads N] [--no-unlock]\n"
"           target must be CHECKPOINTED; refills GPU memory, completes, unlocks.\n"
"  restore-tid --pid P                           driver restore thread id (for CRIU integration)\n"
"\n"
"  version                                       build info (hash impl, cuda header mode)\n"
"\n"
"Environment: GPUCKPT_LIBCUDA  path to libcuda (default libcuda.so.1)\n"
"Exit codes: 0 ok, 1 usage, 2 operation failed, 3 target left in an unexpected state\n";

typedef struct {
    const char *repo, *snapshot, *id, *parent, *input, *output, *note, *libcuda;
    int pid, threads, resume, no_unlock, complete_on_error, chunks, grace_seconds;
    unsigned timeout_ms;
    uint64_t chunk_size;
} opts;

static int parse(int argc, char **argv, opts *o)
{
    memset(o, 0, sizeof *o);
    o->chunk_size = 1u << 20;
    o->timeout_ms = 30000;
    o->grace_seconds = -1;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define TAKE(name, field) if (!strcmp(a, name)) { if (!v) { fprintf(stderr, "%s needs a value\n", a); return 1; } o->field = v; i++; continue; }
#define TAKEI(name, field, conv) if (!strcmp(a, name)) { if (!v) { fprintf(stderr, "%s needs a value\n", a); return 1; } o->field = conv; i++; continue; }
        TAKE("--repo", repo) TAKE("--snapshot", snapshot) TAKE("--id", id) TAKE("--parent", parent)
        TAKE("--input", input) TAKE("--output", output) TAKE("--note", note) TAKE("--libcuda", libcuda)
        TAKEI("--pid", pid, atoi(v)) TAKEI("--threads", threads, atoi(v))
        TAKEI("--timeout-ms", timeout_ms, (unsigned)strtoul(v, NULL, 10))
        TAKEI("--chunk-size", chunk_size, strtoull(v, NULL, 10))
        TAKEI("--grace-seconds", grace_seconds, atoi(v))
#undef TAKE
#undef TAKEI
        if (!strcmp(a, "--resume")) { o->resume = 1; continue; }
        if (!strcmp(a, "--no-unlock")) { o->no_unlock = 1; continue; }
        if (!strcmp(a, "--complete-on-error")) { o->complete_on_error = 1; continue; }
        if (!strcmp(a, "--chunks")) { o->chunks = 1; continue; }
        fprintf(stderr, "unknown option %s\n", a);
        return 1;
    }
    return 0;
}

static int need(const opts *o, const char *what, const void *v)
{
    if (v) return 0;
    (void)o;
    fprintf(stderr, "missing %s\n", what);
    return 1;
}

static int fail(const char *what, int rc)
{
    const char *detail = gc_last_error();
    fprintf(stderr, "error: %s: %s%s%s\n", what, gc_strerror(rc), detail[0] ? ": " : "", detail);
    return 2;
}

static void print_stats(const char *prefix, const gc_stats *s)
{
    printf("%sbytes_total=%llu\n", prefix, (unsigned long long)s->bytes_total);
    printf("%schunks_total=%llu\n", prefix, (unsigned long long)s->chunks_total);
    printf("%schunks_new=%llu\n", prefix, (unsigned long long)s->chunks_new);
    printf("%sbytes_new=%llu\n", prefix, (unsigned long long)s->bytes_new);
    printf("%schunks_same_as_parent=%llu\n", prefix, (unsigned long long)s->chunks_same_as_parent);
    printf("%schunks_missing=%llu\n", prefix, (unsigned long long)s->chunks_missing);
    printf("%schunks_corrupt=%llu\n", prefix, (unsigned long long)s->chunks_corrupt);
    printf("%sthreads=%d\n", prefix, s->threads);
    printf("%sns_wall=%llu\n", prefix, (unsigned long long)s->ns_wall);
    printf("%sns_read_sum=%llu\n", prefix, (unsigned long long)s->ns_read);
    printf("%sns_hash_sum=%llu\n", prefix, (unsigned long long)s->ns_hash);
    printf("%sns_write_sum=%llu\n", prefix, (unsigned long long)s->ns_write);
    double wall_s = (double)s->ns_wall / 1e9;
    if (wall_s > 0)
        printf("%sthroughput_MiB_s=%.1f\n", prefix, (double)s->bytes_total / 1048576.0 / wall_s);
    if (s->bytes_total)
        printf("%snew_fraction=%.4f\n", prefix, (double)s->bytes_new / (double)s->bytes_total);
}

static int split_paths(const char *list, const char **out, int max)
{
    static char buf[32768];
    snprintf(buf, sizeof buf, "%s", list);
    int n = 0;
    char *save = NULL;
    for (char *t = strtok_r(buf, ",", &save); t && n < max; t = strtok_r(NULL, ",", &save)) out[n++] = t;
    return n;
}

static int list_cb(const char *id, void *u) { (void)u; printf("%s\n", id); return 0; }

static int open_repo(const opts *o, gc_repo **r)
{
    if (need(o, "--repo", o->repo)) return 1;
    int rc = gc_repo_open(o->repo, r);
    if (rc) return fail("open repo", rc);
    return 0;
}

/* ----------------------------------------------------------- commands */

static int cmd_init(const opts *o)
{
    if (need(o, "--repo", o->repo)) return 1;
    int rc = gc_repo_init(o->repo, o->chunk_size);
    if (rc) return fail("init", rc);
    printf("initialised %s chunk_size=%llu\n", o->repo, (unsigned long long)o->chunk_size);
    return 0;
}

static int cmd_list(const opts *o)
{
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    rc = gc_repo_list(r, list_cb, NULL);
    gc_repo_close(r);
    return rc ? fail("list", rc) : 0;
}

static int cmd_show(const opts *o)
{
    if (need(o, "--snapshot", o->snapshot)) return 1;
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    gc_manifest *m;
    rc = gc_manifest_load(r, o->snapshot, &m);
    if (rc) { gc_repo_close(r); return fail("load manifest", rc); }
    gc_manifest_print(m, o->chunks, stdout);
    gc_manifest_free(m);
    gc_repo_close(r);
    return 0;
}

typedef struct { gc_repo *r; int bad; } verify_ctx;
static int verify_one(const char *id, void *u)
{
    verify_ctx *v = u;
    gc_stats st = {0};
    int rc = gc_snapshot_verify(v->r, id, &st);
    printf("%s %s chunks=%llu missing=%llu corrupt=%llu\n", id, rc ? "FAIL" : "ok",
           (unsigned long long)st.chunks_total, (unsigned long long)st.chunks_missing,
           (unsigned long long)st.chunks_corrupt);
    if (rc) { v->bad++; fprintf(stderr, "  %s: %s %s\n", id, gc_strerror(rc), gc_last_error()); }
    return 0;
}

static int cmd_verify(const opts *o)
{
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    verify_ctx v = { r, 0 };
    if (o->snapshot) verify_one(o->snapshot, &v);
    else gc_repo_list(r, verify_one, &v);
    gc_repo_close(r);
    return v.bad ? 2 : 0;
}

static int cmd_forget(const opts *o)
{
    if (need(o, "--snapshot", o->snapshot)) return 1;
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    rc = gc_snapshot_forget(r, o->snapshot);
    gc_repo_close(r);
    if (rc) return fail("forget", rc);
    printf("forgot %s (run gc to reclaim chunks)\n", o->snapshot);
    return 0;
}

static int cmd_gc(const opts *o)
{
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    uint64_t n = 0, b = 0;
    if (o->grace_seconds >= 0) gc_repo_set_gc_grace(r, o->grace_seconds);
    rc = gc_repo_gc(r, &n, &b);
    gc_repo_close(r);
    if (rc) return fail("gc", rc);
    printf("chunks_deleted=%llu\nbytes_freed=%llu\n", (unsigned long long)n, (unsigned long long)b);
    return 0;
}

static int cmd_snapshot_file(const opts *o)
{
    if (need(o, "--input", o->input)) return 1;
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    const char *paths[GC_MAX_DEVICES];
    int n = split_paths(o->input, paths, GC_MAX_DEVICES);
    gc_image img;
    rc = gc_image_file_open(&img, "r", (uint32_t)n, paths, NULL);
    if (rc) { gc_repo_close(r); return fail("open input", rc); }
    gc_snapshot_opts so = { .id = o->id, .parent = o->parent, .pid = o->pid, .threads = o->threads, .note = o->note };
    gc_stats st = {0};
    char id[GC_ID_LEN];
    rc = gc_snapshot_create(r, &img, &so, &st, id);
    gc_image_file_close(&img);
    gc_repo_close(r);
    if (rc) return fail("snapshot", rc);
    printf("snapshot=%s\n", id);
    print_stats("", &st);
    return 0;
}

static int cmd_restore_file(const opts *o)
{
    if (need(o, "--snapshot", o->snapshot) || need(o, "--output", o->output)) return 1;
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    gc_manifest *m;
    rc = gc_manifest_load(r, o->snapshot, &m);
    if (rc) { gc_repo_close(r); return fail("load manifest", rc); }
    const char *paths[GC_MAX_DEVICES];
    int n = split_paths(o->output, paths, GC_MAX_DEVICES);
    if ((uint32_t)n != m->device_count) {
        fprintf(stderr, "error: snapshot has %u devices, %d output paths given\n", m->device_count, n);
        gc_manifest_free(m); gc_repo_close(r); return 2;
    }
    uint64_t sizes[GC_MAX_DEVICES];
    for (int i = 0; i < n; i++) sizes[i] = m->dev[i].size;
    gc_manifest_free(m);
    gc_image img;
    rc = gc_image_file_open(&img, "w", (uint32_t)n, paths, sizes);
    if (rc) { gc_repo_close(r); return fail("open output", rc); }
    gc_stats st = {0};
    rc = gc_snapshot_restore(r, o->snapshot, &img, o->threads, &st);
    gc_image_file_close(&img);
    gc_repo_close(r);
    if (rc) return fail("restore", rc);
    printf("restored=%s\n", o->snapshot);
    print_stats("", &st);
    return 0;
}

/* ------------------------------------------------------- CUDA commands */

static int open_cuda(const opts *o, gc_cuda **c)
{
    int rc = gc_cuda_open(c, o->libcuda);
    if (rc) return fail("open CUDA driver", rc);
    return 0;
}

static int report_state(gc_cuda *c, int pid)
{
    int st = GC_PS_UNKNOWN;
    int rc = gc_cuda_get_state(c, pid, &st);
    if (rc) return fail("get state", rc);
    printf("pid=%d state=%s\n", pid, gc_proc_state_name(st));
    return 0;
}

static int cmd_state(const opts *o)
{
    if (!o->pid) { fprintf(stderr, "missing --pid\n"); return 1; }
    gc_cuda *c; int rc = open_cuda(o, &c); if (rc) return rc;
    rc = report_state(c, o->pid);
    gc_cuda_close(c);
    return rc;
}

static int cmd_lock(const opts *o, int lock)
{
    if (!o->pid) { fprintf(stderr, "missing --pid\n"); return 1; }
    gc_cuda *c; int rc = open_cuda(o, &c); if (rc) return rc;
    rc = lock ? gc_cuda_lock(c, o->pid, o->timeout_ms) : gc_cuda_unlock(c, o->pid);
    if (rc) { fail(lock ? "lock" : "unlock", rc); gc_cuda_close(c); return 2; }
    rc = report_state(c, o->pid);
    gc_cuda_close(c);
    return rc;
}

static int cmd_restore_tid(const opts *o)
{
    if (!o->pid) { fprintf(stderr, "missing --pid\n"); return 1; }
    gc_cuda *c; int rc = open_cuda(o, &c); if (rc) return rc;
    int tid = 0;
    rc = gc_cuda_restore_thread_id(c, o->pid, &tid);
    gc_cuda_close(c);
    if (rc) return fail("restore thread id", rc);
    printf("pid=%d restore_tid=%d\n", o->pid, tid);
    return 0;
}

/* Restore a CHECKPOINTED target from snapshot id; on success target is
 * LOCKED (or RUNNING if unlock). Shared by `restore` and `snapshot --resume`. */
static int do_restore(gc_cuda *c, gc_repo *r, const opts *o, const char *id, int unlock)
{
    gc_image img;
    int rc = gc_cuda_restore_begin(c, o->pid, &img);
    if (rc) return fail("restore begin", rc);
    printf("restore: mapped %u device image(s) for refill\n", img.device_count);
    gc_stats st = {0};
    uint64_t t0 = gc_mono_ns();
    rc = gc_snapshot_restore(r, id, &img, o->threads, &st);
    if (rc) {
        fail("restore copy", rc);
        fprintf(stderr, "target pid %d is RESTORING with a partially filled image; operation NOT completed\n", o->pid);
        if (o->complete_on_error) {
            int rc2 = gc_cuda_op_complete(c, &img);
            fprintf(stderr, "--complete-on-error: complete %s\n", rc2 ? gc_last_error() : "ok (target LOCKED with incomplete memory!)");
        }
        gc_cuda_image_close(c, &img);
        return 3;
    }
    uint64_t t1 = gc_mono_ns();
    rc = gc_cuda_op_complete(c, &img);
    uint64_t t2 = gc_mono_ns();
    gc_cuda_image_close(c, &img);
    if (rc) { fail("restore complete", rc); return 3; }
    print_stats("restore.", &st);
    printf("restore.ns_copy=%llu\nrestore.ns_complete=%llu\n", (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1));
    if (unlock) {
        rc = gc_cuda_unlock(c, o->pid);
        if (rc) { fail("unlock", rc); return 3; }
    }
    return 0;
}

static int cmd_snapshot(const opts *o)
{
    if (!o->pid) { fprintf(stderr, "missing --pid\n"); return 1; }
    if (o->pid == getpid()) { fprintf(stderr, "custom-storage checkpoint of the calling process is not supported\n"); return 1; }
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    gc_cuda *c; rc = open_cuda(o, &c); if (rc) { gc_repo_close(r); return rc; }

    int st = GC_PS_UNKNOWN, we_locked = 0, ret = 0;
    rc = gc_cuda_get_state(c, o->pid, &st);
    if (rc) { ret = fail("get state", rc); goto out; }
    uint64_t t_lock0 = gc_mono_ns(), t_lock1 = t_lock0;
    if (st == GC_PS_RUNNING) {
        rc = gc_cuda_lock(c, o->pid, o->timeout_ms);
        if (rc) { ret = fail("lock", rc); goto out; }
        we_locked = 1;
        t_lock1 = gc_mono_ns();
    } else if (st != GC_PS_LOCKED) {
        fprintf(stderr, "error: target pid %d is %s; snapshot needs RUNNING or LOCKED\n", o->pid, gc_proc_state_name(st));
        ret = 2; goto out;
    }

    gc_image img;
    uint64_t t_map0 = gc_mono_ns();
    rc = gc_cuda_checkpoint_begin(c, o->pid, &img);
    uint64_t t_map1 = gc_mono_ns();
    if (rc) {
        fail("checkpoint begin", rc);
        if (we_locked) { if (gc_cuda_unlock(c, o->pid) == 0) fprintf(stderr, "target unlocked again\n"); }
        ret = 2; goto out;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < img.device_count; i++) total += img.dev[i].size;
    printf("checkpoint: mapped %u device image(s), %llu bytes\n", img.device_count, (unsigned long long)total);
    for (uint32_t i = 0; i < img.device_count; i++)
        printf("  device %u size=%llu ordinal=%d uuid=%s\n", i, (unsigned long long)img.dev[i].size,
               img.dev[i].ordinal, img.dev[i].uuid);

    gc_snapshot_opts so = { .id = o->id, .parent = o->parent, .pid = o->pid, .threads = o->threads, .note = o->note };
    gc_stats cst = {0};
    char id[GC_ID_LEN] = "";
    uint64_t t_copy0 = gc_mono_ns();
    rc = gc_snapshot_create(r, &img, &so, &cst, id);
    uint64_t t_copy1 = gc_mono_ns();
    if (rc) {
        fail("snapshot", rc);
        fprintf(stderr, "target pid %d is CHECKPOINTING with GPU memory still mapped; operation NOT completed.\n"
                        "Completing now would release the GPU memory without a stored copy.\n", o->pid);
        if (o->complete_on_error) {
            int rc2 = gc_cuda_op_complete(c, &img);
            fprintf(stderr, "--complete-on-error: complete %s\n", rc2 ? gc_last_error() : "ok (target CHECKPOINTED, data NOT saved)");
        }
        gc_cuda_image_close(c, &img);
        ret = 3; goto out;
    }
    uint64_t t_cpl0 = gc_mono_ns();
    rc = gc_cuda_op_complete(c, &img);
    uint64_t t_cpl1 = gc_mono_ns();
    gc_cuda_image_close(c, &img);
    if (rc) {
        fail("checkpoint complete", rc);
        fprintf(stderr, "snapshot %s is stored and valid; target pid %d state unknown, check with `gpuckpt state`\n", id, o->pid);
        ret = 3; goto out;
    }
    printf("snapshot=%s\n", id);
    print_stats("checkpoint.", &cst);
    printf("checkpoint.ns_lock=%llu\ncheckpoint.ns_map=%llu\ncheckpoint.ns_copy=%llu\ncheckpoint.ns_complete=%llu\n",
           (unsigned long long)(t_lock1 - t_lock0), (unsigned long long)(t_map1 - t_map0),
           (unsigned long long)(t_copy1 - t_copy0), (unsigned long long)(t_cpl1 - t_cpl0));

    if (o->resume) {
        ret = do_restore(c, r, o, id, 1);
        if (ret == 0) printf("resume.ns_total=%llu\n", (unsigned long long)(gc_mono_ns() - t_cpl1));
    }
    if (ret == 0) ret = report_state(c, o->pid);
out:
    gc_cuda_close(c);
    gc_repo_close(r);
    return ret;
}

static int cmd_restore(const opts *o)
{
    if (!o->pid) { fprintf(stderr, "missing --pid\n"); return 1; }
    if (need(o, "--snapshot", o->snapshot)) return 1;
    gc_repo *r; int rc = open_repo(o, &r); if (rc) return rc;
    gc_cuda *c; rc = open_cuda(o, &c); if (rc) { gc_repo_close(r); return rc; }
    int st = GC_PS_UNKNOWN, ret;
    rc = gc_cuda_get_state(c, o->pid, &st);
    if (rc) { ret = fail("get state", rc); goto out; }
    if (st != GC_PS_CHECKPOINTED) {
        fprintf(stderr, "error: target pid %d is %s; restore needs CHECKPOINTED\n", o->pid, gc_proc_state_name(st));
        ret = 2; goto out;
    }
    ret = do_restore(c, r, o, o->snapshot, !o->no_unlock);
    if (ret == 0) { printf("restored=%s\n", o->snapshot); ret = report_state(c, o->pid); }
out:
    gc_cuda_close(c);
    gc_repo_close(r);
    return ret;
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        fputs(USAGE, argc < 2 ? stderr : stdout);
        return argc < 2 ? 1 : 0;
    }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "version")) {
        printf("gpuckpt %s\nhash: sha256 (%s)\ncuda-header: %s\ns3: %s\nformat: %d\n", GC_VERSION, gc_sha256_impl(),
               gc_cuda_header_mode(), gc_s3_backend(), GC_FORMAT_VERSION);
        return 0;
    }
    opts o;
    if (parse(argc, argv, &o)) return 1;
    if (!strcmp(cmd, "init")) return cmd_init(&o);
    if (!strcmp(cmd, "list")) return cmd_list(&o);
    if (!strcmp(cmd, "show")) return cmd_show(&o);
    if (!strcmp(cmd, "verify")) return cmd_verify(&o);
    if (!strcmp(cmd, "forget")) return cmd_forget(&o);
    if (!strcmp(cmd, "gc")) return cmd_gc(&o);
    if (!strcmp(cmd, "snapshot-file")) return cmd_snapshot_file(&o);
    if (!strcmp(cmd, "restore-file")) return cmd_restore_file(&o);
    if (!strcmp(cmd, "state")) return cmd_state(&o);
    if (!strcmp(cmd, "lock")) return cmd_lock(&o, 1);
    if (!strcmp(cmd, "unlock")) return cmd_lock(&o, 0);
    if (!strcmp(cmd, "snapshot")) return cmd_snapshot(&o);
    if (!strcmp(cmd, "restore")) return cmd_restore(&o);
    if (!strcmp(cmd, "restore-tid")) return cmd_restore_tid(&o);
    fprintf(stderr, "unknown command %s\n\n%s", cmd, USAGE);
    return 1;
}
