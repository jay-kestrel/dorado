/* dorado_c — standalone driver for the C port of dorado's particle walk.
 *
 * Standard invocation contract (spec "Interface contracts"):
 *   dorado_c --input bench/fixture_rcm/ --particles N --iterations N \
 *            --seed N --out out/c/
 *
 * The driver loads the normalized fixture (params.json + uncompressed npy)
 * untimed, measures the compute region (Particles-init derivations,
 * particle seeding, all iterations) with a monotonic clock, then writes
 * walk_data.json (json.dumps-compatible) and timing.json (same schema as
 * benchmarks/run_reference.py) into --out.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>

#include "field.h"
#include "json.h"
#include "walker.h"

#define TIMED_REGION "Particles init (velocity/cell-type/D8/weight build) + " \
    "generate_particles + run_iteration loop; npy/params load and "          \
    "walk-data/timing JSON writes excluded"

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s --input <fixture_dir> --out <dir>\n"
            "          [--particles N] [--iterations N] [--seed N]\n"
            "          [--seed-x i,j,...] [--seed-y i,j,...]\n",
            prog);
}

/* basename minus the "fixture_" prefix — mirrors run_reference.py's
 * field-name derivation for the default seed boxes */
static void default_seed_box(const char *input, const int **sx, int *nsx,
                             const int **sy, int *nsy)
{
    static const int rcm_x[2] = { 15, 16 };
    static const int rcm_y[3] = { 137, 138, 139 };
    static const int anuga_x[10] = { 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
    static const int anuga_y[5] = { 48, 49, 50, 51, 52 };
    const char *base = strrchr(input, '/');
    base = base ? base + 1 : input;
    if (strncmp(base, "fixture_", 8) == 0)
        base += 8;
    if (strcmp(base, "rcm") == 0) {
        *sx = rcm_x; *nsx = 2; *sy = rcm_y; *nsy = 3;
    } else if (strcmp(base, "anuga") == 0) {
        *sx = anuga_x; *nsx = 10; *sy = anuga_y; *nsy = 5;
    } else {
        fprintf(stderr,
                "dorado_c: unknown fixture '%s': pass --seed-x/--seed-y "
                "explicitly\n", base);
        exit(2);
    }
}

static int parse_int(const char *s)
{
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || *end) {
        fprintf(stderr, "dorado_c: bad integer '%s'\n", s);
        exit(2);
    }
    return (int)v;
}

static double monotonic_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void write_file(const char *path, const char *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(buf, 1, len, f) != len) {
        fprintf(stderr, "dorado_c: cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

/* mkdir -p equivalent for the output directory */
static void mkdir_p(const char *dir)
{
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", dir);
    size_t len = strlen(tmp);
    while (len > 1 && tmp[len - 1] == '/')
        tmp[--len] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                fprintf(stderr, "dorado_c: cannot create %s\n", tmp);
                exit(1);
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "dorado_c: cannot create %s\n", tmp);
        exit(1);
    }
}

static void write_walk_data(const char *path, const Walk *w)
{
    JsonBuf b;
    jbuf_init(&b);
    static const char *keys[5] = { "xinds", "yinds", "travel_times",
                                   "stranded", "stranded_time" };
    for (int field = 0; field < 5; field++) {
        jbuf_puts(&b, field == 0 ? "{\"" : ", \"");
        jbuf_puts(&b, keys[field]);
        jbuf_puts(&b, "\": [");
        for (int i = 0; i < w->n; i++) {
            if (i)
                jbuf_puts(&b, ", ");
            jbuf_puts(&b, "[");
            int len = w->h.xh_n[i];
            for (int p = 0; p < len; p++) {
                if (p)
                    jbuf_puts(&b, ", ");
                switch (field) {
                case 0:
                    jbuf_number(&b, w->h.xh[i][p], true);
                    break;
                case 1:
                    jbuf_number(&b, w->h.yh[i][p], true);
                    break;
                case 2:
                    jbuf_number(&b, w->h.th[i][p].v, w->h.th[i][p].is_int);
                    break;
                case 3:
                    jbuf_number(&b, w->h.sh[i][p], true);
                    break;
                default:
                    jbuf_number(&b, w->h.sth[i][p].v, w->h.sth[i][p].is_int);
                }
            }
            jbuf_puts(&b, "]");
        }
        jbuf_puts(&b, "]");
    }
    jbuf_puts(&b, "}");
    write_file(path, b.buf, b.len);
    jbuf_free(&b);
}

static void write_timing(const char *path, const char *fixture_dir,
                         const char *model, int particles, int iterations,
                         int seed, double seconds, long appended)
{
    char rev[64] = "unknown";
    FILE *g = popen("git rev-parse HEAD 2>/dev/null", "r");
    if (g && fgets(rev, sizeof(rev), g))
        rev[strcspn(rev, "\n")] = '\0';
    if (g)
        pclose(g);

    char ts[64];
    time_t now = time(NULL);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S+00:00", &tm_utc);

    struct utsname un;
    memset(&un, 0, sizeof(un));
    uname(&un);
    char platform[256];
    snprintf(platform, sizeof(platform), "%s %s %s", un.sysname, un.release,
             un.machine);
    char compiler[128];
#ifdef __clang__
    snprintf(compiler, sizeof(compiler), "clang %s", __VERSION__);
#else
    snprintf(compiler, sizeof(compiler), "gcc %s", __VERSION__);
#endif

    char seconds_str[40];
    json_fmt_double(seconds_str, seconds);

    JsonBuf b;
    jbuf_init(&b);
    char tmp[512];
    jbuf_puts(&b, "{\n");
    jbuf_puts(&b, "  \"schema_version\": 1,\n");
    jbuf_puts(&b, "  \"implementation\": \"c\",\n");
    snprintf(tmp, sizeof(tmp), "  \"fixture_dir\": \"%s\",\n", fixture_dir);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"model\": \"%s\",\n", model);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"particles\": %d,\n", particles);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"iterations\": %d,\n", iterations);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"seed\": %d,\n", seed);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"seconds\": %s,\n", seconds_str);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"appended_steps\": %ld,\n", appended);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"timed_region\": \"%s\",\n", TIMED_REGION);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"timestamp_utc\": \"%s\",\n", ts);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"revision\": \"%s\",\n", rev);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"compiler\": \"%s\",\n", compiler);
    jbuf_puts(&b, tmp);
    snprintf(tmp, sizeof(tmp), "  \"platform\": \"%s\"\n", platform);
    jbuf_puts(&b, tmp);
    jbuf_puts(&b, "}\n");
    write_file(path, b.buf, b.len);
    jbuf_free(&b);
}

int main(int argc, char **argv)
{
    const char *input = NULL, *out = NULL;
    int particles = -1, iterations = -1;
    long seed = -1;
    int seed_x[64], seed_y[64];
    int n_seed_x = 0, n_seed_y = 0;

    for (int i = 1; i < argc; i += 2) {
        const char *flag = argv[i];
        const char *val = i + 1 < argc ? argv[i + 1] : NULL;
        if (!val) {
            usage(argv[0]);
            return 2;
        }
        if (strcmp(flag, "--input") == 0)
            input = val;
        else if (strcmp(flag, "--out") == 0)
            out = val;
        else if (strcmp(flag, "--particles") == 0)
            particles = parse_int(val);
        else if (strcmp(flag, "--iterations") == 0)
            iterations = parse_int(val);
        else if (strcmp(flag, "--seed") == 0)
            seed = strtol(val, NULL, 10);
        else if (strcmp(flag, "--seed-x") == 0 ||
                 strcmp(flag, "--seed-y") == 0) {
            /* "--seed-x" flag[6] == 'x'; "--seed-y" flag[6] == 'y' */
            int *dst = flag[6] == 'x' ? seed_x : seed_y;
            int *n = flag[6] == 'x' ? &n_seed_x : &n_seed_y;
            const char *c = val;
            while (*c && *n < 64) {
                dst[(*n)++] = (int)strtol(c, (char **)&c, 10);
                if (*c == ',')
                    c++;
            }
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!input || !out) {
        usage(argv[0]);
        return 2;
    }

    /* manifest values are the defaults; CLI flags override */
    char path[4096];
    snprintf(path, sizeof(path), "%s/params.json", input);
    JsonValue *manifest = json_parse_file(path);
    if (!manifest) {
        fprintf(stderr, "dorado_c: cannot read %s\n", path);
        return 1;
    }
    const JsonValue *v = json_get(manifest, "Np_tracer");
    if (particles < 0)
        particles = v ? (int)v->num_val : 0;
    v = json_get(manifest, "iterations");
    if (iterations < 0)
        iterations = v ? (int)v->num_val : 0;
    v = json_get(manifest, "seed");
    if (seed < 0)
        seed = v ? (long)v->num_val : 0;
    if (seed < 0 || seed >= 4294967296L) {
        fprintf(stderr, "dorado_c: seed must be in [0, 2**32)\n");
        return 2;
    }
    v = json_get(manifest, "model");
    char model[64] = "";
    if (v && v->kind == JSON_STRING)
        snprintf(model, sizeof(model), "%s", v->str_val);
    json_free(manifest);

    const int *sx, *sy;
    int nsx, nsy;
    if (n_seed_x && n_seed_y) {
        sx = seed_x;
        nsx = n_seed_x;
        sy = seed_y;
        nsy = n_seed_y;
    } else {
        default_seed_box(input, &sx, &nsx, &sy, &nsy);
    }

    /* ---- compute clock starts: init derivations, seeding, iterations ---- */
    double t0 = monotonic_seconds();
    Field *field = field_load(input);
    if (!field)
        return 1;
    mt_seed(&field->rng, (uint32_t)seed);
    Walk walk;
    walk_generate_particles(field, &walk, particles, sx, nsx, sy, nsy);
    for (int it = 0; it < iterations; it++)
        walk_run_iteration(field, &walk);
    double seconds = monotonic_seconds() - t0;
    /* ---- compute clock stopped ---- */

    long appended = 0;
    for (int i = 0; i < walk.n; i++)
        appended += walk.h.xh_n[i] - 1;

    mkdir_p(out);
    snprintf(path, sizeof(path), "%s/walk_data.json", out);
    write_walk_data(path, &walk);
    snprintf(path, sizeof(path), "%s/timing.json", out);
    write_timing(path, input, model, particles, iterations, (int)seed,
                 seconds, appended);

    printf("walk data: %s/walk_data.json\n", out);
    printf("timing: %s/timing.json\n", out);
    printf("compute seconds: %.6f (%ld appended steps, %d particles, "
           "%d iterations)\n", seconds, appended, particles, iterations);

    walk_free(&walk);
    field_free(field);
    return 0;
}
