#include "walker.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "walker: out of memory (%zu bytes)\n", n);
        exit(1);
    }
    return p;
}

int walker_searchsorted_left(const double *cutoffs, int n, double u)
{
    for (int i = 0; i < n; i++)
        if (cutoffs[i] >= u)
            return i;
    return n;
}

/* np.random.uniform(0, high): one random_sample draw scaled to [0, high).
 * high <= 0 still consumes the draw and yields 0. */
static double uniform_draw(Mt19937 *rng, double high)
{
    return (high > 0.0 ? high : 0.0) * mt_next_double(rng);
}

int walker_random_pick(Field *f, const double *probs)
{
    double p[9], cutoffs[9];
    double sum = 0.0;
    for (int k = 0; k < 9; k++) {
        p[k] = isnan(probs[k]) ? 0.0 : probs[k];
        sum += p[k];
        cutoffs[k] = sum;
    }
    double u = uniform_draw(&f->rng, cutoffs[8]);
    return walker_searchsorted_left(cutoffs, 9, u);
}

int walker_steep_descent(Field *f, const double *probs)
{
    double p[9], cutoffs[9];
    double sum = 0.0;
    for (int k = 0; k < 9; k++)
        p[k] = isnan(probs[k]) ? 0.0 : probs[k];
    p[4] = 0.0; /* the current cell never competes */
    bool any = false;
    double max_val = 0.0;
    for (int k = 0; k < 9; k++) {
        if (p[k] > 0.0) {
            any = true;
            if (p[k] > max_val)
                max_val = p[k];
        }
    }
    if (!any)
        return 4; /* dead end: no draw consumed */
    for (int k = 0; k < 9; k++) {
        if (p[k] < max_val)
            p[k] = 0.0;
        sum += p[k];
        cutoffs[k] = sum;
    }
    double u = uniform_draw(&f->rng, cutoffs[8]);
    return walker_searchsorted_left(cutoffs, 9, u);
}

/* ---------------- seeding ---------------- */

static int pick_seed(Field *f, const int *choices, int n_choices)
{
    /* random_pick_seed: probs = ones, cutoffs = [1..n] */
    double *cutoffs = xmalloc((size_t)n_choices * sizeof(double));
    for (int k = 0; k < n_choices; k++)
        cutoffs[k] = (double)(k + 1);
    double u = uniform_draw(&f->rng, cutoffs[n_choices - 1]);
    int idx = walker_searchsorted_left(cutoffs, n_choices, u);
    free(cutoffs);
    return choices[idx];
}

void walk_alloc(Field *f, Walk *w, int n)
{
    (void)f;
    memset(w, 0, sizeof(*w));
    w->n = n;
    w->stranding_dt = 1.0; /* Particles default (params.stranding_dt absent) */
    w->x = xmalloc((size_t)n * sizeof(int));
    w->y = xmalloc((size_t)n * sizeof(int));
    w->stranded = xmalloc((size_t)n * sizeof(int));
    w->stime = xmalloc((size_t)n * sizeof(double));
    w->stime_is_float = xmalloc((size_t)n * sizeof(bool));

    w->h.xh = xmalloc((size_t)n * sizeof(int *));
    w->h.yh = xmalloc((size_t)n * sizeof(int *));
    w->h.th = xmalloc((size_t)n * sizeof(JNum *));
    w->h.sh = xmalloc((size_t)n * sizeof(int *));
    w->h.sth = xmalloc((size_t)n * sizeof(JNum *));
    w->h.xh_n = xmalloc((size_t)n * sizeof(int));
    w->h.xh_cap = xmalloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int cap = 8;
        w->h.xh[i] = xmalloc((size_t)cap * sizeof(int));
        w->h.yh[i] = xmalloc((size_t)cap * sizeof(int));
        w->h.th[i] = xmalloc((size_t)cap * sizeof(JNum));
        w->h.sh[i] = xmalloc((size_t)cap * sizeof(int));
        w->h.sth[i] = xmalloc((size_t)cap * sizeof(JNum));
        w->h.xh_cap[i] = cap;
        w->h.xh_n[i] = 0;
    }
}

static void hist_append(Walk *w, int i, int x, int y, JNum t, int s, JNum st)
{
    if (w->h.xh_n[i] + 1 > w->h.xh_cap[i]) {
        int cap = w->h.xh_cap[i] * 2;
        w->h.xh[i] = realloc(w->h.xh[i], (size_t)cap * sizeof(int));
        w->h.yh[i] = realloc(w->h.yh[i], (size_t)cap * sizeof(int));
        w->h.th[i] = realloc(w->h.th[i], (size_t)cap * sizeof(JNum));
        w->h.sh[i] = realloc(w->h.sh[i], (size_t)cap * sizeof(int));
        w->h.sth[i] = realloc(w->h.sth[i], (size_t)cap * sizeof(JNum));
        if (!w->h.xh[i] || !w->h.yh[i] || !w->h.th[i] || !w->h.sh[i] ||
            !w->h.sth[i]) {
            fprintf(stderr, "walker: out of memory growing history\n");
            exit(1);
        }
        w->h.xh_cap[i] = cap;
    }
    int p = w->h.xh_n[i]++;
    w->h.xh[i][p] = x;
    w->h.yh[i][p] = y;
    w->h.th[i][p] = t;
    w->h.sh[i][p] = s;
    w->h.sth[i][p] = st;
}

void walk_generate_particles(Field *f, Walk *w, int n, const int *seed_x,
                             int n_seed_x, const int *seed_y, int n_seed_y)
{
    walk_alloc(f, w, n);
    /* all x picks, then all y picks — the reference's two list
     * comprehensions, in this order */
    for (int i = 0; i < n; i++)
        w->x[i] = pick_seed(f, seed_x, n_seed_x);
    for (int i = 0; i < n; i++)
        w->y[i] = pick_seed(f, seed_y, n_seed_y);
    for (int i = 0; i < n; i++) {
        w->stranded[i] = 0;
        w->stime[i] = 0.0;
        w->stime_is_float[i] = false;
        JNum t = { 0.0, true };
        hist_append(w, i, w->x[i], w->y[i], t, 0, (JNum){ 0.0, true });
    }
}

void walk_free(Walk *w)
{
    if (!w || !w->x)
        return;
    for (int i = 0; i < w->n; i++) {
        free(w->h.xh[i]);
        free(w->h.yh[i]);
        free(w->h.th[i]);
        free(w->h.sh[i]);
        free(w->h.sth[i]);
    }
    free(w->h.xh);
    free(w->h.yh);
    free(w->h.th);
    free(w->h.sh);
    free(w->h.sth);
    free(w->h.xh_n);
    free(w->h.xh_cap);
    free(w->x);
    free(w->y);
    free(w->stranded);
    free(w->stime);
    free(w->stime_is_float);
    memset(w, 0, sizeof(*w));
}

/* ---------------- stepping ---------------- */

static int get_weight(Field *f, int x, int y)
{
    const double *probs = &f->weight[((size_t)x * f->W + y) * 9];
    return f->steepest_descent ? walker_steep_descent(f, probs)
                               : walker_random_pick(f, probs);
}

double walker_calc_travel_times(Field *f, int new_cell, int x, int y,
                                int nx, int ny, double dist)
{
    if (x == nx && y == ny)
        return 0.0; /* particle did not move */
    double old_vel = f->velocity[(size_t)x * f->W + y];
    double new_vel = f->velocity[(size_t)nx * f->W + ny];
    double diff = 0.0;
    if (f->diff_coeff > 0.0)
        diff = (0.5 - mt_next_double(&f->rng)) * f->diff_coeff;
    double projected_dist =
        dist * fmax(0.0, cos(f->angles[new_cell] -
                             f->velocity_angle[(size_t)x * f->W + y]));
    return 0.5 * projected_dist * (1.0 / old_vel + 1.0 / new_vel) * (1.0 + diff);
}

void walk_run_iteration(Field *f, Walk *w)
{
    int n = w->n;
    int L = f->L, W = f->W;

    /* per-particle step results (mirrors the reference's intermediate
     * lists: park decisions first, then picks, then travel times) */
    bool *park = xmalloc((size_t)n * sizeof(bool));
    double *stime_out = xmalloc((size_t)n * sizeof(double));
    bool *stime_float_out = xmalloc((size_t)n * sizeof(bool));
    int *new_cell = xmalloc((size_t)n * sizeof(int));
    int *nx = xmalloc((size_t)n * sizeof(int));
    int *ny = xmalloc((size_t)n * sizeof(int));
    double *dist = xmalloc((size_t)n * sizeof(double));

    /* pass 0: park-while-dry decisions (no draws). This port supports the
     * default stranding_release = 'immediate' only: a wet particle is
     * always free to move, so park == !wet. */
    for (int i = 0; i < n; i++) {
        int x = w->x[i], y = w->y[i];
        double stranded_time = w->stime[i];
        bool wet = f->depth[(size_t)x * W + y] > f->dry_depth &&
                   f->cell_type[(size_t)x * W + y] != 2;
        bool p = !wet;
        park[i] = p;
        if (p) {
            stime_out[i] = stranded_time + w->stranding_dt;
            stime_float_out[i] = true;
        } else {
            stime_out[i] = stranded_time;
            stime_float_out[i] = w->stime_is_float[i];
        }
    }

    /* pass 1: pick a new cell for every non-parked particle away from
     * (0,0); parked and (0,0) particles take cell 4 without drawing */
    for (int i = 0; i < n; i++) {
        int x = w->x[i], y = w->y[i];
        if ((x != 0 || y != 0) && !park[i])
            new_cell[i] = get_weight(f, x, y);
        else
            new_cell[i] = 4;
    }

    /* pass 2: indices, distances, boundary/out-of-bounds freezes */
    for (int i = 0; i < n; i++) {
        int x = w->x[i], y = w->y[i];
        if (new_cell[i] != 4) {
            nx[i] = x + f->jwalk[new_cell[i]];
            ny[i] = y + f->iwalk[new_cell[i]];
        } else {
            nx[i] = x;
            ny[i] = y;
        }
        dist[i] = f->distances[new_cell[i]] * f->dx;
        if (dist[i] == 0.0) {
            nx[i] = x;
            ny[i] = y;
        }
        /* check_for_boundary: freeze when the *current* neighborhood has
         * an edge cell; slices are clamped at 0 (current master) */
        bool boundary = false;
        int i0 = x - 1 > 0 ? x - 1 : 0;
        int i1 = x + 1 < L - 1 ? x + 1 : L - 1;
        int j0 = y - 1 > 0 ? y - 1 : 0;
        int j1 = y + 1 < W - 1 ? y + 1 : W - 1;
        for (int ii = i0; ii <= i1 && !boundary; ii++)
            for (int jj = j0; jj <= j1; jj++)
                if (f->cell_type[(size_t)ii * W + jj] == -1) {
                    boundary = true;
                    break;
                }
        if (boundary || nx[i] < 0 || nx[i] >= L || ny[i] < 0 || ny[i] >= W) {
            nx[i] = x;
            ny[i] = y;
        }
    }

    /* pass 3: travel times, in list order (diffusion draws here) */
    double *trav = xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (park[i]) {
            trav[i] = w->stranding_dt;
        } else {
            trav[i] = walker_calc_travel_times(f, new_cell[i], w->x[i], w->y[i],
                                        nx[i], ny[i], dist[i]);
        }
    }

    /* append when the particle moved or parked; travel time is the
     * cumulative value, stranded_time keeps its numeric type */
    for (int i = 0; i < n; i++) {
        bool moved = nx[i] != w->x[i] || ny[i] != w->y[i];
        if (moved || park[i]) {
            double t_new = (w->h.th[i][w->h.xh_n[i] - 1].v) + trav[i];
            JNum t = { t_new, false };
            /* JNum.is_int is the *rendering* flag (true -> "0" style);
             * stime_float_out is the *state* flag (true -> value is a
             * float), so invert when storing */
            JNum st = { stime_out[i], !stime_float_out[i] };
            hist_append(w, i, nx[i], ny[i], t, park[i] ? 1 : 0, st);
            w->x[i] = nx[i];
            w->y[i] = ny[i];
            w->stranded[i] = park[i] ? 1 : 0;
            w->stime[i] = stime_out[i];
            w->stime_is_float[i] = stime_float_out[i];
        }
    }

    free(park);
    free(stime_out);
    free(stime_float_out);
    free(new_cell);
    free(nx);
    free(ny);
    free(dist);
    free(trav);
}
