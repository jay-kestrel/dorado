/* The Lagrangian walker: seeding, per-particle stepping, and the
 * steady-mode iteration, ported 1:1 from dorado.lagrangian_walker /
 * dorado.particle_track (current master).
 *
 * Draw-order contract (spec "RNG parity"): MT19937 legacy doubles; every
 * draw is one mt_next_double(). Steady mode consumes, in list-index order:
 * pass-1 pick draws for all non-parked particles away from (0,0), then
 * pass-2 diffusion draws for all moved particles (when diff_coeff > 0).
 * Parked particles and particles at (0,0) consume nothing in pass 1.
 */
#ifndef DORADO_C_WALKER_H
#define DORADO_C_WALKER_H

#include <stdbool.h>
#include <stddef.h>

#include "field.h"

/* A number that remembers whether it was written as an int or a float:
 * walk_data mixes them (initial times are int 0; appended times are
 * floats; stranded_time keeps its type when a particle moves freely).
 * The JSON writer preserves the distinction so the output diffs cleanly
 * against the reference's json.dumps output. */
typedef struct {
    double v;
    bool is_int;
} JNum;

/* One growable history per particle per walk-data field. */
typedef struct {
    int **xh, **yh;        /* positions */
    JNum **th;             /* travel times */
    int **sh;              /* stranded flags */
    JNum **sth;            /* stranded times */
    int *xh_n, *xh_cap;    /* one shared length per particle for all five */
} WalkHist;

typedef struct {
    int n;
    int *x, *y;            /* current positions */
    int *stranded;         /* current flag */
    double *stime;         /* current stranded time */
    bool *stime_is_float;
    double stranding_dt;
    WalkHist h;
} Walk;

/* Seed Np particles: Np x-picks then Np y-picks (generate_particles
 * 'random' method, seed_time 0). seed_x/seed_y are the candidate lists;
 * n_seed_x/n_seed_y their lengths. */
void walk_generate_particles(Field *f, Walk *w, int n, const int *seed_x,
                             int n_seed_x, const int *seed_y, int n_seed_y);

/* One steady-mode run_iteration: step + append. */
void walk_run_iteration(Field *f, Walk *w);

/* calc_travel_times parity (exposed for unit tests): zero when the
 * particle did not move, else 0.5 * projected_dist * harmonic-mean
 * velocity factor * (1 + diffusion draw). Consumes one draw only when
 * diff_coeff > 0. */
double walker_calc_travel_times(Field *f, int new_cell, int x, int y,
                                int nx, int ny, double dist);

void walk_alloc(Field *f, Walk *w, int n);
void walk_free(Walk *w);

/* numpy searchsorted(cutoffs, u, side='left'): first index with
 * cutoffs[idx] >= u (n when u exceeds all). Shared by the pick functions. */
int walker_searchsorted_left(const double *cutoffs, int n, double u);

/* random_pick parity (exposed for unit tests): copy of probs with NaN -> 0,
 * cumsum, uniform draw, searchsorted. */
int walker_random_pick(Field *f, const double *probs);

/* steep_descent parity: NaN -> 0, index 4 excluded, dead end -> 4 (no
 * draw), else a uniform draw among the max-value cells. */
int walker_steep_descent(Field *f, const double *probs);

#endif /* DORADO_C_WALKER_H */
