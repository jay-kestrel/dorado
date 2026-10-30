/* Field state and the Particles-init derivations, ported 1:1.
 *
 * Mirrors dorado.particle_track.Particles.__init__ (DeltaRCM and
 * non-DeltaRCM branches) and dorado.lagrangian_walker.make_weight for the
 * benchmark path: float64 grid fields, default stranding handling ('park',
 * 'immediate'), no per-particle attributes, no roi_grid. Operation order
 * and dtype promotions follow the Python reference exactly — the parity
 * gates compare against golden values, so nothing here is "improved".
 */
#ifndef DORADO_C_FIELD_H
#define DORADO_C_FIELD_H

#include <stdbool.h>
#include <stdint.h>

#include "mt19937.h"

typedef struct {
    int L, W;              /* grid rows (x) and columns (y) */
    double dx, theta, gamma, diff_coeff, dry_depth;
    bool steepest_descent;
    bool delta_rcm;        /* model == "DeltaRCM" (else the axis-swap branch) */

    /* D8 tables. distances/ivec/jvec are fields (not constants) because the
     * reference's unit tests override them (e.g. all-NaN distances); the
     * driver always uses the D8 constants. Row-major 3x3, flat index k =
     * 0..8 ordered NW,N,NE,W,0,E,SW,S,SE like np.ravel(). */
    double distances[9];
    double ivec[9], jvec[9];
    const int *iwalk;      /* static tables below */
    const int *jwalk;
    const double *angles;

    /* Grid arrays, all L*W row-major (weight is L*W*9). */
    double *depth, *stage, *qx, *qy, *u, *v, *velocity, *velocity_angle;
    int64_t *cell_type;
    double *weight;

    Mt19937 rng;
} Field;

/* Load grid arrays from a fixture directory (params.json + npy files named
 * by its "arrays" map) and run the Particles-init derivations. Returns
 * NULL on I/O or validation failure (diagnostic already printed). */
Field *field_load(const char *fixture_dir);

/* Run the derivations on already-loaded arrays (test path). Takes ownership
 * of the six malloc'd L*W arrays; cell_type must be malloc'd too. */
Field *field_from_arrays(double *depth, double *stage, double *qx, double *qy,
                         int64_t *cell_type, int L, int W, double dx,
                         double theta, double gamma, double diff_coeff,
                         double dry_depth, bool steepest_descent,
                         bool delta_rcm);

void field_free(Field *f);

/* make_weight parity: builds f->weight (L*W*9) from the derived fields. */
void field_make_weight(Field *f);

/* The all-NaN distances table used by the reference's weight unit tests. */
void field_set_nan_distances(Field *f);

#endif /* DORADO_C_FIELD_H */
