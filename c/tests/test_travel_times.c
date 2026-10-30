/* calc_travel_times parity: the reference's single-particle travel-time
 * test (tests/test_lagrangian_walker.py::test_calc_travel_times) — hardset
 * velocities/angles, a step from (1,1) to (0,1) with dist 1 must give
 * 0.5609806565385976 (rel 1e-6). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "field.h"
#include "walker.h"

static int failures = 0;

int main(void)
{
    const int L = 5, W = 5, N = L * W;
    double *depth = malloc((size_t)N * sizeof(double));
    double *stage = malloc((size_t)N * sizeof(double));
    double *qx = calloc((size_t)N, sizeof(double));
    double *qy = malloc((size_t)N * sizeof(double));
    int64_t *cell_type = calloc((size_t)N, sizeof(int64_t));
    for (int i = 0; i < N; i++) {
        depth[i] = 1.0;
        stage[i] = 1.0;
        qy[i] = 1.0;
    }

    Field *f = field_from_arrays(depth, stage, qx, qy, cell_type, L, W,
                                 1.0, 1.0, 0.02, 0.0, 0.1, false, false);
    if (!f) {
        printf("FAIL test_travel_times: field_from_arrays returned NULL\n");
        return 1;
    }
    /* hardset like the reference test: velocity 1 everywhere except
     * velocity[0, 0:2] = 3; velocity_angle 1 everywhere */
    for (int i = 0; i < N; i++) {
        f->velocity[i] = 1.0;
        f->velocity_angle[i] = 1.0;
    }
    f->velocity[0 * W + 0] = 3.0;
    f->velocity[0 * W + 1] = 3.0;

    /* the diff_coeff = 0 path: no draw, no diffusion factor */
    double trav = walker_calc_travel_times(f, 1, 1, 1, 0, 1, 1.0);
    const double want = 0.5609806565385976;
    double rel = fabs(trav - want) / fabs(want);
    if (rel > 1e-6) {
        printf("FAIL travel time = %.17g, want %.17g (rel %.3g)\n", trav,
               want, rel);
        failures++;
    }

    /* a particle that did not move books zero time and consumes no draw */
    double zero = walker_calc_travel_times(f, 4, 1, 1, 1, 1, 0.0);
    if (zero != 0.0) {
        printf("FAIL non-moving travel time = %g, want 0\n", zero);
        failures++;
    }

    field_free(f);
    if (failures) {
        printf("test_travel_times: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_travel_times: ok\n");
    return 0;
}
