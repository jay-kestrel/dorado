/* make_weight parity: the reference's "shallow" weight test
 * (tests/test_lagrangian_walker.py::test_make_weight_shallow) — a wet
 * particle cell surrounded by dry cells must put all weight on the
 * current cell, and border cells must be cleared. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "field.h"

static int failures = 0;

int main(void)
{
    const int L = 5, W = 5, N = L * W;
    double *depth = calloc((size_t)N, sizeof(double));
    double *stage = calloc((size_t)N, sizeof(double));
    double *qx = calloc((size_t)N, sizeof(double));
    double *qy = calloc((size_t)N, sizeof(double));
    int64_t *cell_type = calloc((size_t)N, sizeof(int64_t));
    depth[1 * W + 1] = 10.0; /* only the particle cell is wet */

    Field *f = field_from_arrays(depth, stage, qx, qy, cell_type, L, W,
                                 1.0, 1.0, 0.02, 0.0, 0.1, true, false);
    if (!f) {
        printf("FAIL test_weight: field_from_arrays returned NULL\n");
        return 1;
    }
    /* the reference's weight tests set all distances to NaN */
    field_set_nan_distances(f);
    field_make_weight(f);

    const double *w11 = &f->weight[((size_t)1 * W + 1) * 9];
    if (w11[4] != 1.0) {
        printf("FAIL weight[1,1,4] = %.17g, want 1.0\n", w11[4]);
        failures++;
    }
    double sum = 0.0;
    for (int k = 0; k < 9; k++)
        sum += w11[k];
    if (sum != 1.0) {
        printf("FAIL sum(weight[1,1,:]) = %.17g, want 1.0\n", sum);
        failures++;
    }
    /* border cells: cleared (all nine entries zero) */
    for (int j = 0; j < W; j++) {
        for (int k = 0; k < 9; k++) {
            double w = f->weight[((size_t)0 * W + j) * 9 + k];
            if (w != 0.0) {
                printf("FAIL weight[0,%d,%d] = %g, want 0\n", j, k, w);
                failures++;
            }
        }
    }
    for (int j = 0; j < W; j++)
        for (int k = 0; k < 9; k++) {
            double w = f->weight[((size_t)(L - 1) * W + j) * 9 + k];
            if (w != 0.0) {
                printf("FAIL weight[%d,%d,%d] = %g, want 0\n", L - 1, j, k, w);
                failures++;
            }
        }
    for (int i = 0; i < L; i++)
        for (int k = 0; k < 9; k++) {
            double w = f->weight[((size_t)i * W + 0) * 9 + k];
            if (w != 0.0) {
                printf("FAIL weight[%d,0,%d] = %g, want 0\n", i, k, w);
                failures++;
            }
            w = f->weight[((size_t)i * W + (W - 1)) * 9 + k];
            if (w != 0.0) {
                printf("FAIL weight[%d,%d,%d] = %g, want 0\n", i, W - 1, k, w);
                failures++;
            }
        }

    field_free(f);
    if (failures) {
        printf("test_weight: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_weight: ok\n");
    return 0;
}
