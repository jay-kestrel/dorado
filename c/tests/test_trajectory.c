/* Seeded-trajectory parity: the reference's RCM example case
 * (tests/test_examplecases.py::TestRCM::test_few_steps_RCM) — one particle
 * seeded at (16, 140) on the normalized 120x240 fixture, 3 iterations at
 * seed 0, must follow xinds [16, 17, 18, 19] with travel times matching
 * the pinned values at rel 1e-6. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "field.h"
#include "walker.h"

#ifndef DORADO_REPO_ROOT
#define DORADO_REPO_ROOT "../.."
#endif

static int failures = 0;

static void expect_rel(const char *what, double got, double want)
{
    double rel = fabs(got - want) / fabs(want);
    if (!(rel <= 1e-6) && !(got == want)) {
        printf("FAIL %s: got %.17g, want %.17g (rel %.3g)\n", what, got,
               want, rel);
        failures++;
    }
}

int main(void)
{
    const char *fixture = DORADO_REPO_ROOT "/bench/fixture_rcm";
    Field *f = field_load(fixture);
    if (!f) {
        printf("FAIL test_trajectory: cannot load %s\n", fixture);
        return 1;
    }
    mt_seed(&f->rng, 0);

    Walk w;
    const int seed_x[1] = { 16 }, seed_y[1] = { 140 };
    walk_generate_particles(f, &w, 1, seed_x, 1, seed_y, 1);

    for (int it = 0; it < 3; it++)
        walk_run_iteration(f, &w);

    const int want_x[4] = { 16, 17, 18, 19 };
    const int want_y[4] = { 140, 141, 141, 141 };
    const double want_t[4] = { 0.0, 7007593448.337235,
                               10439964733.462337, 13698473384.876724 };
    for (int p = 0; p < 4; p++) {
        if (w.h.xh[0][p] != want_x[p]) {
            printf("FAIL xinds[%d] = %d, want %d\n", p, w.h.xh[0][p],
                   want_x[p]);
            failures++;
        }
        if (w.h.yh[0][p] != want_y[p]) {
            printf("FAIL yinds[%d] = %d, want %d\n", p, w.h.yh[0][p],
                   want_y[p]);
            failures++;
        }
        if (p == 0) {
            if (w.h.th[0][0].v != 0.0 || !w.h.th[0][0].is_int) {
                printf("FAIL travel_times[0] = %g, want int 0\n",
                       w.h.th[0][0].v);
                failures++;
            }
        } else {
            expect_rel("travel_times", w.h.th[0][p].v, want_t[p]);
        }
    }

    walk_free(&w);
    field_free(f);
    if (failures) {
        printf("test_trajectory: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_trajectory: ok\n");
    return 0;
}
