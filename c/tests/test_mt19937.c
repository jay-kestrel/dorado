/* MT19937 parity tests: NumPy legacy RandomState anchors. */
#include <stdio.h>
#include <stdlib.h>

#include "mt19937.h"

static int failures = 0;

static void check_stream(const char *label, uint32_t seed, int n,
                         const double *expected)
{
    Mt19937 m;
    mt_seed(&m, seed);
    for (int i = 0; i < n; i++) {
        double got = mt_next_double(&m);
        if (got != expected[i]) {
            printf("FAIL %s: draw %d = %.17g, want %.17g\n", label, i, got,
                   expected[i]);
            failures++;
        }
    }
}

int main(void)
{
    /* Anchors from np.random (legacy RandomState), probed in the repo
     * sandbox with numpy 2.x this session:
     *   np.random.seed(0); np.random.random() -> 0.5488135039273248
     *   second draw                            -> 0.7151893663724195
     *   np.random.seed(42); first draw         -> 0.3745401188473625
     */
    double seed0[3] = { 0.5488135039273248, 0.7151893663724195,
                        0.6027633760716439 };
    check_stream("seed 0", 0, 3, seed0);
    double seed42[2] = { 0.3745401188473625, 0.9507143064099162 };
    check_stream("seed 42", 42, 2, seed42);

    /* init_genrand seeding with a large seed (numpy accepts any 32-bit);
     * anchor verified against live numpy this session */
    Mt19937 m;
    mt_seed(&m, 4294967295U);
    double big = mt_next_double(&m);
    if (big != 0.0976320289940138) {
        printf("FAIL seed 2**32-1 first double: %.17g, want 0.0976320289940138\n",
               big);
        failures++;
    }

    if (failures) {
        printf("test_mt19937: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_mt19937: ok\n");
    return 0;
}
