/* MT19937 with NumPy legacy RandomState semantics.
 *
 * The Python reference draws from the global `np.random` RandomState
 * (seeded by the caller). Parity (spec "RNG parity contract") requires the
 * same stream: MT19937 via init_genrand seeding, and doubles formed from
 * two 32-bit words as ((a >> 5) * 2^26 + (b >> 6)) / 2^53.
 *
 * Anchor: seed 0 -> first double == 0.5488135039273248 (verified against
 * numpy 2.x `np.random.seed(0); np.random.random()` in this repo's sandbox).
 */
#ifndef DORADO_C_MT19937_H
#define DORADO_C_MT19937_H

#include <stdint.h>

typedef struct {
    uint32_t mt[624];
    int idx;
} Mt19937;

/* init_genrand: legacy RandomState seeding for a scalar seed. */
void mt_seed(Mt19937 *m, uint32_t seed);

/* One tempered 32-bit word. */
uint32_t mt_next_u32(Mt19937 *m);

/* One legacy double: two words, (a>>5)*2^26 + (b>>6) over 2^53. */
double mt_next_double(Mt19937 *m);

#endif /* DORADO_C_MT19937_H */
