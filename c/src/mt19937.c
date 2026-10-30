/* MT19937 implementation matching NumPy's legacy RandomState.
 * Draw order and tempering follow the reference MT19937; the legacy double
 * conversion mirrors numpy.random.random_sample (two words per double). */
#include "mt19937.h"

void mt_seed(Mt19937 *m, uint32_t seed)
{
    m->mt[0] = seed;
    for (int i = 1; i < 624; i++) {
        uint32_t prev = m->mt[i - 1];
        m->mt[i] = 1812433253U * (prev ^ (prev >> 30)) + (uint32_t)i;
    }
    m->idx = 624;
}

uint32_t mt_next_u32(Mt19937 *m)
{
    if (m->idx >= 624) {
        for (int i = 0; i < 624; i++) {
            uint32_t y = (m->mt[i] & 0x80000000U) | (m->mt[(i + 1) % 624] & 0x7fffffffU);
            uint32_t n = m->mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1U)
                n ^= 0x9908b0dfU;
            m->mt[i] = n;
        }
        m->idx = 0;
    }
    uint32_t y = m->mt[m->idx++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= y >> 18;
    return y;
}

double mt_next_double(Mt19937 *m)
{
    /* numpy legacy: (a >> 5) * 2^26 + (b >> 6), scaled by 2^-53. Each factor
     * is an exact integer in double (<= 2^53), so the arithmetic is exact
     * and the result is bit-identical to RandomState.random_sample. */
    uint32_t a = mt_next_u32(m) >> 5;
    uint32_t b = mt_next_u32(m) >> 6;
    return (double)(a * 67108864.0 + (double)b) * (1.0 / 9007199254740992.0);
}
