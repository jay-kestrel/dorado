//! MT19937 with NumPy's legacy `RandomState` semantics.
//!
//! Parity contract (spec "RNG parity"): the ports reproduce the reference's
//! MT19937 stream (init_genrand seeding) and NumPy's legacy double conversion
//! — two 32-bit words per double, `(a >> 5) * 2^26 + (b >> 6)` over `2^53` —
//! in the reference's exact serial draw order.
//!
//! The generator mirrors NumPy's `mt19937.c` (the original Matsumoto &
//! Nishimura algorithm): in-place two-loop twist over 624 words, tempering,
//! and `rk_double`-style sampling. `uniform(low, high)` follows the legacy
//! `RandomState.uniform` form `low + (high - low) * random_sample()`.

/// State size of the MT19937 generator (words).
const N: usize = 624;
/// Offset in the twist step.
const M: usize = 397;
/// Coefficient of the twist rational recurrence.
const MATRIX_A: u32 = 0x9908_b0df;
/// High bit of the twist mixing word.
const UPPER_MASK: u32 = 0x8000_0000;
/// Low 31 bits of the twist mixing word.
const LOWER_MASK: u32 = 0x7fff_ffff;

/// MT19937 generator seeded the way `np.random.seed(int)` seeds it
/// (`init_genrand` on the seed's low 32 bits).
pub struct Mt19937 {
    state: [u32; N],
    /// Words remaining in the cached block; `N` forces a twist before the
    /// first draw, matching NumPy's freshly-seeded state.
    mti: usize,
}

impl Mt19937 {
    /// Seed exactly as NumPy's `RandomState(seed)` does for a non-negative
    /// integer seed: `init_genrand(seed & 0xffffffff)`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Mt19937 {
            state: [0; N],
            mti: N,
        };
        rng.seed_genrand((seed & 0xffff_ffff) as u32);
        rng
    }

    /// `init_genrand` from the reference MT19937 definition.
    pub fn seed_genrand(&mut self, s: u32) {
        self.state[0] = s;
        for i in 1..N {
            let prev = self.state[i - 1];
            self.state[i] = 1812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    /// Twist the 624-word state in place (NumPy's `mt19937_gen`).
    fn generate(&mut self) {
        let mag01 = [0u32, MATRIX_A];
        for kk in 0..(N - M) {
            let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
            self.state[kk] = self.state[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        for kk in (N - M)..(N - 1) {
            let y = (self.state[kk] & UPPER_MASK) | (self.state[kk + 1] & LOWER_MASK);
            self.state[kk] =
                self.state[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        let y = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
        self.state[N - 1] = self.state[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
    }

    /// Next tempered 32-bit word (NumPy's `mt19937_next`).
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            self.generate();
            self.mti = 0;
        }
        let mut y = self.state[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// NumPy's legacy `random_sample()` double: two words per sample,
    /// `(a >> 5) * 2^26 + (b >> 6)` over `2^53`.
    ///
    /// Both multiplications are exact powers of two, so the result is the
    /// correctly-rounded IEEE-754 value — bit-identical to NumPy's C.
    #[allow(clippy::did_you_mean)] // name mirrors numpy.random.random_sample
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// Legacy `RandomState.uniform(low, high)`:
    /// `low + (high - low) * random_sample()`.
    ///
    /// Always consumes one sample — including `high == low` (the all-zero
    /// weight-row case returns `low` but still advances the stream, exactly
    /// like the reference).
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchor: `np.random.seed(0)`, then the first ten `random_sample`
    /// values, verified against NumPy 2.3.5 in the sandbox (2026-10-30).
    #[test]
    fn first_ten_doubles_match_numpy_seed_zero() {
        let expected = [
            0.5488_1350_3927_3248_f64,
            0.7151_8936_6372_4195,
            0.6027_6337_6071_6439,
            0.5448_8318_2996_8969,
            0.4236_5479_9338_9047,
            0.6458_9411_3066_6561,
            0.4375_8721_1262_6925,
            0.8917_7300_0782_0798,
            0.9636_6276_0501_0293,
            0.3834_4151_8825_7777,
        ];
        let mut rng = Mt19937::new(0);
        for &value in &expected {
            assert_eq!(rng.random_sample(), value);
        }
    }

    /// Anchor: the first `uniform` draw at seed 0 (spec parity anchor).
    #[test]
    fn first_uniform_matches_spec_anchor() {
        let mut rng = Mt19937::new(0);
        assert_eq!(rng.uniform(0.0, 1.0), 0.5488_1350_3927_3248);
    }

    /// Anchor: seeding draws for the rcm benchmark workload —
    /// `uniform(0, len(seed_x))` then `uniform(0, len(seed_y))` per the
    /// reference's two-pass `generate_particles` order (verified values from
    /// NumPy 2.3.5, seed 0).
    #[test]
    fn seeding_draw_sequence_matches_numpy() {
        let mut rng = Mt19937::new(0);
        let x_draws: Vec<f64> = (0..4).map(|_| rng.uniform(0.0, 2.0)).collect();
        let y_draws: Vec<f64> = (0..4).map(|_| rng.uniform(0.0, 3.0)).collect();
        let expected_x = [
            1.0976_2700_7854_6495,
            1.4303_7873_2744_839,
            1.2055_2675_2143_2877,
            1.0897_6636_5993_7937,
        ];
        let expected_y = [
            1.2709_6439_8016_714,
            1.9376_8233_9199_9682,
            1.3127_6163_3788_0776,
            2.6753_1900_2346_239,
        ];
        for (got, want) in x_draws.iter().zip(expected_x) {
            assert_eq!(got, &want);
        }
        for (got, want) in y_draws.iter().zip(expected_y) {
            assert_eq!(got, &want);
        }
    }

    /// Anchor: the first weight-pick draw of the rcm fixture run — after the
    /// eight seeding draws, `uniform(0, 0.2302514329470293)` must equal
    /// `0.2218847314830519` (verified against NumPy 2.3.5).
    #[test]
    fn uniform_consumes_stream_even_when_high_is_low() {
        let mut rng = Mt19937::new(0);
        for _ in 0..8 {
            rng.random_sample();
        }
        let draw = rng.uniform(0.0, 0.2302_5143_2947_0293);
        assert_eq!(draw, 0.2218_8473_1483_0519);

        // uniform(0, 0) still consumes one sample and returns 0.
        let mut rng = Mt19937::new(0);
        assert_eq!(rng.uniform(0.0, 0.0), 0.0);
        assert_eq!(rng.random_sample(), 0.7151_8936_6372_4195);
    }

    /// Same seed reproduces the same state.
    #[test]
    fn same_seed_reproduces_stream() {
        let mut a = Mt19937::new(42);
        let mut b = Mt19937::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }
}
