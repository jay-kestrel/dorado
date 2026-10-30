//! Routing-weight raster construction — the port of
//! `dorado.particle_track.make_weight`.
//!
//! Slot order is the reference's flattened 3x3: `[NW, N, NE, W, 0, E, SW, S, SE]`
//! (row-major over the -1..=1 neighborhood). All arithmetic is float64 in the
//! reference's exact operation order — masked-out slots zero out, NaN slots
//! are skipped by the `nansum`-equivalent normalizer, and all-zero rows fall
//! back to the deepest neighbor under the `park` handling (parity quirks are
//! replicated, not fixed — see rust/README.md).

use crate::fields::Fields;

/// NumPy `np.nansum(a, axis=2)` for a length-9 slot axis: NaN slots read as
/// zero, then NumPy's pairwise summation over 9 elements
/// (eight running values, the ninth folded into the first, then four pair
/// sums and two combines).
pub fn nansum9(slots: &[f64; 9]) -> f64 {
    let b = [
        if slots[0].is_nan() { 0.0 } else { slots[0] },
        if slots[1].is_nan() { 0.0 } else { slots[1] },
        if slots[2].is_nan() { 0.0 } else { slots[2] },
        if slots[3].is_nan() { 0.0 } else { slots[3] },
        if slots[4].is_nan() { 0.0 } else { slots[4] },
        if slots[5].is_nan() { 0.0 } else { slots[5] },
        if slots[6].is_nan() { 0.0 } else { slots[6] },
        if slots[7].is_nan() { 0.0 } else { slots[7] },
        if slots[8].is_nan() { 0.0 } else { slots[8] },
    ];
    let mut r = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
    r[0] += b[8];
    ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
}

/// NumPy `np.max(a, axis=2)` for a length-9 slot axis: NaN propagates.
pub fn max9(slots: &[f64; 9]) -> f64 {
    let mut m = slots[0];
    for &s in &slots[1..] {
        if s.is_nan() || m.is_nan() {
            return f64::NAN;
        }
        if s > m {
            m = s;
        }
    }
    m
}

/// Build the weight raster in place. Mirrors `make_weight` block for block.
pub fn make_weight(fields: &mut Fields) {
    let (l, w) = (fields.stage.l, fields.stage.w);
    let weight_len = l * w * 9;
    let mut weight_sfc = vec![0.0f64; weight_len];
    let mut weight_int = vec![0.0f64; weight_len];

    // weight_sfc = (tile(stage) - bsw(stage)) / tile(distances), zeroed
    // where non-positive; border windows read as 0 (big_sliding_window)
    for x in 0..l {
        for y in 0..w {
            for k in 0..9 {
                let nb = bsw_value(&fields.stage, x, y, k);
                let diff = fields.stage.get(x, y) - nb;
                let slot = diff / fields.distances[k];
                let idx = (x * w + y) * 9 + k;
                if slot.is_nan() || slot > 0.0 {
                    weight_sfc[idx] = slot;
                }
            }
        }
    }

    // weight_int = (tile(qx)*tile(jvec) + tile(qy)*tile(ivec)) / tile(distances)
    for x in 0..l {
        for y in 0..w {
            for k in 0..9 {
                let idx = (x * w + y) * 9 + k;
                let slot = (fields.qx.get(x, y) * fields.jvec[k]
                    + fields.qy.get(x, y) * fields.ivec[k])
                    / fields.distances[k];
                if slot.is_nan() || slot > 0.0 {
                    weight_int[idx] = slot;
                }
            }
        }
    }

    clear_borders(&mut weight_sfc, l, w);
    clear_borders(&mut weight_int, l, w);

    // zero out dry and land cells (depth_ind <= dry_depth | ct_ind == 2)
    apply_dry_land_mask(&mut weight_sfc, fields);
    apply_dry_land_mask(&mut weight_int, fields);

    // normalize the surface weights by their cell sums where positive
    for x in 0..l {
        for y in 0..w {
            let mut slots = [0.0f64; 9];
            for (k, slot) in slots.iter_mut().enumerate() {
                *slot = weight_sfc[(x * w + y) * 9 + k];
            }
            let norm = nansum9(&slots);
            if norm > 0.0 {
                let base = (x * w + y) * 9;
                for slot in &mut weight_sfc[base..base + 9] {
                    *slot /= norm;
                }
            }
        }
    }

    // combine: weight = gamma*weight_sfc + (1-gamma)*weight_int
    let mut weight = vec![0.0f64; weight_len];
    for i in 0..weight_len {
        weight[i] = fields.gamma * weight_sfc[i] + (1.0 - fields.gamma) * weight_int[i];
    }

    // weight = depth_ind**theta * weight
    for x in 0..l {
        for y in 0..w {
            let base = (x * w + y) * 9;
            for (k, slot) in weight[base..base + 9].iter_mut().enumerate() {
                let depth_ind = bsw_value(&fields.depth, x, y, k);
                *slot *= depth_ind.powf(fields.theta);
            }
        }
    }

    apply_dry_land_mask(&mut weight, fields);

    // all-zero (or all-NaN) rows pick the deepest neighbor, park-aware
    for x in 0..l {
        for y in 0..w {
            let mut depth_slots = [0.0f64; 9];
            let mut weight_slots = [0.0f64; 9];
            let mut ct_slots = [0.0f64; 9];
            for k in 0..9 {
                let idx = (x * w + y) * 9 + k;
                depth_slots[k] = bsw_value(&fields.depth, x, y, k);
                weight_slots[k] = weight[idx];
                ct_slots[k] = bsw_value(&fields.cell_type, x, y, k);
            }
            let zero_weights = nansum9(&weight_slots) <= 0.0;
            if !zero_weights {
                continue;
            }
            let mut deepest = [false; 9];
            let any_not_land = ct_slots.iter().any(|&c| c != 2.0);
            if fields.stranding_handling == crate::fields::StrandingHandling::Park && any_not_land {
                // reference: deepest where cell_type != 2 — the land-only
                // rows have no say while a non-land neighbor exists
                let mut masked = [0.0f64; 9];
                for k in 0..9 {
                    masked[k] = if ct_slots[k] != 2.0 {
                        depth_slots[k]
                    } else {
                        f64::NEG_INFINITY
                    };
                }
                let max_masked = max9(&masked);
                for k in 0..9 {
                    deepest[k] = depth_slots[k] == max_masked && ct_slots[k] != 2.0;
                }
            } else {
                let max_depth = max9(&depth_slots);
                for k in 0..9 {
                    deepest[k] = depth_slots[k] == max_depth;
                }
            }
            for k in 0..9 {
                if deepest[k] {
                    weight[(x * w + y) * 9 + k] = 1.0;
                }
            }
        }
    }

    clear_borders(&mut weight, l, w);
    fields.weight = weight;
}

/// `big_sliding_window` read: interior windows see the shifted neighbor,
/// border windows read 0.0 (the reference fills a zeros canvas).
#[inline]
fn bsw_value(grid: &crate::fields::Grid, x: usize, y: usize, k: usize) -> f64 {
    if x == 0 || x == grid.l - 1 || y == 0 || y == grid.w - 1 {
        return 0.0;
    }
    grid.get(x - 1 + k / 3, y - 1 + k % 3)
}

fn clear_borders(weight: &mut [f64], l: usize, w: usize) {
    for x in 0..l {
        for y in 0..w {
            if x == 0 || x == l - 1 || y == 0 || y == w - 1 {
                for k in 0..9 {
                    weight[(x * w + y) * 9 + k] = 0.0;
                }
            }
        }
    }
}

fn apply_dry_land_mask(weight: &mut [f64], fields: &Fields) {
    let (l, w) = (fields.stage.l, fields.stage.w);
    for x in 0..l {
        for y in 0..w {
            for k in 0..9 {
                let idx = (x * w + y) * 9 + k;
                let depth_ind = bsw_value(&fields.depth, x, y, k);
                let ct_ind = bsw_value(&fields.cell_type, x, y, k);
                if depth_ind <= fields.dry_depth || ct_ind == 2.0 {
                    weight[idx] = 0.0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{D8Tables, Fields, Model, Params, ReleaseRule, StrandingHandling};

    /// Mirror of the repo-test setup: `distances*np.nan` in
    /// tests/test_lagrangian_walker.py — the reference object carries the
    /// NaN table through make_weight.
    fn nan_distances_tables() -> D8Tables {
        let mut tables = D8Tables::standard();
        tables.distances = [f64::NAN; 9];
        tables
    }

    /// Flat index into the 5x5 test grid.
    fn ix(x: usize, y: usize) -> usize {
        x * 5 + y
    }

    /// Particles(tools) for the four repo weight tests: stage/depth/qx/qy
    /// (5x5), provided zero cell_type, gamma 0.02, dry_depth 0.1, theta 1,
    /// steepest descent on.
    fn derive_repo_case(
        stage: &[f64],
        depth: &[f64],
        qx: &[f64],
        qy: &[f64],
        tables: D8Tables,
    ) -> Fields {
        let params = Params::new(
            1.0,
            1.0,
            0.02,
            0.0,
            0.1,
            true,
            Model::DeltaRCM,
            StrandingHandling::Park,
            ReleaseRule::Immediate,
            0.0,
            1.0,
            1.0,
        )
        .unwrap();
        Fields::derive(
            &params,
            5,
            5,
            stage,
            depth,
            qx,
            qy,
            Some(vec![0.0; 25]),
            tables,
        )
        .unwrap()
    }

    /// Replica of `test_make_weight_shallow`: a lone wet cell keeps only its
    /// self-slot; border sums are zero.
    #[test]
    fn make_weight_shallow_lone_deep_cell() {
        let stage = vec![0.0; 25];
        let mut depth = vec![0.0; 25];
        depth[ix(1, 1)] = 10.0; // particle location will be wet
        let fields = derive_repo_case(&stage, &depth, &stage, &stage, nan_distances_tables());
        let at = |x: usize, y: usize, k: usize| fields.weight[(x * 5 + y) * 9 + k];

        assert_eq!(at(1, 1, 4), 1.0);
        let row: f64 = (0..9).map(|k| at(1, 1, k)).sum();
        assert_eq!(row, 1.0);
        for x in [0, 4] {
            let sum: f64 = (0..5).map(|y| at(x, y, 4)).sum();
            assert_eq!(sum, 0.0);
        }
        for y in [0, 4] {
            let sum: f64 = (0..5).map(|x| at(x, y, 4)).sum();
            assert_eq!(sum, 0.0);
        }
    }

    /// Replica of `test_make_weight_deep`: the deep cell to the SE wins the
    /// fallback slot (SE = k 8), giving get_weight == 8 under steep descent.
    #[test]
    fn make_weight_deep_neighbor_fallback() {
        let stage = vec![1.0; 25];
        let mut depth = vec![1.0; 25];
        depth[ix(2, 2)] = 10.0; // define index 8 as the deepest
        let zeros = vec![0.0; 25];
        let fields = derive_repo_case(&stage, &depth, &zeros, &stage, nan_distances_tables());

        let at = |x: usize, y: usize, k: usize| fields.weight[(x * 5 + y) * 9 + k];
        assert_eq!(at(1, 1, 8), 1.0);
    }

    /// Replica of `test_make_weight_equal_opportunity`: three equally deep
    /// cells in the neighborhood give three equiprobable fallback slots.
    #[test]
    fn make_weight_equal_opportunity() {
        let stage = vec![0.0; 25];
        let mut depth = vec![0.0; 25];
        depth[ix(1, 1)] = 10.0;
        depth[ix(1, 2)] = 10.0;
        depth[ix(2, 2)] = 10.0;
        let fields = derive_repo_case(&stage, &depth, &stage, &stage, nan_distances_tables());

        let row_sum = |x: usize, y: usize| -> f64 {
            (0..9).map(|k| fields.weight[(x * 5 + y) * 9 + k]).sum()
        };
        assert_eq!(row_sum(1, 1), 3.0);
        assert_eq!(row_sum(1, 2), 3.0);
        assert_eq!(row_sum(2, 2), 3.0);
        for x in [0, 4] {
            let sum: f64 = (0..5).map(|y| fields.weight[(x * 5 + y) * 9 + 4]).sum();
            assert_eq!(sum, 0.0);
        }
        for y in [0, 4] {
            let sum: f64 = (0..5).map(|x| fields.weight[(x * 5 + y) * 9 + 4]).sum();
            assert_eq!(sum, 0.0);
        }
    }

    /// Replica of `test_make_weight_unequal_opportunity`: with standard
    /// distances, staying put outranks the shallower neighbors, and the two
    /// equally-deep neighbors stay equiprobable.
    #[test]
    fn make_weight_unequal_opportunity() {
        let stage = vec![0.0; 25];
        let mut depth = vec![0.0; 25];
        depth[ix(1, 1)] = 10.0;
        depth[ix(1, 2)] = 5.0;
        depth[ix(2, 2)] = 5.0;
        let fields = derive_repo_case(&stage, &depth, &stage, &stage, D8Tables::standard());

        let at = |x: usize, y: usize, k: usize| fields.weight[(x * 5 + y) * 9 + k];
        assert!(at(1, 1, 4) > at(1, 1, 5));
        assert!(at(1, 1, 4) > at(1, 1, 8));
        assert_eq!(at(1, 1, 5), at(1, 1, 8));
        for x in [0, 4] {
            let sum: f64 = (0..5).map(|y| at(x, y, 4)).sum();
            assert_eq!(sum, 0.0);
        }
        for y in [0, 4] {
            let sum: f64 = (0..5).map(|x| at(x, y, 4)).sum();
            assert_eq!(sum, 0.0);
        }
    }

    /// Replica of `test_wet_boundary_no_weight`: boundary cells stay zero
    /// even when deep enough; interior rows keep nonzero weight.
    #[test]
    fn wet_boundary_no_weight() {
        let stage = vec![1.0; 25];
        let depth = vec![10.0; 25];
        let zeros = vec![0.0; 25];
        let fields = derive_repo_case(&stage, &depth, &zeros, &stage, nan_distances_tables());

        let row_sum = |x: usize, y: usize| -> f64 {
            (0..9).map(|k| fields.weight[(x * 5 + y) * 9 + k]).sum()
        };
        for x in [0, 4] {
            for y in 0..5 {
                assert_eq!(row_sum(x, y), 0.0);
            }
        }
        for x in 1..4 {
            for y in 1..4 {
                assert_ne!(row_sum(x, y), 0.0);
            }
        }
        assert!(fields.depth.data.iter().all(|&d| d == 10.0));
    }

    /// Pairwise-`nansum` order matters: with a leading 2^53 the reference's
    /// np.nansum pairs slots before combining, which differs from a plain
    /// left-to-right fold. Anchor row and values verified against
    /// NumPy 2.3.5 (np.sum -> 9007199254740998.0, sequential fold ->
    /// 9007199254740992.0).
    #[test]
    fn nansum9_uses_numpy_pairwise_order() {
        let two53 = 9_007_199_254_740_992.0_f64;
        let slots = [two53, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5];
        assert_eq!(nansum9(&slots), 9_007_199_254_740_998.0);
        // a plain sequential fold would give 9007199254740992.0
        assert_ne!(slots.iter().fold(0.0, |a, b| a + b), nansum9(&slots));

        // NaN slots read as zero without disturbing the pairing
        let with_nan = [two53, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, f64::NAN];
        assert_eq!(nansum9(&with_nan), 9_007_199_254_740_998.0);
    }

    /// NaN slots never win the max and propagate it instead.
    #[test]
    fn max9_propagates_nan() {
        let slots = [1.0, 2.0, 3.0, f64::NAN, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(max9(&slots).is_nan());
        let clean = [1.0, 2.0, 3.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(max9(&clean), 5.0);
    }
}
