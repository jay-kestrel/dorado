//! Lagrangian particle walker — the port of
//! `dorado.lagrangian_walker` (steady mode: `particle_stepper`,
//! `run_iteration` without `target_time`) plus `generate_particles`.
//!
//! Draw order is the parity contract: per iteration the reference (1)
//! evaluates the park state machine in list order (`probability` release
//! consumes one uniform per wet stranded particle), then (2) runs the
//! pick pass in list order (one draw per routing pick, skipped for
//! parked particles and the `(0, 0)` sentinel), then (3) runs the
//! travel-time pass in list order (one draw per moved particle when
//! `diff_coeff > 0`). `generate_particles` consumes all x draws before
//! all y draws. Particle kinds/sizes and deposition (`particle_size`,
//! `particle_type`, `deposited`) are out of the port's surface — the
//! harness workloads never set them, and with them absent the reference
//! takes the identity-multiplier path replicated here.

use crate::fields::{Fields, ReleaseRule, StrandingHandling};
use crate::rng::Mt19937;

/// Walk data for one or more iterations: `field[particle][iteration]`,
/// matching the reference's `walk_data` dictionary. Under park
/// stranding, `stranded`/`stranded_time` are `Some` and mirror the
/// reference's optional-variable channel.
#[derive(Debug, Clone)]
pub struct WalkData {
    pub xinds: Vec<Vec<i64>>,
    pub yinds: Vec<Vec<i64>>,
    pub travel_times: Vec<Vec<f64>>,
    pub stranded: Option<Vec<Vec<i64>>>,
    pub stranded_time: Option<Vec<Vec<f64>>>,
}

impl WalkData {
    fn new(n: usize, park: bool) -> Self {
        WalkData {
            xinds: Vec::with_capacity(n),
            yinds: Vec::with_capacity(n),
            travel_times: Vec::with_capacity(n),
            stranded: park.then(|| Vec::with_capacity(n)),
            stranded_time: park.then(|| Vec::with_capacity(n)),
        }
    }

    /// Most recent location of particle `i` (the stepper's `start_pairs`).
    fn last_pair(&self, i: usize) -> (i64, i64) {
        let n = self.xinds[i].len();
        (self.xinds[i][n - 1], self.yinds[i][n - 1])
    }
}

/// `random_pick_seed`: one uniform over the cumulative [1..n] ladder
/// (probs of ones → `cutoffs[i] == i+1`), first cutoff >= draw wins
/// (searchsorted-left).
pub fn random_pick_seed(rng: &mut Mt19937, choices: &[i64]) -> i64 {
    let n = choices.len();
    let u = rng.uniform(0.0, n as f64);
    let idx = (0..n).find(|&i| (i + 1) as f64 >= u).unwrap_or(n - 1);
    choices[idx]
}

/// `random_pick`: weighted pick over a 9-slot weight row. Parity quirks
/// mirrored from the reference: NaN slots read as zero, and an all-zero
/// row still consumes the uniform draw and resolves to slot 0 (NW)
/// because every cutoff equals 0 and `searchsorted(0)` lands left.
pub fn random_pick(rng: &mut Mt19937, row: &[f64; 9]) -> usize {
    let mut probs = *row;
    for p in &mut probs {
        if p.is_nan() {
            *p = 0.0;
        }
    }
    // cumulative sum, left-to-right like np.cumsum
    let mut cutoffs = [0.0f64; 9];
    let mut run = 0.0;
    for (c, p) in cutoffs.iter_mut().zip(&probs) {
        run += p;
        *c = run;
    }
    let u = rng.uniform(0.0, cutoffs[8]);
    cutoffs.iter().position(|&c| c >= u).unwrap_or(8)
}

/// `steep_descent`: pick the max-weight slot (index 4 excluded); ties
/// are broken randomly over the tied slots. A dead end (no neighbor
/// weight above zero) returns slot 4 without consuming a draw — the
/// reference's post-#5 behavior.
pub fn steep_descent(rng: &mut Mt19937, row: &[f64; 9]) -> usize {
    let mut probs = *row;
    for p in &mut probs {
        if p.is_nan() {
            *p = 0.0;
        }
    }
    probs[4] = 0.0;
    if !probs.iter().any(|&p| p > 0.0) {
        return 4;
    }
    let max_val = probs.iter().copied().fold(f64::NAN, f64::max);
    for p in &mut probs {
        if *p < max_val {
            *p = 0.0;
        }
    }
    let mut cutoffs = [0.0f64; 9];
    let mut run = 0.0;
    for (c, p) in cutoffs.iter_mut().zip(&probs) {
        run += p;
        *c = run;
    }
    let u = rng.uniform(0.0, cutoffs[8]);
    cutoffs.iter().position(|&c| c >= u).unwrap_or(8)
}

/// `get_weight`: routing pick at `(x, y)` from the weight raster.
fn get_weight(fields: &Fields, rng: &mut Mt19937, x: i64, y: i64) -> usize {
    let base = ((x as usize) * fields.depth.w + y as usize) * 9;
    let mut row = [0.0f64; 9];
    row.copy_from_slice(&fields.weight[base..base + 9]);
    if fields.steepest_descent {
        steep_descent(rng, &row)
    } else {
        random_pick(rng, &row)
    }
}

/// `stranding_park_decisions`: the park-while-dry state machine, per
/// particle in list order. `prev_*` carry the previous step's stranded
/// flag and cumulative time (`None` treated as never stranded — the
/// reference's `[[None]]` initialization).
fn stranding_park_decisions(
    fields: &Fields,
    rng: &mut Mt19937,
    inds: &[(i64, i64)],
    prev_stranded: &[i64],
    prev_times: &[f64],
) -> (Vec<bool>, Vec<f64>) {
    let mut park_mask = Vec::with_capacity(inds.len());
    let mut stranded_times = Vec::with_capacity(inds.len());
    for (i, &(x, y)) in inds.iter().enumerate() {
        let stranded_time = prev_times[i];
        let stranded = prev_stranded[i];
        let wet = fields.depth.get(x as usize, y as usize) > fields.dry_depth
            && fields.cell_type.get(x as usize, y as usize) != 2.0;
        let mut park = true;
        if wet {
            if stranded == 0 {
                park = false;
            } else {
                match fields.stranding_release {
                    ReleaseRule::Immediate => park = false,
                    ReleaseRule::Rest => park = stranded_time < fields.rest_time,
                    ReleaseRule::Probability => {
                        park = rng.random_sample() >= fields.entrain_prob;
                    }
                }
            }
        }
        park_mask.push(park);
        stranded_times.push(if park {
            stranded_time + fields.stranding_dt
        } else {
            stranded_time
        });
    }
    (park_mask, stranded_times)
}

/// `calculate_new_ind`: step `(x, y)` by the D8 walk tables. The
/// reference adds **jwalk** to x and **iwalk** to y — the crossed
/// naming is the reference's, replicated verbatim.
fn calculate_new_ind(fields: &Fields, ind: (i64, i64), new_cell: usize) -> (i64, i64) {
    (
        ind.0 + fields.jwalk[new_cell] as i64,
        ind.1 + fields.iwalk[new_cell] as i64,
    )
}

/// `step_update`: distance traveled for the chosen slot.
fn step_update(new_cell: usize, distances: &[f64; 9], dx: f64) -> f64 {
    distances[new_cell] * dx
}

/// `calc_travel_times`: travel time for a (possibly) moved particle.
/// Consumes one uniform draw when the particle moved and
/// `diff_coeff > 0` (the reference's legacy `random()`).
fn calc_travel_times(
    fields: &Fields,
    rng: &mut Mt19937,
    new_cell: usize,
    ind: (i64, i64),
    new_ind: (i64, i64),
    dist: f64,
) -> f64 {
    if ind != new_ind {
        let old_vel = fields.velocity.get(ind.0 as usize, ind.1 as usize);
        let new_vel = fields.velocity.get(new_ind.0 as usize, new_ind.1 as usize);
        let diff = if fields.diff_coeff > 0.0 {
            (0.5 - rng.random_sample()) * fields.diff_coeff
        } else {
            0.0
        };
        let angle =
            fields.angles[new_cell] - fields.velocity_angle.get(ind.0 as usize, ind.1 as usize);
        // Python's max(0, x) returns 0 for NaN (the comparison fails);
        // f64::max(0.0, NaN) is likewise 0.0.
        let projected_dist = dist * 0.0_f64.max(angle.cos());
        0.5 * projected_dist * (1.0 / old_vel + 1.0 / new_vel) * (1.0 + diff)
    } else {
        0.0
    }
}

/// `check_for_boundary`: cancel the move of any particle whose current
/// neighborhood (clamped at the lower edge like the reference's
/// post-#5 slice fix) contains an edge cell (`-1`).
fn check_for_boundary(fields: &Fields, new_inds: &mut [(i64, i64)], current: &[(i64, i64)]) {
    let l = fields.cell_type.l as i64;
    let w = fields.cell_type.w as i64;
    for (i, &(cx, cy)) in current.iter().enumerate() {
        let x0 = (cx - 1).max(0);
        let y0 = (cy - 1).max(0);
        // Python slices upper-bounds truncate at the array edge
        let x1 = (cx + 2).min(l);
        let y1 = (cy + 2).min(w);
        let mut borders_edge = false;
        for xx in x0..x1 {
            for yy in y0..y1 {
                if fields.cell_type.get(xx as usize, yy as usize) == -1.0 {
                    borders_edge = true;
                }
            }
        }
        if borders_edge {
            new_inds[i] = current[i];
        }
    }
}

/// Step outcome for one `particle_stepper` pass, per particle in list
/// order: new index, updated total travel time, stranded flag, and
/// cumulative stranded time.
struct StepperResult {
    new_inds: Vec<(i64, i64)>,
    totals: Vec<f64>,
    stranded: Vec<i64>,
    stranded_times: Vec<f64>,
}

/// One steady-mode `particle_stepper` pass: park decisions, picks,
/// movement, boundary/out-of-bounds freezes, and travel times, with the
/// reference's exact draw order (park, then picks, then travel).
fn particle_stepper(
    fields: &Fields,
    rng: &mut Mt19937,
    current: &[(i64, i64)],
    travel_times: &[f64],
    prev_stranded: &[i64],
    prev_times: &[f64],
) -> StepperResult {
    let n = current.len();

    let (park_mask, stranded_times) = if fields.stranding_handling == StrandingHandling::Park {
        stranding_park_decisions(fields, rng, current, prev_stranded, prev_times)
    } else {
        (vec![false; n], vec![0.0; n])
    };

    // picks: parked particles and the (0, 0) sentinel keep slot 4
    let new_cells: Vec<usize> = current
        .iter()
        .enumerate()
        .map(|(i, &ind)| {
            if ind != (0, 0) && !park_mask[i] {
                get_weight(fields, rng, ind.0, ind.1)
            } else {
                4
            }
        })
        .collect();

    let mut new_inds: Vec<(i64, i64)> = new_cells
        .iter()
        .zip(current)
        .map(|(&cell, &ind)| {
            if cell != 4 {
                calculate_new_ind(fields, ind, cell)
            } else {
                ind
            }
        })
        .collect();

    // zero-distance picks keep the particle at its current indices
    let dists: Vec<f64> = new_cells
        .iter()
        .map(|&cell| step_update(cell, &fields.distances, fields.dx))
        .collect();
    for i in 0..n {
        if dists[i] == 0.0 {
            new_inds[i] = current[i];
        }
    }

    check_for_boundary(fields, &mut new_inds, current);

    // freeze steps that would leave the domain (no -1 border to react to)
    let l = fields.cell_type.l as i64;
    let w = fields.cell_type.w as i64;
    for i in 0..n {
        let (nx, ny) = new_inds[i];
        if nx < 0 || nx >= l || ny < 0 || ny >= w {
            new_inds[i] = current[i];
        }
    }

    // travel times, then parked steps book the stranding clock instead
    let mut temp_travel: Vec<f64> = new_cells
        .iter()
        .enumerate()
        .map(|(i, &cell)| calc_travel_times(fields, rng, cell, current[i], new_inds[i], dists[i]))
        .collect();
    for (i, &parked) in park_mask.iter().enumerate() {
        if parked {
            temp_travel[i] = fields.stranding_dt;
        }
    }
    let total: Vec<f64> = travel_times
        .iter()
        .zip(&temp_travel)
        .map(|(&t, &dt)| t + dt)
        .collect();

    let stranded: Vec<i64> = park_mask.iter().map(|&p| i64::from(p)).collect();
    StepperResult {
        new_inds,
        totals: total,
        stranded,
        stranded_times,
    }
}

/// `generate_particles` (exact + random seeding): with `seed_x`/`seed_y`
/// seed regions, seeds `n` particles — random method consumes all x
/// draws, then all y draws; walk data starts with one entry per
/// particle at travel time 0.
pub fn generate_particles(
    fields: &Fields,
    rng: &mut Mt19937,
    n: usize,
    seed_x: &[i64],
    seed_y: &[i64],
    method: SeedMethod,
) -> WalkData {
    let park = fields.stranding_handling == StrandingHandling::Park;
    let mut walk = WalkData::new(n, park);

    let (start_x, start_y): (Vec<i64>, Vec<i64>) = match method {
        SeedMethod::Random => {
            let xs: Vec<i64> = (0..n).map(|_| random_pick_seed(rng, seed_x)).collect();
            let ys: Vec<i64> = (0..n).map(|_| random_pick_seed(rng, seed_y)).collect();
            (xs, ys)
        }
        SeedMethod::Exact => {
            // divmod(n, len(seed)) repeats, then the remainder one by one
            let per = n / seed_x.len();
            let rem = n % seed_x.len();
            let mut xs = Vec::with_capacity(n);
            let mut ys = Vec::with_capacity(n);
            for _ in 0..per {
                xs.extend(seed_x);
                ys.extend(seed_y);
            }
            for i in 0..rem {
                xs.push(seed_x[i]);
                ys.push(seed_y[i]);
            }
            (xs, ys)
        }
    };

    for i in 0..n {
        walk.xinds.push(vec![start_x[i]]);
        walk.yinds.push(vec![start_y[i]]);
        walk.travel_times.push(vec![0.0]);
        if let Some(s) = walk.stranded.as_mut() {
            s.push(vec![0]);
        }
        if let Some(s) = walk.stranded_time.as_mut() {
            s.push(vec![0.0]);
        }
    }
    walk
}

/// Seeding method (`generate_particles`'s `method` argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedMethod {
    /// `'random'`: draws from the seed regions (2 draws per particle).
    Random,
    /// `'exact'`: assigned directly, no draws.
    Exact,
}

/// One steady-mode `run_iteration` (no `target_time`): step all
/// particles, appending per-particle entries for every particle that
/// moved or parked (parked steps are recorded so strand durations are
/// measurable — the position repeats, time advances).
pub fn run_iteration(fields: &Fields, rng: &mut Mt19937, walk: &mut WalkData) {
    let n = walk.xinds.len();
    let current: Vec<(i64, i64)> = (0..n).map(|i| walk.last_pair(i)).collect();
    let travel: Vec<f64> = (0..n)
        .map(|i| {
            let t = &walk.travel_times[i];
            t[t.len() - 1]
        })
        .collect();
    let park = fields.stranding_handling == StrandingHandling::Park;

    let default_stranded = vec![0i64; n];
    let default_times = vec![0.0f64; n];
    let prev_stranded: Vec<i64> = if park {
        (0..n)
            .map(|i| {
                let s = walk.stranded.as_ref().map_or(&default_stranded, |s| &s[i]);
                s[s.len() - 1]
            })
            .collect()
    } else {
        Vec::new()
    };
    let prev_times: Vec<f64> = if park {
        (0..n)
            .map(|i| {
                let s = walk
                    .stranded_time
                    .as_ref()
                    .map_or(&default_times, |s| &s[i]);
                s[s.len() - 1]
            })
            .collect()
    } else {
        Vec::new()
    };

    let StepperResult {
        new_inds,
        totals,
        stranded,
        stranded_times,
    } = particle_stepper(fields, rng, &current, &travel, &prev_stranded, &prev_times);

    for i in 0..n {
        let parked = park && stranded[i] == 1;
        if new_inds[i] != current[i] || parked {
            walk.xinds[i].push(new_inds[i].0);
            walk.yinds[i].push(new_inds[i].1);
            walk.travel_times[i].push(totals[i]);
            if let Some(s) = walk.stranded.as_mut() {
                s[i].push(stranded[i]);
            }
            if let Some(s) = walk.stranded_time.as_mut() {
                s[i].push(stranded_times[i]);
            }
        }
    }
}

/// The reference runner's whole timed region: seed the stream, generate
/// particles, and run `iterations` steady-mode steps.
pub fn run_steady_walk(
    fields: &Fields,
    seed: u32,
    particles: usize,
    seed_x: &[i64],
    seed_y: &[i64],
    method: SeedMethod,
    iterations: usize,
) -> WalkData {
    let mut rng = Mt19937::new(seed as u64);
    let mut walk = generate_particles(fields, &mut rng, particles, seed_x, seed_y, method);
    for _ in 0..iterations {
        run_iteration(fields, &mut rng, &mut walk);
    }
    walk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{D8Tables, Fields, Grid, Model, Params, ReleaseRule, StrandingHandling};

    /// Flat index into an `l x w` test grid.
    fn ix(l: usize, x: usize, y: usize) -> usize {
        x * l + y
    }

    fn params_for(
        dx: f64,
        gamma: f64,
        diff_coeff: f64,
        steepest: bool,
        handling: StrandingHandling,
    ) -> Params {
        Params::new(
            dx,
            1.0,
            gamma,
            diff_coeff,
            0.1,
            steepest,
            Model::DeltaRCM,
            handling,
            ReleaseRule::Immediate,
            0.0,
            1.0,
            1.0,
        )
        .unwrap()
    }

    /// The module-level `params` of tests/test_lagrangian_walker.py:
    /// 5x5, stage ones, cell_type zeros, qy ones, qx zeros, standard D8
    /// tables, depth = stage.copy() (all ones), steepest descent on.
    fn walker_test_fields(depth: &[f64], stage: &[f64]) -> Fields {
        let params = params_for(1.0, 0.02, 0.0, true, StrandingHandling::Park);
        Fields::derive(
            &params,
            5,
            5,
            stage,
            depth,
            &[0.0; 25],
            &[1.0; 25],
            Some(vec![0.0; 25]),
            D8Tables::standard(),
        )
        .unwrap()
    }

    /// Replica of `test_random_pick`: probs zeros with slot 0 set to 1
    /// picks slot 0 at seed 0.
    #[test]
    fn random_pick_single_one() {
        let mut rng = Mt19937::new(0);
        let probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(random_pick(&mut rng, &probs), 0);
    }

    /// Replica of `test_random_pick_seed`: a single choice returns it.
    #[test]
    fn random_pick_seed_single_choice() {
        let mut rng = Mt19937::new(0);
        assert_eq!(random_pick_seed(&mut rng, &[0]), 0);
    }

    /// Replica of `test_get_weight`: uniform depth, steepest descent,
    /// qy ones — the normalized inertial weights put the max in slot 5.
    #[test]
    fn get_weight_picks_max_slot() {
        let depth = vec![1.0; 25];
        let stage = vec![1.0; 25];
        let fields = walker_test_fields(&depth, &stage);
        let mut rng = Mt19937::new(0);
        assert_eq!(get_weight(&fields, &mut rng, 1, 1), 5);
    }

    /// Replica of `test_get_weight_norm`: stage spike at the center
    /// does not change the pick (surface slope out of the cell is
    /// non-positive everywhere).
    #[test]
    fn get_weight_with_stage_spike() {
        let depth = vec![1.0; 25];
        let mut stage = vec![1.0; 25];
        stage[ix(5, 1, 1)] = 100.0;
        let fields = walker_test_fields(&depth, &stage);
        let mut rng = Mt19937::new(0);
        assert_eq!(get_weight(&fields, &mut rng, 1, 1), 5);
    }

    /// Replica of `test_get_weight_deep`: NaN distances with a deep
    /// SE neighbor resolve the fallback to slot 8.
    #[test]
    fn get_weight_deep_neighbor() {
        let mut depth = vec![1.0; 25];
        depth[ix(5, 2, 2)] = 10.0;
        let stage = vec![1.0; 25];
        let params = params_for(1.0, 0.02, 0.0, true, StrandingHandling::Park);
        let tables = {
            let mut t = D8Tables::standard();
            t.distances = [f64::NAN; 9];
            t
        };
        let fields = Fields::derive(
            &params,
            5,
            5,
            &stage,
            &depth,
            &[0.0; 25],
            &[1.0; 25],
            Some(vec![0.0; 25]),
            tables,
        )
        .unwrap();
        let mut rng = Mt19937::new(0);
        assert_eq!(get_weight(&fields, &mut rng, 1, 1), 8);
    }

    /// Replica of `test_calculate_new_ind`: slot 0 steps NW. The
    /// reference adds jwalk to x and iwalk to y.
    #[test]
    fn calculate_new_ind_northwest() {
        let fields = walker_test_fields(&[1.0; 25], &[1.0; 25]);
        assert_eq!(calculate_new_ind(&fields, (1, 1), 0), (0, 0));
    }

    /// Replicas of `test_step_update_straight` / `_diagonal`.
    #[test]
    fn step_update_distances() {
        let distances = D8Tables::standard().distances;
        assert_eq!(step_update(1, &distances, 1.0), 1.0);
        assert_eq!(step_update(2, &distances, 1.0), std::f64::consts::SQRT_2);
    }

    /// Replica of `test_calc_travel_times`: hard-set velocities and a
    /// unit velocity angle give the golden 0.5609806565385976 with no
    /// diffusion draw (diff_coeff = 0). velocity[0, 0:2] = 3, all other
    /// velocities 1, velocity_angle all ones.
    #[test]
    fn calc_travel_times_golden() {
        let mut fields = walker_test_fields(&[1.0; 25], &[1.0; 25]);
        for v in fields.velocity.data.iter_mut() {
            *v = 1.0;
        }
        fields.velocity.data[0] = 3.0;
        fields.velocity.data[1] = 3.0;
        for a in fields.velocity_angle.data.iter_mut() {
            *a = 1.0;
        }
        let mut rng = Mt19937::new(0);
        let t = calc_travel_times(&fields, &mut rng, 1, (1, 1), (0, 1), 1.0);
        assert!((t - 0.5609806565385976).abs() < 1e-12);
    }

    /// Replica of `test_check_for_boundary`: a proposed move next to an
    /// edge cell is cancelled.
    #[test]
    fn check_for_boundary_cancels() {
        // 3x3 cell types with -1 at (0,0) and (0,1)
        let mut ct = vec![1.0; 9];
        ct[ix(3, 0, 0)] = -1.0;
        ct[ix(3, 0, 1)] = -1.0;
        let mut fields = walker_test_fields(&[1.0; 25], &[1.0; 25]);
        fields.cell_type = Grid::new(3, 3, ct).unwrap();
        let mut new_inds = vec![(0, 1)];
        check_for_boundary(&fields, &mut new_inds, &[(1, 1)]);
        assert_eq!(new_inds[0], (1, 1));
    }

    /// Replica of `test_park_while_dry`: a dry-cell particle and a
    /// permanent-land particle park in place, booking stranding_dt per
    /// step; no draws are consumed.
    #[test]
    fn park_while_dry() {
        // 5x5: all wet except a dry center; (4,4) is permanent land at
        // a wet depth
        let mut depth = vec![1.0; 25];
        depth[ix(5, 2, 2)] = 0.05;
        let mut ct = vec![1.0; 25];
        ct[ix(5, 4, 4)] = 2.0;
        let stage = vec![1.0; 25];
        let params = params_for(1.0, 0.02, 0.0, false, StrandingHandling::Park);
        let fields = Fields::derive(
            &params,
            5,
            5,
            &stage,
            &depth,
            &[1.0; 25],
            &[0.0; 25],
            Some(ct),
            D8Tables::standard(),
        )
        .unwrap();
        let mut rng = Mt19937::new(0);
        let mut walk =
            generate_particles(&fields, &mut rng, 2, &[2, 4], &[2, 4], SeedMethod::Exact);
        for _ in 0..3 {
            run_iteration(&fields, &mut rng, &mut walk);
        }
        assert_eq!(walk.xinds, vec![vec![2, 2, 2, 2], vec![4, 4, 4, 4]]);
        assert_eq!(walk.yinds, vec![vec![2, 2, 2, 2], vec![4, 4, 4, 4]]);
        assert_eq!(
            walk.stranded.as_ref().unwrap(),
            &vec![vec![0, 1, 1, 1], vec![0, 1, 1, 1]]
        );
        assert_eq!(
            walk.travel_times,
            vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]]
        );
        assert_eq!(
            walk.stranded_time.as_ref().unwrap(),
            &vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]]
        );
    }

    /// The draw-order contract under probability release: the i-th wet
    /// stranded particle's release decision consumes the i-th uniform
    /// draw of the stream.
    #[test]
    fn probability_release_draw_order() {
        let mut fields = walker_test_fields(&[1.0; 25], &[1.0; 25]);
        fields.stranding_release = ReleaseRule::Probability;
        fields.entrain_prob = 0.25;
        // 8 wet stranded particles
        let n = 8;
        let current: Vec<(i64, i64)> = (0..n).map(|i| (2, i as i64)).collect();
        let prev_stranded = vec![1i64; n];
        let prev_times = vec![1.0f64; n];
        let expected: Vec<bool> = {
            let mut rng = Mt19937::new(42);
            (0..n).map(|_| rng.random_sample() < 0.25).collect()
        };
        let mut rng = Mt19937::new(42);
        let (park_mask, times) =
            stranding_park_decisions(&fields, &mut rng, &current, &prev_stranded, &prev_times);
        for i in 0..n {
            assert_eq!(park_mask[i], !expected[i], "particle {i}");
            assert_eq!(
                times[i],
                if expected[i] {
                    1.0
                } else {
                    1.0 + fields.stranding_dt
                }
            );
        }
    }

    /// Classic RCM end-to-end: `tests/test_examplecases.py::test_few_steps_RCM`.
    #[test]
    fn classic_rcm_trajectory() {
        // stage/depth exported from tests/data/ex_deltarcm_data.npz
        let stage = crate::npy::read_npy("tests/data/rust_port_stage_f32.npy").expect("stage npy");
        let depth = crate::npy::read_npy("tests/data/rust_port_depth_f32.npy").expect("depth npy");
        let (l, w) = (120usize, 240usize);
        // defaults of test_examplecases: gamma 0.05, diff_coeff 0.2
        // (steepest descent off), dry_depth 0.1, park handling
        let params = params_for(50.0, 0.05, 0.2, false, StrandingHandling::Park);
        let fields = Fields::derive(
            &params,
            l,
            w,
            &stage.data,
            &depth.data,
            &vec![0.0; l * w],
            &vec![0.0; l * w],
            None,
            D8Tables::standard(),
        )
        .unwrap();
        let walk = run_steady_walk(&fields, 0, 1, &[16], &[140], SeedMethod::Random, 3);
        assert_eq!(walk.xinds[0], vec![16, 17, 18, 19]);
        assert_eq!(walk.yinds[0], vec![140, 141, 141, 141]);
        let expected = [
            0.0,
            7007593448.337235,
            10439964733.462337,
            13698473384.876724,
        ];
        for (got, want) in walk.travel_times[0].iter().zip(&expected) {
            assert!((got - want).abs() <= want.abs() * 1e-6, "{got} vs {want}");
        }
    }
}
