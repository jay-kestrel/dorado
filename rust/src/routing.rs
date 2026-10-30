//! Pure routing operations. No Python, NumPy, or shared RNG state.

#[derive(Clone, Copy)]
pub enum Release {
    Immediate,
    Rest(f64),
    Probability(f64),
}

pub struct Fields {
    pub rows: usize,
    pub cols: usize,
    pub weight: Vec<f64>,
    pub velocity: Vec<f64>,
    pub velocity_angle: Vec<f64>,
    pub cell_type: Vec<i8>,
    pub depth: Vec<f64>,
    pub distances: [f64; 9],
    pub iwalk: [i64; 9],
    pub jwalk: [i64; 9],
    pub angles: [f64; 9],
    pub dx: f64,
    pub diff_coeff: f64,
    pub steepest: bool,
    pub park: bool,
    pub dry_depth: f64,
    pub stranding_dt: f64,
    pub release: Release,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub position: [i64; 2],
    pub time: f64,
    pub deposited: bool,
    pub stranded: bool,
    pub stranded_time: f64,
}

#[derive(Clone, Copy)]
pub struct Behavior {
    pub time_mult: f64,
    pub diff_mult: f64,
    pub deposition_prob: f64,
}

impl Default for Behavior {
    fn default() -> Self {
        Self {
            time_mult: 1.0,
            diff_mult: 1.0,
            deposition_prob: 0.0,
        }
    }
}

/// NumPy searchsorted(side='left'), including zero totals and exact CDF ties.
pub fn pick(weights: &[f64], uniform: f64, steepest: bool) -> usize {
    let mut probs = [0.0; 9];
    for (dest, &source) in probs.iter_mut().zip(weights) {
        *dest = if source.is_nan() { 0.0 } else { source };
    }
    if steepest {
        probs[4] = 0.0;
        let max = probs.iter().copied().fold(0.0, f64::max);
        if max <= 0.0 {
            return 4;
        }
        for value in &mut probs {
            if *value < max {
                *value = 0.0;
            }
        }
    }
    let total: f64 = probs.iter().sum();
    let cutoff = uniform * total;
    let mut cumulative = 0.0;
    for (index, prob) in probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= cutoff {
            return index;
        }
    }
    8 // Roundoff at the end of the CDF cannot produce an off-grid D8 slot.
}

pub fn new_position(position: [i64; 2], cell: usize, fields: &Fields) -> [i64; 2] {
    if cell == 4 {
        position
    } else {
        [
            position[0] + fields.jwalk[cell],
            position[1] + fields.iwalk[cell],
        ]
    }
}

pub fn travel_time(
    old_velocity: f64,
    new_velocity: f64,
    projected_distance: f64,
    diffusion: f64,
    time_mult: f64,
) -> f64 {
    0.5 * projected_distance
        * (1.0 / old_velocity + 1.0 / new_velocity)
        * (1.0 + diffusion)
        * time_mult
}

impl Fields {
    pub fn index(&self, position: [i64; 2]) -> usize {
        position[0] as usize * self.cols + position[1] as usize
    }

    pub fn contains(&self, p: [i64; 2]) -> bool {
        p[0] >= 0 && p[1] >= 0 && p[0] < self.rows as i64 && p[1] < self.cols as i64
    }

    pub fn boundary(&self, p: [i64; 2]) -> bool {
        for row in (p[0] - 1).max(0)..(p[0] + 2).min(self.rows as i64) {
            for col in (p[1] - 1).max(0)..(p[1] + 2).min(self.cols as i64) {
                if self.cell_type[self.index([row, col])] == -1 {
                    return true;
                }
            }
        }
        false
    }
}

/// Park/release precedes movement and deposition, as in the Python walker.
fn park(p: &mut Particle, fields: &Fields, draw: &mut impl FnMut() -> f64) -> bool {
    if !fields.park {
        return false;
    }
    let index = fields.index(p.position);
    let wet = fields.depth[index] > fields.dry_depth && fields.cell_type[index] != 2;
    let parked = !wet
        || (p.stranded
            && match fields.release {
                Release::Immediate => false,
                Release::Rest(rest) => p.stranded_time < rest,
                Release::Probability(prob) => draw() >= prob,
            });
    p.stranded = parked;
    if parked {
        p.stranded_time += fields.stranding_dt;
        p.time += fields.stranding_dt;
    }
    parked
}

/// Returns whether a walk-data row is appended (moving or parked).
pub fn step(
    p: &mut Particle,
    fields: &Fields,
    behavior: Behavior,
    draw: &mut impl FnMut() -> f64,
) -> bool {
    if park(p, fields, draw) {
        return true;
    }
    if p.position == [0, 0] || p.deposited || fields.boundary(p.position) {
        return false;
    }
    let old = p.position;
    let old_index = fields.index(old);
    let weights = &fields.weight[old_index * 9..old_index * 9 + 9];
    // A steepest dead end is deterministic and consumes no draw.
    let dead_end = fields.steepest && !weights.iter().enumerate().any(|(i, w)| i != 4 && *w > 0.0);
    let cell = if dead_end {
        4
    } else {
        pick(weights, draw(), fields.steepest)
    };
    let distance = fields.distances[cell] * fields.dx;
    let next = new_position(old, cell, fields);
    if distance == 0.0 || next == old || !fields.contains(next) {
        return false;
    }
    if behavior.deposition_prob > 0.0 && draw() < behavior.deposition_prob {
        p.deposited = true;
        return false;
    }
    let diffusion = if fields.diff_coeff > 0.0 {
        (0.5 - draw()) * fields.diff_coeff * behavior.diff_mult
    } else {
        0.0
    };
    let projected = distance
        * (fields.angles[cell] - fields.velocity_angle[old_index])
            .cos()
            .max(0.0);
    p.time += travel_time(
        fields.velocity[old_index],
        fields.velocity[fields.index(next)],
        projected,
        diffusion,
        behavior.time_mult,
    );
    p.position = next;
    true
}

#[derive(Default)]
pub struct Outcome {
    pub history: Vec<Particle>,
    pub recorded: bool,
    pub hit_max_iter: bool,
}

pub fn advance(
    p: &mut Particle,
    fields: &Fields,
    behavior: Behavior,
    draw: &mut impl FnMut() -> f64,
    target: Option<f64>,
    mut estimate: f64,
    max_iter: usize,
) -> Outcome {
    let mut out = Outcome::default();
    let Some(target) = target else {
        out.recorded = step(p, fields, behavior, draw);
        return out;
    };
    // run_iteration skips boundary particles before entering the adaptive loop.
    if fields.boundary(p.position) {
        return out;
    }
    let mut count = 1;
    while (p.time - target).abs() >= (p.time + estimate - target).abs() {
        let previous = *p;
        if !step(p, fields, behavior, draw) {
            // run_iteration only persists a deposition flip when no row was
            // appended. A release flag without movement is not committed.
            p.stranded = previous.stranded;
            p.stranded_time = previous.stranded_time;
            if let Some(last) = out.history.last_mut() {
                last.deposited = p.deposited;
            }
            break;
        }
        out.history.push(*p);
        out.recorded = true;
        estimate = (p.time - previous.time).max(0.1);
        count += 1;
        if count > max_iter {
            out.hit_max_iter = true;
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_and_steepest_edges() {
        let weights = [f64::NAN, 1.0, 2.0, 0.0, 99.0, 2.0, 0.0, 0.0, 0.0];
        assert_eq!(pick(&weights, 0.5, true), 2);
        assert_eq!(pick(&weights, 0.9, true), 5);
        assert_eq!(pick(&[0.0; 9], 0.5, false), 0);
        assert_eq!(pick(&[0.0; 9], 0.5, true), 4);
        assert_eq!(pick(&weights, 0.0, false), 0);
    }

    #[test]
    fn travel_time_order_matches_reference() {
        assert_eq!(travel_time(2.0, 4.0, 10.0, 0.2, 3.0), 13.5);
    }
}
