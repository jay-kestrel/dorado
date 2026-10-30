//! Hydrodynamic field derivation — the port of
//! `dorado.particle_track.Particles.__init__` up to (and including) the
//! routing-weight build.
//!
//! Operation order and elementwise semantics mirror the Python reference
//! exactly; every array the reference derives as float64 from
//! harness-normalized (float64) inputs is kept float64 here. `cell_type` is
//! stored as an f64 mirror of the reference's array: the reference compares
//! cell types only by value (`== 2`, `-1 in ...`), which an f64 mirror
//! reproduces bit-exactly for the small integer codes in play.

/// Hydrodynamic model axis convention (the `model` params key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// `DeltaRCM`: `qx`/`qy` used as given.
    DeltaRCM,
    /// Any other model (e.g. `Anuga`): the reference swaps and negates the
    /// discharge components.
    Other,
}

/// Stranding/re-entrainment handling. The reference defaults to
/// `'park'` (a particle on a dry or land cell parks in place and is
/// released by rule once its cell is wet again).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrandingHandling {
    /// `park`: park-while-dry with a release rule.
    Park,
    /// `none`: legacy behavior, no parking.
    None,
}

/// Release rule for a parked particle whose cell is wet again
/// (`stranding_release`). Re-wetting is a necessary condition under
/// every rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseRule {
    /// `immediate` (reference default): release on the first wet step.
    Immediate,
    /// `rest`: release once the accumulated stranded time reaches
    /// `rest_time`.
    Rest,
    /// `probability`: per-step Bernoulli re-entrainment trial with
    /// `entrain_prob` (consumes one uniform draw per wet stranded step).
    Probability,
}

/// Scalar model parameters (the `modelParams` values the kernel consumes).
#[derive(Debug, Clone)]
pub struct Params {
    pub dx: f64,
    pub theta: f64,
    pub gamma: f64,
    pub diff_coeff: f64,
    pub dry_depth: f64,
    pub steepest_descent: bool,
    pub model: Model,
    pub stranding_handling: StrandingHandling,
    pub stranding_release: ReleaseRule,
    /// `rest_time`: model time a particle must stay stranded before
    /// release under `rest` (must be >= 0).
    pub rest_time: f64,
    /// `entrain_prob`: per-step release probability under `probability`
    /// (must lie in (0, 1]).
    pub entrain_prob: f64,
    /// `stranding_dt`: model time booked for every parked step (must be
    /// > 0); also the travel time a parked step records.
    pub stranding_dt: f64,
}

impl Params {
    /// Validated like the reference constructor: `gamma` must lie in
    /// `[0, 1]`, a negative `diff_coeff` rounds up to zero, `rest_time`
    /// >= 0, `entrain_prob` in (0, 1], `stranding_dt` > 0.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dx: f64,
        theta: f64,
        gamma: f64,
        diff_coeff: f64,
        dry_depth: f64,
        steepest_descent: bool,
        model: Model,
        stranding_handling: StrandingHandling,
        stranding_release: ReleaseRule,
        rest_time: f64,
        entrain_prob: f64,
        stranding_dt: f64,
    ) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&gamma) {
            return Err("Specified gamma value is outside the range [0, 1].".to_string());
        }
        let diff_coeff = if diff_coeff < 0.0 { 0.0 } else { diff_coeff };
        if rest_time < 0.0 {
            return Err("rest_time must be >= 0.".to_string());
        }
        if !(entrain_prob > 0.0 && entrain_prob <= 1.0) {
            return Err("entrain_prob must be in (0, 1].".to_string());
        }
        if stranding_dt <= 0.0 {
            return Err("stranding_dt must be > 0.".to_string());
        }
        Ok(Params {
            dx,
            theta,
            gamma,
            diff_coeff,
            dry_depth,
            steepest_descent,
            model,
            stranding_handling,
            stranding_release,
            rest_time,
            entrain_prob,
            stranding_dt,
        })
    }
}

/// A row-major 2D float64 grid (`L` rows x `W` columns).
#[derive(Debug, Clone)]
pub struct Grid {
    pub l: usize,
    pub w: usize,
    pub data: Vec<f64>,
}

impl Grid {
    pub fn new(l: usize, w: usize, data: Vec<f64>) -> Result<Self, String> {
        if data.len() != l * w {
            return Err(format!(
                "grid data length {} does not match {l} x {w}",
                data.len()
            ));
        }
        Ok(Grid { l, w, data })
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.data[x * self.w + y]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        self.data[x * self.w + y] = value;
    }
}

/// Derived model fields, including the routing-weight raster.
pub struct Fields {
    pub dx: f64,
    pub theta: f64,
    pub gamma: f64,
    pub diff_coeff: f64,
    pub dry_depth: f64,
    pub steepest_descent: bool,
    pub stranding_handling: StrandingHandling,
    pub stranding_release: ReleaseRule,
    pub rest_time: f64,
    pub entrain_prob: f64,
    pub stranding_dt: f64,

    pub stage: Grid,
    pub depth: Grid,
    pub qx: Grid,
    pub qy: Grid,
    pub u: Grid,
    pub v: Grid,
    pub velocity: Grid,
    pub velocity_angle: Grid,
    /// f64 mirror of the reference's `cell_type` array (same shape as the
    /// grid; `2.0` = land, `-1.0` = edge, `0.0`/`1.0` = water types).
    pub cell_type: Grid,
    /// Routing-weight raster, `L * W * 9`, slot order
    /// `[NW, N, NE, W, 0, E, SW, S, SE]`.
    pub weight: Vec<f64>,

    /// D8 tables (flattened 3x3, row-major).
    pub distances: [f64; 9],
    pub ivec: [f64; 9],
    pub jvec: [f64; 9],
    pub iwalk: [f64; 9],
    pub jwalk: [f64; 9],
    pub angles: [f64; 9],
}

impl Fields {
    /// Derive all fields from raw inputs, mirroring the reference's
    /// constructor arithmetic in its exact operation order.
    ///
    /// `cell_type` is `Some` when the fixture provides it, `None` to
    /// estimate it from depth (reference: zeros, `2` where
    /// `depth < dry_depth`, border ring padded to `-1`).
    #[allow(clippy::too_many_arguments)]
    pub fn derive(
        params: &Params,
        l: usize,
        w: usize,
        stage_in: &[f64],
        depth_in: &[f64],
        qx_in: &[f64],
        qy_in: &[f64],
        cell_type: Option<Vec<f64>>,
        tables: D8Tables,
    ) -> Result<Fields, String> {
        for (name, arr) in [
            ("stage", stage_in),
            ("depth", depth_in),
            ("qx", qx_in),
            ("qy", qy_in),
        ] {
            if arr.len() != l * w {
                return Err(format!("{name} length does not match the {l} x {w} grid"));
            }
        }

        // depth: copy with NaN -> 0
        let mut depth = depth_in.to_vec();
        for v in &mut depth {
            if v.is_nan() {
                *v = 0.0;
            }
        }

        // stage: copy, then dry cells (depth == 0) -> NaN
        let mut stage = stage_in.to_vec();
        for (i, s) in stage.iter_mut().enumerate() {
            if depth[i] == 0.0 {
                *s = f64::NAN;
            }
        }

        // discharge and velocities (model axis convention)
        let mut qx;
        let mut qy;
        if params.model == Model::DeltaRCM {
            qx = qx_in.to_vec();
            for v in &mut qx {
                if v.is_nan() {
                    *v = 0.0;
                }
            }
            qy = qy_in.to_vec();
            for v in &mut qy {
                if v.is_nan() {
                    *v = 0.0;
                }
            }
        } else {
            // non-DeltaRCM: reference swaps and negates the components
            qx = qy_in.to_vec();
            for v in qx.iter_mut() {
                *v *= -1.0;
                if v.is_nan() {
                    *v = 0.0;
                }
            }
            qy = qx_in.to_vec();
            for v in &mut qy {
                if v.is_nan() {
                    *v = 0.0;
                }
            }
        }

        let mut u = vec![0.0; qx.len()];
        let mut v = vec![0.0; qy.len()];
        for i in 0..u.len() {
            // u = qx*depth/(depth**2 + 1e-8)
            let denom = depth[i] * depth[i] + 1e-8;
            let mut ui = qx[i] * depth[i] / denom;
            if ui.is_nan() {
                ui = 0.0;
            }
            let mut vi = qy[i] * depth[i] / denom;
            if vi.is_nan() {
                vi = 0.0;
            }
            u[i] = ui;
            v[i] = vi;
        }

        // velocity magnitude with 1e-8 floor
        let mut velocity = vec![0.0; u.len()];
        for i in 0..velocity.len() {
            let mut vel = (u[i] * u[i] + v[i] * v[i]).sqrt();
            if vel < 1e-8 {
                vel = 1e-8;
            }
            velocity[i] = vel;
        }

        // floor tiny velocity components, preserving the sign of tiny
        // negative values and giving exact zeros a positive floor
        for value in u.iter_mut().chain(v.iter_mut()) {
            if value.abs() < 1e-8 {
                let sign = if *value > 0.0 {
                    1.0
                } else if *value < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let zero_bump = if *value == 0.0 { 1.0 } else { 0.0 };
                *value = (sign + zero_bump) * 1e-8;
            }
        }

        // velocity orientation
        let velocity_angle: Vec<f64> = u.iter().zip(&v).map(|(&ui, &vi)| (-ui).atan2(vi)).collect();

        // cell types: provided, or estimated from depth
        let cell_type = match cell_type {
            Some(ct) => {
                if ct.len() != stage.len() {
                    return Err("cell_type length does not match stage grid".to_string());
                }
                ct
            }
            None => {
                let mut ct = vec![0.0; stage.len()];
                for (i, c) in ct.iter_mut().enumerate() {
                    if depth[i] < params.dry_depth {
                        *c = 2.0;
                    }
                }
                // pad the border ring with -1 (reference: strip interior,
                // np.pad constant -1)
                for y in 0..w {
                    ct[y] = -1.0;
                    ct[(l - 1) * w + y] = -1.0;
                }
                for x in 0..l {
                    ct[x * w] = -1.0;
                    ct[x * w + (w - 1)] = -1.0;
                }
                ct
            }
        };

        let grid = |data: Vec<f64>| Grid::new(l, w, data);
        let mut fields = Fields {
            dx: params.dx,
            theta: params.theta,
            gamma: params.gamma,
            diff_coeff: params.diff_coeff,
            dry_depth: params.dry_depth,
            steepest_descent: params.steepest_descent,
            stranding_handling: params.stranding_handling,
            stranding_release: params.stranding_release,
            rest_time: params.rest_time,
            entrain_prob: params.entrain_prob,
            stranding_dt: params.stranding_dt,
            stage: grid(stage)?,
            depth: grid(depth)?,
            qx: grid(qx)?,
            qy: grid(qy)?,
            u: grid(u)?,
            v: grid(v)?,
            velocity: grid(velocity)?,
            velocity_angle: grid(velocity_angle)?,
            cell_type: grid(cell_type)?,
            weight: vec![0.0; l * w * 9],
            distances: tables.distances,
            ivec: tables.ivec,
            jvec: tables.jvec,
            iwalk: tables.iwalk,
            jwalk: tables.jwalk,
            angles: tables.angles,
        };
        crate::weight::make_weight(&mut fields);
        Ok(fields)
    }
}

/// D8 neighborhood tables, mirroring the reference's per-object tables
/// (which callers may override — `Particles.__init__` keeps any provided
/// `distances`/`ivec`/`jvec`/`iwalk`/`jwalk`, and `angles` likewise).
#[derive(Debug, Clone)]
pub struct D8Tables {
    /// `[[sqrt(2), 1, sqrt(2)], [1, 1, 1], [sqrt(2), 1, sqrt(2)]]`
    pub distances: [f64; 9],
    /// `[[-sqrt(.5), 0, sqrt(.5)], [-1, 0, 1], [-sqrt(.5), 0, sqrt(.5)]]`
    pub ivec: [f64; 9],
    /// `[[-sqrt(.5), -1, -sqrt(.5)], [0, 0, 0], [sqrt(.5), 1, sqrt(.5)]]`
    pub jvec: [f64; 9],
    /// `[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]`
    pub iwalk: [f64; 9],
    /// `[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`
    pub jwalk: [f64; 9],
    /// `[[3*pi/4, pi/2, pi/4], [pi, 0, 0], [5*pi/4, 3*pi/2, 7*pi/4]]`,
    /// computed left-to-right like the Python literals.
    pub angles: [f64; 9],
}

impl D8Tables {
    /// The reference's default tables.
    pub fn standard() -> Self {
        let sqrt2 = 2.0_f64.sqrt();
        let sqrt05 = 0.5_f64.sqrt();
        let pi = std::f64::consts::PI;
        D8Tables {
            distances: [sqrt2, 1.0, sqrt2, 1.0, 1.0, 1.0, sqrt2, 1.0, sqrt2],
            ivec: [-sqrt05, 0.0, sqrt05, -1.0, 0.0, 1.0, -sqrt05, 0.0, sqrt05],
            jvec: [-sqrt05, -1.0, -sqrt05, 0.0, 0.0, 0.0, sqrt05, 1.0, sqrt05],
            iwalk: [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            jwalk: [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            angles: [
                3.0 * pi / 4.0,
                pi / 2.0,
                pi / 4.0,
                pi,
                0.0,
                0.0,
                5.0 * pi / 4.0,
                3.0 * pi / 2.0,
                7.0 * pi / 4.0,
            ],
        }
    }
}
