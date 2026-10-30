//! Validated Python/NumPy boundary; the routing loop lives in `routing`.
mod routing;

use numpy::{
    PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2,
    PyUntypedArrayMethods,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use routing::{Behavior, Fields, Outcome, Particle, Release};
use std::collections::HashMap;

fn require(condition: bool, message: &str) -> PyResult<()> {
    if condition {
        Ok(())
    } else {
        Err(PyValueError::new_err(message.to_owned()))
    }
}

fn finite(value: f64, name: &str) -> PyResult<()> {
    require(value.is_finite(), &format!("{name} must be finite"))
}

/// An immutable, owned snapshot. Construct once per static flow field.
#[pyclass(frozen, module = "dorado._native")]
struct RoutingFields {
    inner: Fields,
}

#[pymethods]
impl RoutingFields {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (weight, velocity, velocity_angle, cell_type, depth,
        distances, iwalk, jwalk, angles, dx, diff_coeff, *, steepest_descent=false,
        stranding_handling="park", dry_depth=0.1, stranding_dt=1.0,
        stranding_release="immediate", rest_time=0.0, entrain_prob=1.0))]
    fn new(
        weight: PyReadonlyArray3<'_, f64>,
        velocity: PyReadonlyArray2<'_, f64>,
        velocity_angle: PyReadonlyArray2<'_, f64>,
        cell_type: PyReadonlyArray2<'_, i8>,
        depth: PyReadonlyArray2<'_, f64>,
        distances: PyReadonlyArray1<'_, f64>,
        iwalk: PyReadonlyArray1<'_, i64>,
        jwalk: PyReadonlyArray1<'_, i64>,
        angles: PyReadonlyArray1<'_, f64>,
        dx: f64,
        diff_coeff: f64,
        steepest_descent: bool,
        stranding_handling: &str,
        dry_depth: f64,
        stranding_dt: f64,
        stranding_release: &str,
        rest_time: f64,
        entrain_prob: f64,
    ) -> PyResult<Self> {
        let shape = weight.shape();
        require(
            shape[0] > 0 && shape[1] > 0 && shape[2] == 9,
            "weight must have shape (L, W, 9) with L, W > 0",
        )?;
        for (name, other) in [
            ("velocity", velocity.shape()),
            ("velocity_angle", velocity_angle.shape()),
            ("cell_type", cell_type.shape()),
            ("depth", depth.shape()),
        ] {
            require(
                other == &shape[..2],
                &format!("{name} must have shape (L, W)"),
            )?;
        }
        require(
            weight.is_c_contiguous()
                && velocity.is_c_contiguous()
                && velocity_angle.is_c_contiguous()
                && cell_type.is_c_contiguous()
                && depth.is_c_contiguous(),
            "fields must be C-contiguous",
        )?;
        let weights = weight.as_slice()?;
        for row in weights.chunks_exact(9) {
            require(
                row.iter()
                    .all(|w| w.is_nan() || (w.is_finite() && *w >= 0.0))
                    && row.iter().filter(|w| !w.is_nan()).sum::<f64>().is_finite(),
                "weights must be nonnegative finite values or NaN, with finite row sums",
            )?;
        }
        require(
            velocity
                .as_slice()?
                .iter()
                .all(|v| v.is_finite() && *v > 0.0),
            "velocity must be finite and positive",
        )?;
        require(
            velocity_angle.as_slice()?.iter().all(|v| v.is_finite()),
            "velocity_angle must be finite",
        )?;
        require(
            depth
                .as_slice()?
                .iter()
                .all(|v| v.is_finite() || v.is_nan()),
            "depth cannot contain infinities",
        )?;
        require(
            cell_type.as_slice()?.iter().all(|v| (-1..=2).contains(v)),
            "cell_type must be in [-1, 2]",
        )?;
        let distances: [f64; 9] = distances
            .as_slice()?
            .try_into()
            .map_err(|_| PyValueError::new_err("distances must have length 9"))?;
        let angles: [f64; 9] = angles
            .as_slice()?
            .try_into()
            .map_err(|_| PyValueError::new_err("angles must have length 9"))?;
        let iwalk: [i64; 9] = iwalk
            .as_slice()?
            .try_into()
            .map_err(|_| PyValueError::new_err("iwalk must have length 9"))?;
        let jwalk: [i64; 9] = jwalk
            .as_slice()?
            .try_into()
            .map_err(|_| PyValueError::new_err("jwalk must have length 9"))?;
        require(
            distances.iter().all(|d| d.is_finite() && *d >= 0.0),
            "distances must be finite and nonnegative",
        )?;
        require(
            angles.iter().all(|a| a.is_finite()),
            "angles must be finite",
        )?;
        require(
            iwalk.iter().chain(&jwalk).all(|v| (-1..=1).contains(v)),
            "D8 offsets must be in [-1, 1]",
        )?;
        for (name, value) in [
            ("dx", dx),
            ("diff_coeff", diff_coeff),
            ("dry_depth", dry_depth),
            ("stranding_dt", stranding_dt),
            ("rest_time", rest_time),
            ("entrain_prob", entrain_prob),
        ] {
            finite(value, name)?;
        }
        require(
            dx > 0.0
                && diff_coeff >= 0.0
                && stranding_dt > 0.0
                && rest_time >= 0.0
                && (0.0..=1.0).contains(&entrain_prob),
            "invalid routing or stranding scalar range",
        )?;
        let park = match stranding_handling {
            "none" => false,
            "park" => true,
            _ => {
                return Err(PyValueError::new_err(
                    "stranding_handling must be 'none' or 'park'",
                ))
            }
        };
        let release = match stranding_release {
            "immediate" => Release::Immediate,
            "rest" => Release::Rest(rest_time),
            "probability" => Release::Probability(entrain_prob),
            _ => return Err(PyValueError::new_err("unknown stranding_release")),
        };
        Ok(Self {
            inner: Fields {
                rows: shape[0],
                cols: shape[1],
                weight: weights.to_vec(),
                velocity: velocity.as_slice()?.to_vec(),
                velocity_angle: velocity_angle.as_slice()?.to_vec(),
                cell_type: cell_type.as_slice()?.to_vec(),
                depth: depth.as_slice()?.to_vec(),
                distances,
                iwalk,
                jwalk,
                angles,
                dx,
                diff_coeff,
                steepest: steepest_descent,
                park,
                dry_depth,
                stranding_dt,
                release,
            },
        })
    }
}

struct Stream {
    rng: ChaCha8Rng,
    draws: u64,
}

/// Persistent independent ChaCha8 streams, indexed by stable particle ID.
#[pyclass(module = "dorado._native")]
struct RngStreams {
    seed: u64,
    streams: Vec<Stream>,
    pools: HashMap<usize, rayon::ThreadPool>,
}

impl RngStreams {
    fn grow(&mut self, count: usize) {
        while self.streams.len() < count {
            let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
            rng.set_stream(self.streams.len() as u64);
            self.streams.push(Stream { rng, draws: 0 });
        }
    }
}

#[pymethods]
impl RngStreams {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> PyResult<Self> {
        let seed = match seed {
            Some(seed) => seed,
            None => {
                let mut bytes = [0u8; 8];
                rand::RngCore::try_fill_bytes(&mut rand::rngs::OsRng, &mut bytes).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("OS entropy failed: {e}"))
                })?;
                u64::from_le_bytes(bytes)
            }
        };
        Ok(Self {
            seed,
            streams: Vec::new(),
            pools: HashMap::new(),
        })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }

    #[getter]
    fn draw_counts(&self) -> Vec<u64> {
        self.streams.iter().map(|s| s.draws).collect()
    }

    /// Diagnostic for parity tests: consume a particle stream's uniform draws.
    fn uniforms(&mut self, particle_id: usize, count: usize) -> PyResult<Vec<f64>> {
        let length = particle_id
            .checked_add(1)
            .ok_or_else(|| PyValueError::new_err("particle_id overflow"))?;
        self.grow(length);
        let stream = &mut self.streams[particle_id];
        Ok((0..count)
            .map(|_| {
                stream.draws += 1;
                stream.rng.gen::<f64>()
            })
            .collect())
    }
}

type HistoryRow = (i64, i64, f64, bool, bool, f64);

/// Adaptive histories are materialized as Python lists only when requested.
#[pyclass(frozen, module = "dorado._native")]
struct StepResult {
    outcomes: Vec<Outcome>,
}

#[pymethods]
impl StepResult {
    #[getter]
    fn recorded(&self) -> Vec<bool> {
        self.outcomes.iter().map(|o| o.recorded).collect()
    }
    #[getter]
    fn hit_max_iter(&self) -> Vec<bool> {
        self.outcomes.iter().map(|o| o.hit_max_iter).collect()
    }
    #[getter]
    fn histories(&self) -> Vec<Vec<HistoryRow>> {
        self.outcomes
            .iter()
            .map(|o| {
                o.history
                    .iter()
                    .map(|p| {
                        (
                            p.position[0],
                            p.position[1],
                            p.time,
                            p.deposited,
                            p.stranded,
                            p.stranded_time,
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

/// Mutate positions, times, and [deposited, stranded, stranded_time] in place.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (fields, positions, travel_times, particle_state, rng, *,
    behavior=None, n_threads=1, target_time=None, est_next_dt=None, max_iter=10000))]
fn step_particles(
    fields: &RoutingFields,
    mut positions: PyReadwriteArray2<'_, i64>,
    mut travel_times: PyReadwriteArray1<'_, f64>,
    mut particle_state: PyReadwriteArray2<'_, f64>,
    rng: &mut RngStreams,
    behavior: Option<PyReadonlyArray2<'_, f64>>,
    n_threads: usize,
    target_time: Option<f64>,
    est_next_dt: Option<PyReadonlyArray1<'_, f64>>,
    max_iter: usize,
) -> PyResult<StepResult> {
    let n = positions.shape()[0];
    require(
        positions.shape()[1] == 2,
        "positions must have shape (N, 2)",
    )?;
    require(
        travel_times.shape() == [n],
        "travel_times must have shape (N,)",
    )?;
    require(
        particle_state.shape() == [n, 3],
        "particle_state must have shape (N, 3)",
    )?;
    require(
        positions.is_c_contiguous() && particle_state.is_c_contiguous(),
        "mutable arrays must be C-contiguous",
    )?;
    require(
        (1..=256).contains(&n_threads),
        "n_threads must be in [1, 256]",
    )?;
    require(
        max_iter > 0 && max_iter < usize::MAX,
        "max_iter must be positive and below usize::MAX",
    )?;
    if let Some(t) = target_time {
        finite(t, "target_time")?;
    }
    let behavior = behavior
        .as_ref()
        .map(|b| -> PyResult<&[f64]> {
            require(
                b.shape() == [n, 3] && b.is_c_contiguous(),
                "behavior must be C-contiguous with shape (N, 3)",
            )?;
            Ok(b.as_slice()?)
        })
        .transpose()?;
    let estimates = est_next_dt
        .as_ref()
        .map(|e| -> PyResult<&[f64]> {
            require(e.shape() == [n], "est_next_dt must have shape (N,)")?;
            Ok(e.as_slice()?)
        })
        .transpose()?;
    let positions = positions.as_slice_mut()?;
    let times = travel_times.as_slice_mut()?;
    let states = particle_state.as_slice_mut()?;
    // Validate the whole batch before advancing any particle or RNG.
    for i in 0..n {
        require(
            fields
                .inner
                .contains([positions[2 * i], positions[2 * i + 1]]),
            "particle position outside domain",
        )?;
        finite(times[i], "travel_times")?;
        let state = &states[3 * i..3 * i + 3];
        require(
            (state[0] == 0.0 || state[0] == 1.0)
                && (state[1] == 0.0 || state[1] == 1.0)
                && state[2].is_finite()
                && state[2] >= 0.0,
            "invalid particle_state flags or stranded time",
        )?;
        if let Some(b) = behavior {
            let b = &b[3 * i..3 * i + 3];
            require(
                b.iter().all(|v| v.is_finite())
                    && b[0] > 0.0
                    && b[1] >= 0.0
                    && (0.0..=1.0).contains(&b[2]),
                "invalid transport multipliers or deposition probability",
            )?;
        }
        if let Some(e) = estimates {
            finite(e[i], "est_next_dt")?;
        }
    }
    if n_threads > 1 && !rng.pools.contains_key(&n_threads) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "thread pool creation failed: {e}"
                ))
            })?;
        rng.pools.insert(n_threads, pool);
    }
    rng.grow(n);
    let run =
        |i: usize, position: &mut [i64], time: &mut f64, state: &mut [f64], stream: &mut Stream| {
            let mut particle = Particle {
                position: [position[0], position[1]],
                time: *time,
                deposited: state[0] == 1.0,
                stranded: state[1] == 1.0,
                stranded_time: state[2],
            };
            let b = behavior
                .map(|b| Behavior {
                    time_mult: b[3 * i],
                    diff_mult: b[3 * i + 1],
                    deposition_prob: b[3 * i + 2],
                })
                .unwrap_or_default();
            let mut draw = || {
                stream.draws += 1;
                stream.rng.gen::<f64>()
            };
            let result = routing::advance(
                &mut particle,
                &fields.inner,
                b,
                &mut draw,
                target_time,
                estimates.map(|e| e[i]).unwrap_or(0.1),
                max_iter,
            );
            position.copy_from_slice(&particle.position);
            *time = particle.time;
            state.copy_from_slice(&[
                f64::from(particle.deposited),
                f64::from(particle.stranded),
                particle.stranded_time,
            ]);
            result
        };
    // The GIL remains held while NumPy memory is borrowed. Only Rust's scoped
    // workers touch disjoint mutable particle rows; fields are owned/read-only.
    let outcomes = if n_threads == 1 {
        positions
            .chunks_exact_mut(2)
            .zip(times)
            .zip(states.chunks_exact_mut(3))
            .zip(&mut rng.streams[..n])
            .enumerate()
            .map(|(i, (((p, t), s), r))| run(i, p, t, s, r))
            .collect()
    } else {
        let streams = &mut rng.streams[..n];
        rng.pools[&n_threads].install(|| {
            positions
                .par_chunks_exact_mut(2)
                .zip(times.par_iter_mut())
                .zip(states.par_chunks_exact_mut(3))
                .zip(streams.par_iter_mut())
                .enumerate()
                .map(|(i, (((p, t), s), r))| run(i, p, t, s, r))
                .collect()
        })
    };
    Ok(StepResult { outcomes })
}

/// Private deterministic seam for cross-checking NumPy's CDF helper.
#[pyfunction]
#[pyo3(signature = (weights, uniform, steepest=false))]
fn _pick(weights: Vec<f64>, uniform: f64, steepest: bool) -> PyResult<usize> {
    require(weights.len() == 9, "weights must have length 9")?;
    require(
        uniform.is_finite() && (0.0..1.0).contains(&uniform),
        "uniform must be in [0, 1)",
    )?;
    require(
        weights
            .iter()
            .all(|w| w.is_nan() || (w.is_finite() && *w >= 0.0))
            && weights
                .iter()
                .filter(|w| !w.is_nan())
                .sum::<f64>()
                .is_finite(),
        "invalid weights",
    )?;
    Ok(routing::pick(&weights, uniform, steepest))
}

#[pyfunction]
fn kernel_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RoutingFields>()?;
    module.add_class::<RngStreams>()?;
    module.add_class::<StepResult>()?;
    module.add_function(wrap_pyfunction!(step_particles, module)?)?;
    module.add_function(wrap_pyfunction!(_pick, module)?)?;
    module.add_function(wrap_pyfunction!(kernel_version, module)?)?;
    Ok(())
}
