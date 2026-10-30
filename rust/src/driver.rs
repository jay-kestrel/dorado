//! Fixture loading, run, and output writing shared by the bin driver and the
//! PyO3 surface. Mirrors `benchmarks/run_reference.py`: the fixture is read
//! untimed, the timed region covers field derivation + particle seeding +
//! the iteration loop, and walk-data/timing JSON are written untimed.

use crate::fields::{D8Tables, Fields, Model, Params, ReleaseRule, StrandingHandling};
use crate::json_out::write_walk_data;
use crate::npy::read_npy;
use crate::walker::{run_steady_walk, SeedMethod};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Default seeding regions per fixture field, following
/// `benchmarks/run_reference.py` (SEED_BOXES) and the routing
/// benchmark table.
const SEED_BOXES: [(&str, &str, &str); 2] = [
    ("rcm", "15,16", "137,138,139"),
    ("anuga", "20,21,22,23,24,25,26,27,28,29", "48,49,50,51,52"),
];

const TIMED_REGION: &str = "Particles init (velocity/cell-type/D8/weight build) + \
     generate_particles + run_iteration loop; npy/params load and \
     walk-data/timing JSON writes excluded";

/// A loaded fixture: raw grid arrays plus resolved run parameters.
/// Field derivation happens later, inside the timed region.
pub struct Fixture {
    pub params: Params,
    pub stage: Vec<f64>,
    pub depth: Vec<f64>,
    pub qx: Vec<f64>,
    pub qy: Vec<f64>,
    pub cell_type: Option<Vec<f64>>,
    pub l: usize,
    pub w: usize,
    pub model_name: String,
    pub fixture_dir: PathBuf,
    pub particles: usize,
    pub iterations: usize,
    pub seed: u32,
    pub seed_x: Vec<i64>,
    pub seed_y: Vec<i64>,
}

/// Outcome of one timed run plus the paths it produced.
pub struct RunSummary {
    pub seconds: f64,
    pub appended_steps: i64,
    pub particles: usize,
    pub iterations: usize,
    pub seed: u32,
    pub walk_data_path: PathBuf,
    pub timing_path: PathBuf,
}

/// Pull a scalar field out of the flat params JSON. The fixture manifest is
/// a flat dict of numbers/bools/strings (plus the `arrays` map), so a
/// quoted-key scan is sufficient for this format.
fn json_field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\":");
    let start = json.find(&pat)? + pat.len();
    let rest = json[start..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn json_number(json: &str, key: &str) -> Option<f64> {
    json_field(json, key)?.parse().ok()
}

fn json_bool(json: &str, key: &str) -> Option<bool> {
    json_field(json, key)?.parse().ok()
}

fn json_string(json: &str, key: &str) -> Option<String> {
    let raw = json_field(json, key)?;
    Some(raw.trim_matches('"').to_owned())
}

/// Comma-separated index list ("15,16") into i64s.
fn csv_indices(spec: &str) -> Result<Vec<i64>, String> {
    spec.split(',')
        .map(|s| {
            s.trim()
                .parse::<i64>()
                .map_err(|e| format!("bad index '{s}': {e}"))
        })
        .collect()
}

/// Default seeding regions for a fixture named `fixture_<field>`; explicit
/// CSV overrides win, mirroring the reference runner's CLI precedence.
fn seed_regions(
    fixture_dir: &Path,
    seed_x: Option<&str>,
    seed_y: Option<&str>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if let (Some(x), Some(y)) = (seed_x, seed_y) {
        return Ok((csv_indices(x)?, csv_indices(y)?));
    }
    let field = fixture_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .trim_start_matches("fixture_")
        .to_owned();
    let found = SEED_BOXES.iter().find(|(name, _, _)| *name == field);
    if seed_x.is_some() || seed_y.is_some() {
        return Err("explicit seeding requires both --seed-x and --seed-y".to_owned());
    }
    let (_, default_x, default_y) = found
        .ok_or_else(|| format!("unknown fixture '{field}': pass --seed-x/--seed-y explicitly"))?;
    Ok((csv_indices(default_x)?, csv_indices(default_y)?))
}

/// Load one normalized fixture directory (params.json + uncompressed .npy
/// arrays). I/O only — nothing here is timed.
pub fn load_fixture(
    input_dir: &Path,
    particles: Option<usize>,
    iterations: Option<usize>,
    seed: Option<u32>,
    seed_x: Option<&str>,
    seed_y: Option<&str>,
) -> Result<Fixture, String> {
    let manifest_path = input_dir.join("params.json");
    let json = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("reading {}: {e}", manifest_path.display()))?;
    let num = |key: &str| {
        json_number(&json, key).ok_or_else(|| format!("params.json missing number '{key}'"))
    };

    let dx = num("dx")?;
    let theta = num("theta")?;
    let gamma = num("gamma")?;
    let diff_coeff = num("diff_coeff")?;
    let dry_depth = num("dry_depth")?;
    let steepest_descent = json_bool(&json, "steepest_descent").unwrap_or(false);
    let model_name =
        json_string(&json, "model").ok_or_else(|| "params.json missing 'model'".to_owned())?;
    let model = match model_name.as_str() {
        "DeltaRCM" => Model::DeltaRCM,
        _ => Model::Other,
    };

    let array_file = |key: &str| -> Result<PathBuf, String> {
        let name = json_string(&json, key)
            .ok_or_else(|| format!("params.json missing arrays entry '{key}'"))?;
        Ok(input_dir.join(name))
    };
    let stage_arr = read_npy(array_file("stage")?)?;
    let depth_arr = read_npy(array_file("depth")?)?;
    let qx_arr = read_npy(array_file("qx")?)?;
    let qy_arr = read_npy(array_file("qy")?)?;
    let cell_arr = match json_string(&json, "cell_type") {
        Some(_) => Some(read_npy(array_file("cell_type")?)?),
        None => None,
    };
    let l = depth_arr.rows();
    let w = depth_arr.cols();
    let cell_type = cell_arr.map(|a| a.data);

    // Reference constructor defaults: park handling, immediate release,
    // rest_time 0.0, entrain_prob 1.0, stranding_dt 1.0 (see
    // check_stranding_handling in dorado/particle_track.py).
    let params = Params {
        dx,
        theta,
        gamma,
        diff_coeff,
        dry_depth,
        steepest_descent,
        model,
        stranding_handling: StrandingHandling::Park,
        stranding_release: ReleaseRule::Immediate,
        rest_time: 0.0,
        entrain_prob: 1.0,
        stranding_dt: 1.0,
    };

    let particles = match particles {
        Some(p) => p,
        None => num("Np_tracer")? as usize,
    };
    let iterations = match iterations {
        Some(i) => i,
        None => num("iterations")? as usize,
    };
    let seed = seed.unwrap_or_else(|| num("seed").unwrap_or(0.0) as u32);
    if particles == 0 || iterations == 0 {
        return Err("--particles and --iterations must be positive".to_owned());
    }
    let (seed_x, seed_y) = seed_regions(input_dir, seed_x, seed_y)?;

    Ok(Fixture {
        params,
        stage: stage_arr.data,
        depth: depth_arr.data,
        qx: qx_arr.data,
        qy: qy_arr.data,
        cell_type,
        l,
        w,
        model_name,
        fixture_dir: input_dir.to_owned(),
        particles,
        iterations,
        seed,
        seed_x,
        seed_y,
    })
}

/// The reference runner's whole timed region on a loaded fixture: field
/// derivation, particle seeding, and the steady-mode iteration loop.
pub fn run_fixture(fixture: &Fixture) -> Result<crate::walker::WalkData, String> {
    let fields = Fields::derive(
        &fixture.params,
        fixture.l,
        fixture.w,
        &fixture.stage,
        &fixture.depth,
        &fixture.qx,
        &fixture.qy,
        fixture.cell_type.clone(),
        D8Tables::standard(),
    )?;
    Ok(run_steady_walk(
        &fields,
        fixture.seed,
        fixture.particles,
        &fixture.seed_x,
        &fixture.seed_y,
        SeedMethod::Random,
        fixture.iterations,
    ))
}

fn appended_steps(walk: &crate::walker::WalkData) -> i64 {
    walk.xinds.iter().map(|path| path.len() as i64 - 1).sum()
}

/// UTC timestamp in the reference's `datetime.now(UTC).isoformat()` shape
/// (second precision, `+00:00`), without pulling in a date library.
fn timestamp_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil_from_days: days since 1970-01-01 -> y/m/d.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}+00:00")
}

fn first_line(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next().map(str::to_owned)
}

/// Timing metadata mirroring the reference runner's report schema
/// (`benchmarks/run_reference.py`), with the C port's field order and a
/// `toolchain` field where C reports `compiler`.
fn timing_json(fixture: &Fixture, seconds: f64, appended: i64) -> Result<String, String> {
    if !seconds.is_finite() {
        return Err("non-finite timing".to_owned());
    }
    let revision =
        first_line("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let platform = first_line("uname", &["-s", "-r", "-m"]).unwrap_or_else(|| "unknown".to_owned());
    let toolchain =
        first_line("rustc", &["--version"]).unwrap_or_else(|| "rustc unknown".to_owned());
    // Python repr form for the seconds value (same formatting as walk data).
    let fmt_seconds = if seconds == 0.0 {
        "0.0".to_owned()
    } else if seconds == seconds.trunc() && seconds.abs() < 1e16 {
        format!("{seconds:.1}")
    } else {
        seconds.to_string()
    };
    Ok(format!(
        "{{\n  \"schema_version\": 1,\n  \"implementation\": \"rust\",\n  \"fixture_dir\": \"{}\",\n  \"model\": \"{}\",\n  \"particles\": {},\n  \"iterations\": {},\n  \"seed\": {},\n  \"seconds\": {},\n  \"appended_steps\": {},\n  \"timed_region\": \"{}\",\n  \"timestamp_utc\": \"{}\",\n  \"revision\": \"{}\",\n  \"toolchain\": \"{}\",\n  \"platform\": \"{}\"\n}}\n",
        fixture.fixture_dir.display(),
        fixture.model_name,
        fixture.particles,
        fixture.iterations,
        fixture.seed,
        fmt_seconds,
        appended,
        TIMED_REGION,
        timestamp_utc(),
        revision,
        toolchain,
        platform,
    ))
}

/// Load a fixture, run the timed region, and write `walk_data.json` +
/// `timing.json` under `out_dir`. Used by the bin driver and the PyO3
/// surface alike.
pub fn run_and_write(
    input_dir: &Path,
    out_dir: &Path,
    particles: Option<usize>,
    iterations: Option<usize>,
    seed: Option<u32>,
    seed_x: Option<&str>,
    seed_y: Option<&str>,
) -> Result<RunSummary, String> {
    let fixture = load_fixture(input_dir, particles, iterations, seed, seed_x, seed_y)?;
    let t0 = Instant::now();
    let walk = run_fixture(&fixture)?;
    let seconds = t0.elapsed().as_secs_f64();
    let appended = appended_steps(&walk);

    fs::create_dir_all(out_dir).map_err(|e| format!("creating {}: {e}", out_dir.display()))?;
    let walk_path = out_dir.join("walk_data.json");
    write_walk_data(&walk_path, &walk)?;
    let timing_path = out_dir.join("timing.json");
    fs::write(&timing_path, timing_json(&fixture, seconds, appended)?)
        .map_err(|e| format!("writing {}: {e}", timing_path.display()))?;

    Ok(RunSummary {
        seconds,
        appended_steps: appended,
        particles: fixture.particles,
        iterations: fixture.iterations,
        seed: fixture.seed,
        walk_data_path: walk_path,
        timing_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_json_scalar_extraction() {
        let json = r#"{"dx": 50.0, "theta": 1.0, "steepest_descent": false,
            "model": "DeltaRCM", "seed": 0, "arrays": {"depth": "depth.f64.npy"}}"#;
        assert_eq!(json_number(json, "dx"), Some(50.0));
        assert_eq!(json_bool(json, "steepest_descent"), Some(false));
        assert_eq!(json_string(json, "model").as_deref(), Some("DeltaRCM"));
        // Inner keys of the arrays map are findable too — the driver relies on
        // this for the array filenames.
        assert_eq!(json_string(json, "depth").as_deref(), Some("depth.f64.npy"));
        assert_eq!(json_number(json, "missing"), None);
    }

    #[test]
    fn csv_index_parsing() {
        assert_eq!(csv_indices("15,16").unwrap(), vec![15, 16]);
        assert!(csv_indices("15, x").is_err());
    }

    #[test]
    fn seed_region_defaults_and_overrides() {
        let dir = Path::new("/tmp/fixture_rcm");
        let (x, y) = seed_regions(dir, None, None).unwrap();
        assert_eq!(x, vec![15, 16]);
        assert_eq!(y, vec![137, 138, 139]);
        let (x, y) = seed_regions(Path::new("/tmp/fixture_anuga"), None, None).unwrap();
        assert_eq!(x.len(), 10);
        assert_eq!(y, vec![48, 49, 50, 51, 52]);
        let (x, y) = seed_regions(dir, Some("1,2"), Some("3")).unwrap();
        assert_eq!((x, y), (vec![1, 2], vec![3]));
        assert!(seed_regions(dir, Some("1,2"), None).is_err());
        assert!(seed_regions(Path::new("/tmp/fixture_zzz"), None, None).is_err());
    }

    #[test]
    fn utc_timestamp_shape() {
        let ts = timestamp_utc();
        // 2026-10-30T12:34:56+00:00
        assert_eq!(ts.len(), 25);
        assert!(ts.ends_with("+00:00"));
        assert_eq!(&ts[4..5], "-");
        assert_eq!(&ts[10..11], "T");
    }
}
