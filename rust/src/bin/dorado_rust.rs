//! CLI driver for the parity kernel: runs steady-mode routing on a
//! normalized fixture directory, mirroring `benchmarks/run_reference.py`'s
//! contract (same fixture layout, same output files, same timed region).
//!
//! ```text
//! dorado_rust --input bench/fixture_rcm/ --particles 10000 \
//!     --iterations 100 --seed 0 --out out/rust/
//! ```
//!
//! `--particles`, `--iterations`, and `--seed` default to the fixture
//! manifest values; `--seed-x`/`--seed-y` (comma-separated) override the
//! default seeding regions.

use _native::driver::run_and_write;
use std::path::PathBuf;

fn usage(prog: &str) -> ! {
    eprintln!(
        "usage: {prog} --input <fixture_dir> --out <out_dir> \
         [--particles N] [--iterations N] [--seed S] \
         [--seed-x CSV] [--seed-y CSV]"
    );
    std::process::exit(2);
}

fn main() {
    let prog = std::env::args()
        .next()
        .unwrap_or_else(|| "dorado_rust".to_owned());
    let mut input: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut particles: Option<usize> = None;
    let mut iterations: Option<usize> = None;
    let mut seed: Option<u32> = None;
    let mut seed_x: Option<String> = None;
    let mut seed_y: Option<String> = None;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it.next().unwrap_or_else(|| usage(&prog));
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--particles" => particles = value.parse().ok().or_else(|| usage(&prog)),
            "--iterations" => iterations = value.parse().ok().or_else(|| usage(&prog)),
            "--seed" => seed = value.parse().ok().or_else(|| usage(&prog)),
            "--seed-x" => seed_x = Some(value.clone()),
            "--seed-y" => seed_y = Some(value.clone()),
            _ => usage(&prog),
        }
    }
    let (input, out) = match (input, out) {
        (Some(i), Some(o)) => (i, o),
        _ => usage(&prog),
    };

    match run_and_write(
        &input,
        &out,
        particles,
        iterations,
        seed,
        seed_x.as_deref(),
        seed_y.as_deref(),
    ) {
        Ok(summary) => {
            println!("walk data: {}", summary.walk_data_path.display());
            println!("timing: {}", summary.timing_path.display());
            println!(
                "compute seconds: {:.6} ({} appended steps, {} particles, {} iterations)",
                summary.seconds, summary.appended_steps, summary.particles, summary.iterations
            );
        }
        Err(e) => {
            eprintln!("dorado_rust: {e}");
            std::process::exit(1);
        }
    }
}
