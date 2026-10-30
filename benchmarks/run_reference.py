"""Run the Python reference on normalized fixture bytes; time the compute region.

Loads one normalized fixture directory (params.json + uncompressed .npy arrays
from ``normalize_fixtures.py``), then times everything the native ports must
reimplement: ``Particles`` init (velocity/cell-type/D8/weight derivation),
``generate_particles``, and the ``run_iteration`` loop. Loading arrays and
parsing params happen before the clock starts; writing walk-data and timing
JSON happens after it stops, so the reported seconds exclude all I/O.

The walk-data JSON is the reference output the C and Rust binaries must
reproduce from the same bytes; the timing JSON carries the seed, workload, and
environment provenance the comparison report aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Prefer this checkout, not an unrelated installed dorado distribution.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the sys.path bootstrap so the script runs from a source
# checkout without installing the package.
from dorado import cobol_walk as cw  # noqa: E402
from dorado.particle_track import Particles, modelParams  # noqa: E402

# Default seeding regions per field, following example_data/define_params.py
# and the seed rows/columns of the routing benchmark table.
SEED_BOXES = {
    "rcm": (list(range(15, 17)), list(range(137, 140))),
    "anuga": (list(range(20, 30)), list(range(48, 53))),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(fixture_dir: Path) -> tuple[modelParams, dict]:
    """Build modelParams from the fixture bytes; arrays load untimed."""
    manifest = json.loads((fixture_dir / "params.json").read_text())
    params = modelParams()
    for key in ("depth", "stage", "qx", "qy", "cell_type"):
        setattr(params, key, np.load(fixture_dir / manifest["arrays"][key]))
    for key in ("dx", "theta", "gamma", "diff_coeff", "dry_depth",
                "steepest_descent", "model"):
        setattr(params, key, manifest[key])
    return params, manifest


def run_reference(params: modelParams, seed: int, particles: int, iterations: int,
                  seed_x: list[int], seed_y: list[int]) -> tuple[dict, float]:
    """Timed region: init derivation, particle seeding, and all iterations."""
    start = time.perf_counter()
    particle = Particles(params)
    np.random.seed(seed)
    particle.generate_particles(particles, seed_x, seed_y)
    walk_data = None
    for _ in range(iterations):
        walk_data = particle.run_iteration()
    return walk_data, time.perf_counter() - start


def jsonable(value):
    """Convert numpy scalars/arrays so json.dumps can serialize walk data."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def appended_steps(walk_data: dict) -> int:
    return sum(len(path) - 1 for path in walk_data["xinds"])


def metadata(input_dir: Path, manifest: dict, particles: int, iterations: int,
             seed: int, seconds: float, moved: int) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    report = {
        "schema_version": 1,
        "implementation": ("cobol" if cw.cobol_active() else "python"),
        "fixture_dir": str(input_dir),
        "model": manifest["model"],
        "particles": particles,
        "iterations": iterations,
        "seed": seed,
        "seconds": seconds,
        "appended_steps": moved,
        "timed_region": ("Particles init (velocity/cell-type/D8/weight build) + "
                         "generate_particles + run_iteration loop; npy/params "
                         "load and walk-data/timing JSON writes excluded"),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "routing_source_sha256": {
            name: sha256(ROOT / "dorado" / name)
            for name in ("particle_track.py", "lagrangian_walker.py")
        },
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
    }
    if cw.cobol_active():
        report["gnucobol_version"] = cw.gnucobol_version()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="normalized fixture directory (params.json + npy files)")
    parser.add_argument("--particles", type=int,
                        help="override the manifest Np_tracer")
    parser.add_argument("--iterations", type=int,
                        help="override the manifest iteration count")
    parser.add_argument("--seed", type=int, help="override the manifest seed")
    parser.add_argument("--seed-x", help="comma-separated x seeding indices")
    parser.add_argument("--seed-y", help="comma-separated y seeding indices")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "out" / "python")
    args = parser.parse_args()

    params, manifest = load_inputs(args.input)
    particles = args.particles if args.particles is not None else manifest["Np_tracer"]
    iterations = args.iterations if args.iterations is not None else manifest["iterations"]
    seed = args.seed if args.seed is not None else manifest["seed"]
    if not 0 <= seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if particles <= 0 or iterations <= 0:
        parser.error("--particles and --iterations must be positive")

    field = args.input.resolve().name.removeprefix("fixture_")
    if args.seed_x is None or args.seed_y is None:
        if field not in SEED_BOXES:
            parser.error(f"unknown fixture '{field}': pass --seed-x/--seed-y explicitly")
        default_x, default_y = SEED_BOXES[field]
        seed_x = ([int(v) for v in args.seed_x.split(",")] if args.seed_x else default_x)
        seed_y = ([int(v) for v in args.seed_y.split(",")] if args.seed_y else default_y)
    else:
        seed_x = [int(v) for v in args.seed_x.split(",")]
        seed_y = [int(v) for v in args.seed_y.split(",")]

    walk_data, seconds = run_reference(params, seed, particles, iterations, seed_x, seed_y)
    moved = appended_steps(walk_data)

    args.out.mkdir(parents=True, exist_ok=True)
    walk_path = args.out / "walk_data.json"
    walk_path.write_bytes(json.dumps(jsonable(walk_data), allow_nan=False).encode())
    report = metadata(args.input, manifest, particles, iterations, seed, seconds, moved)
    timing_path = args.out / "timing.json"
    timing_path.write_bytes((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())

    print(f"walk data: {walk_path}")
    print(f"timing: {timing_path}")
    print(f"compute seconds: {seconds:.6f} "
          f"({moved} appended steps, {particles} particles, {iterations} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
