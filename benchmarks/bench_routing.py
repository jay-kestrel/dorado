"""Wall-clock routing benchmark; run from any directory with --output PATH."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Prefer this checkout, not an unrelated installed dorado distribution.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the sys.path bootstrap so the script runs from a source
# checkout without installing the package.
from dorado import lagrangian_walker as lw  # noqa: E402
from dorado import cobol_walk as cw  # noqa: E402
from dorado.particle_track import Particles, modelParams  # noqa: E402


@dataclass
class Workload:
    name: str
    params: modelParams
    seed_x: list[int]
    seed_y: list[int]
    steps: int
    data_sha256: str


def load_workload(name: str) -> Workload:
    """Load one bundled field once, following example_data/define_params.py."""
    filename = "ex_deltarcm_data.npz" if name == "rcm" else "ex_anuga_data.npz"
    path = ROOT / "dorado" / "example_data" / filename
    params = modelParams()
    with np.load(path) as data:
        params.depth = data["depth"]
        params.stage = data["stage"] if name == "rcm" else params.depth.copy()
        params.qx = np.zeros_like(params.depth) if name == "rcm" else data["qx"]
        params.qy = np.zeros_like(params.depth) if name == "rcm" else data["qy"]
    params.theta = 1.0
    params.dx = 50.0 if name == "rcm" else 10.0
    params.model = "DeltaRCM" if name == "rcm" else "Anuga"
    return Workload(
        name, params,
        list(range(15, 17)) if name == "rcm" else list(range(20, 30)),
        list(range(137, 140)) if name == "rcm" else list(range(48, 53)),
        20 if name == "rcm" else 50,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def rust_available() -> bool:
    """A native module alone is insufficient: the public switch must exist too."""
    return (
        "backend" in inspect.signature(Particles).parameters
        and importlib.util.find_spec("dorado._native") is not None
    )


def cobol_available() -> bool:
    """The COBOL backend needs its built shared object on disk."""
    return (ROOT / "dorado" / "lib" / "dorado_cobol.so").exists()


def stage_particles(workload: Workload, count: int, backend: str, seed: int) -> Particles:
    np.random.seed(seed)
    kwargs = {}
    if "backend" in inspect.signature(Particles).parameters:
        kwargs = {"backend": "rust" if backend == "rust" else "python",
                  "seed": seed, "n_threads": 1}
    particles = Particles(workload.params, **kwargs)
    particles.generate_particles(count, workload.seed_x, workload.seed_y)
    return particles


def route(particles: Particles, steps: int) -> None:
    for _ in range(steps):
        particles.run_iteration()


def appended_steps(particles: Particles) -> int:
    return sum(len(path) - 1 for path in particles.walk_data["xinds"])


def measure(staged: Particles, steps: int, repeats: int, seed: int) -> tuple[list[float], list[int]]:
    """Copy pristine staged state before every run; time only run_iteration."""
    warmup = copy.deepcopy(staged)
    np.random.seed(seed)
    route(warmup, 1)
    del warmup
    elapsed, moved = [], []
    for _ in range(repeats):
        particles = copy.deepcopy(staged)
        np.random.seed(seed)
        start = time.perf_counter()
        route(particles, steps)
        elapsed.append(time.perf_counter() - start)
        moved.append(appended_steps(particles))
    return elapsed, moved


def benchmark(workload: Workload, count: int, backend: str, steps: int,
              repeats: int, seed: int) -> dict:
    staged = stage_particles(workload, count, backend, seed)
    context = nullcontext()
    if backend == "numpy":
        from benchmarks.numpy_stepper import make_stepper
        context = patch.object(lw, "particle_stepper", make_stepper(staged))
    elif backend == "cobol":
        # the walk core dispatches on DORADO_COBOL inside
        # particle_stepper; a missing .so raises there rather than
        # falling back silently
        context = patch.dict(os.environ, {"DORADO_COBOL": "1"})
    with context:
        elapsed, moved = measure(staged, steps, repeats, seed)
    median = statistics.median(elapsed)
    row = {
        "workload": workload.name, "shape": list(workload.params.depth.shape),
        "dx": workload.params.dx, "data_sha256": workload.data_sha256,
        "backend": backend, "particles": count, "steps": steps,
        "repeats": repeats, "seed": seed, "n_threads": 1,
        "seconds": elapsed, "median_seconds": median,
        "microseconds_per_particle_step": median * 1e6 / (count * steps),
        "appended_steps": moved,
    }
    if backend == "cobol":
        row["implementation"] = "cobol"
        row["gnucobol_version"] = cw.gnucobol_version()
    return row


def metadata() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "routing_source_sha256": {
            name: hashlib.sha256((ROOT / "dorado" / name).read_bytes()).hexdigest()
            for name in ("particle_track.py", "lagrangian_walker.py")
        },
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "timer": "time.perf_counter", "timed_region": "run_iteration only",
        "warmup_iterations": 1,
        "state_reset": "deepcopy staged particles and reset RNG outside timing",
        "normalization": "attempted particle-steps, including frozen particles",
        "thread_environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def table(results: list[dict]) -> str:
    lines = ["| Field | Backend | Particles | Steps | Median (s) | us/particle-step |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in results:
        lines.append(
            f"| {row['workload']} | {row['backend']} | {row['particles']} | "
            f"{row['steps']} | {row['median_seconds']:.6f} | "
            f"{row['microseconds_per_particle_step']:.3f} |")
    return "\n".join(lines)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="ANUGA, 500 particles, 3 steps, 1 repeat")
    parser.add_argument("--backend", nargs="+", choices=("python", "numpy", "rust", "cobol"))
    parser.add_argument("--steps", type=positive_int, help="override both field iteration counts")
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if os.environ.get("DORADO_BACKEND"):
        parser.error("unset DORADO_BACKEND so the benchmark can select each backend")
    backends = args.backend or ["python", "numpy", "rust"]
    skipped = []
    if "rust" in backends and not rust_available():
        reason = "Rust backend not implemented/installed; no native timing recorded"
        if args.backend:
            parser.error(reason)
        backends.remove("rust")
        skipped.append({"backend": "rust", "reason": reason})
    if "cobol" in backends and not cobol_available():
        reason = "COBOL kernels not built; run `make -C cobol` (requires gnucobol3)"
        if args.backend:
            parser.error(reason)
        backends.remove("cobol")
        skipped.append({"backend": "cobol", "reason": reason})
    workloads = [load_workload(name) for name in (["anuga"] if args.quick else ["rcm", "anuga"])]
    counts = [500] if args.quick else [500, 2500, 5000]
    repeats = args.repeats or (1 if args.quick else 3)
    report = {"schema_version": 1, "metadata": metadata(), "skipped": skipped, "results": []}
    for workload in workloads:
        steps = args.steps or (3 if args.quick else workload.steps)
        for count in counts:
            for backend in backends:
                report["results"].append(benchmark(workload, count, backend, steps, repeats, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(table(report["results"]))
    for entry in skipped:
        print(f"Skipped {entry['backend']}: {entry['reason']}")
    print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
