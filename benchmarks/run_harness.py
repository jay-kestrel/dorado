"""Run python, C, and Rust implementations on identical fixture bytes.

The head-to-head rig from the spec ("Two harnesses", locked decision 4): one
normalized fixture directory feeds all three implementations; parity is
byte-identical walk-data JSON against the Python reference; timing is warm-state
medians over ``--repeats`` (default 5). Compute seconds come from each
implementation's own clock (port binaries report ``timing.json`` seconds with
I/O excluded; the Python reference is timed in-process by
``run_reference.run_reference`` over the same region). Wall time around each
whole port process is recorded separately — that is the price a user actually
pays, startup and output I/O included. dorado has no JIT, so there is no
cold-start line item.

A parity mismatch fails the run (nonzero exit, loud report, parity status in
results.json) and is never resolved by loosening tolerance: walk data is
byte-exact by design, so a difference is a port bug, not a tolerance issue.

Usage (mirrors the spec's rig):
    python benchmarks/run_harness.py --input bench/fixture_rcm/ \
        --out bench/out/harness/
    python benchmarks/run_harness.py --particles 500 --iterations 20 --repeats 3

Both native binaries are built fresh (CMake Release; cargo release) unless
``--c-binary``/``--rust-binary`` point at prebuilt ones. Workload defaults
(particles, iterations, seed, seeding region) come from the fixture manifest,
so the harness and ``run_reference.py`` benchmark the same workload by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

import numpy as np

# Prefer this checkout, not an unrelated installed dorado distribution.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import run_reference as ref  # noqa: E402

# Warmup size: same code paths as the real workload, bounded cost. Runs untimed
# before the timed repetitions so first-touch costs (imports, page cache,
# allocator) stay out of the medians.
WARMUP_PARTICLES = 200
WARMUP_ITERATIONS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_environment() -> None:
    """Refuse env overrides that would mislabel or hijack the python reference.

    ``DORADO_BACKEND`` overrides the Particles constructor argument
    (particle_track._resolve_backend), so run_reference's explicit
    ``backend="python"`` would be silently swapped for the native kernel;
    ``DORADO_COBOL=1`` swaps the walker for the COBOL experiment.
    """
    if os.environ.get("DORADO_COBOL") == "1":
        raise SystemExit("DORADO_COBOL=1 would hijack the python reference; "
                         "unset it first")
    backend = os.environ.get("DORADO_BACKEND")
    if backend is not None and backend != "python":
        raise SystemExit(f"DORADO_BACKEND={backend!r} overrides the constructor "
                         "argument and would swap the python reference for "
                         "another backend; unset it or set DORADO_BACKEND=python")


def build_c_port(build_dir: Path) -> Path:
    """Configure and build the C port (Release); return the driver path."""
    subprocess.run(["cmake", "-S", str(ROOT / "c"), "-B", str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel"], check=True)
    binary = build_dir / "dorado_c"
    if not binary.is_file():
        raise RuntimeError(f"CMake reported success but {binary} is missing")
    return binary


def build_rust_port() -> Path:
    """Build the Rust driver (cargo release); return the binary path."""
    if shutil.which("cargo") is None:
        raise RuntimeError("cargo not found on PATH; install rustup "
                           "(https://rustup.rs) or pass --rust-binary")
    subprocess.run(["cargo", "build", "--release", "--manifest-path",
                    str(ROOT / "rust" / "Cargo.toml"), "--bin", "dorado_rust"],
                   check=True)
    binary = ROOT / "rust" / "target" / "release" / "dorado_rust"
    if not binary.is_file():
        raise RuntimeError(f"cargo reported success but {binary} is missing")
    return binary


def run_port_rep(binary: Path, fixture: Path, particles: int, iterations: int,
                 seed: int, out_dir: Path) -> dict:
    """One port repetition as a secondary process; wall-timed end to end.

    Returns the port's own compute seconds (timing.json; its clock excludes
    fixture load and JSON write), the harness-observed wall time, and the
    appended-step count.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--input", str(fixture), "--particles", str(particles),
           "--iterations", str(iterations), "--seed", str(seed),
           "--out", str(out_dir)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} failed (rc={proc.returncode}):\n"
                           f"{proc.stderr}")
    timing = json.loads((out_dir / "timing.json").read_text())
    return {"seconds": float(timing["seconds"]), "wall_seconds": wall,
            "appended_steps": int(timing["appended_steps"]),
            "timing_meta": timing}


def run_python_rep(params, seed: int, particles: int, iterations: int,
                   seed_x: list[int], seed_y: list[int]) -> dict:
    """One python-reference repetition, timed in-process over the same region.

    The shared params object is safe across reps: reps re-seed np.random and
    rebuild every derived structure inside the timed region (verified
    byte-identical across reps in tests/test_benchmark_harness.py).
    """
    start = time.perf_counter()
    walk_data, seconds = ref.run_reference(params, seed, particles, iterations,
                                           seed_x, seed_y)
    wall = time.perf_counter() - start
    return {"seconds": seconds, "wall_seconds": wall,
            "appended_steps": ref.appended_steps(walk_data),
            "walk_data": walk_data}


def parity_check(reference_bytes: bytes, port_bytes: bytes) -> dict:
    """Byte-exact parity gate; any difference fails, tolerance never loosens."""
    if reference_bytes == port_bytes:
        return {"status": "pass",
                "detail": "byte-identical walk_data.json vs python reference"}
    n = min(len(reference_bytes), len(port_bytes))
    offset = next((i for i in range(n) if reference_bytes[i] != port_bytes[i]), n)
    detail = (f"walk_data.json differs from the python reference at byte "
              f"{offset} (reference {len(reference_bytes)} bytes, "
              f"port {len(port_bytes)} bytes); parity is byte-exact by design, "
              "so this is a port bug, not a tolerance issue")
    return {"status": "fail", "detail": detail}


def consistent_steps(reps: list[dict]) -> int:
    """Return the common appended-step count; fixed-seed reps must agree."""
    counts = {r["appended_steps"] for r in reps}
    if len(counts) != 1:
        raise SystemExit(f"repetitions disagree on appended steps "
                         f"{sorted(counts)}: fixed-seed workloads must be "
                         "deterministic (check DORADO_* env overrides)")
    return counts.pop()


def summarize_reps(reps: list[dict], iterations: int) -> dict:
    """Warm-state medians, per-iteration time, and every sample (pure)."""
    seconds = [r["seconds"] for r in reps]
    walls = [r["wall_seconds"] for r in reps]
    med = median(seconds)
    return {
        "all_seconds": seconds, "median_seconds": med,
        "all_wall_seconds": walls, "median_wall_seconds": median(walls),
        "seconds_per_iteration": med / iterations,
        "appended_steps": consistent_steps(reps),
    }


def build_results(workload: dict, provenance: dict, summaries: dict,
                  parity: dict) -> dict:
    """Assemble the machine-readable results.json payload (pure)."""
    python_median = summaries["python"]["median_seconds"]
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "workload": workload,
        "provenance": provenance,
        "results": {name: {**summary, "parity": parity[name]}
                    for name, summary in summaries.items()},
        "speedup_vs_python": {
            name: python_median / summaries[name]["median_seconds"]
            for name in ("c", "rust")
        },
    }


def fixture_hashes(fixture: Path) -> dict:
    """Hash every byte the implementations consume (identical-inputs evidence)."""
    return {p.name: sha256(p) for p in sorted(fixture.iterdir()) if p.is_file()}


def build_provenance(fixture: Path, workload: dict,
                     port_meta: dict) -> dict:
    """Hardware, versions, revisions, and binary hashes for results.json."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    provenance = {
        "revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "platform": platform.platform(), "machine": platform.machine(),
        "python": platform.python_version(), "numpy": np.__version__,
        "backend_env": os.environ.get("DORADO_BACKEND"),
        "fixture_sha256": fixture_hashes(fixture),
    }
    if "c" in port_meta:
        provenance["c_binary_sha256"] = sha256(Path(workload["c_binary"]))
        provenance["c_compiler"] = port_meta["c"].get("compiler")
    if "rust" in port_meta:
        provenance["rust_binary_sha256"] = sha256(Path(workload["rust_binary"]))
        provenance["rust_toolchain"] = port_meta["rust"].get("toolchain")
    return provenance


def print_summary(results: dict) -> None:
    """Human-readable one-screen table over the results payload."""
    workload = results["workload"]
    print(f"\nworkload: {workload['fixture_dir']}  particles="
          f"{workload['particles']} iterations={workload['iterations']} "
          f"seed={workload['seed']} repeats={workload['repetitions']}")
    for name in ("c", "rust"):
        parity = results["results"][name]["parity"]
        mark = "PASS" if parity["status"] == "pass" else "FAIL"
        print(f"parity {name}: {mark} ({parity['detail']})")
    header = (f"{'implementation':<15}{'median s':>12}{'s/iteration':>14}"
              f"{'median wall s':>15}{'appended steps':>17}")
    print(header)
    for name in ("python", "c", "rust"):
        r = results["results"][name]
        print(f"{name:<15}{r['median_seconds']:>12.6f}"
              f"{r['seconds_per_iteration']:>14.6f}"
              f"{r['median_wall_seconds']:>15.6f}{r['appended_steps']:>17}")
    print("speedup vs python (compute medians): "
          + "  ".join(f"{name} {results['speedup_vs_python'][name]:.2f}x"
                      for name in ("c", "rust")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "bench" / "fixture_rcm",
                        help="normalized fixture directory (params.json + npy files)")
    parser.add_argument("--particles", type=int,
                        help="override the manifest Np_tracer")
    parser.add_argument("--iterations", type=int,
                        help="override the manifest iteration count")
    parser.add_argument("--seed", type=int, help="override the manifest seed")
    parser.add_argument("--seed-x", help="comma-separated x seeding indices")
    parser.add_argument("--seed-y", help="comma-separated y seeding indices")
    parser.add_argument("--repeats", type=int, default=5,
                        help="timed repetitions per implementation (default 5)")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "bench" / "out" / "harness",
                        help="output directory for results.json and rep outputs")
    parser.add_argument("--c-binary", type=Path,
                        help="prebuilt dorado_c path (default: build fresh)")
    parser.add_argument("--rust-binary", type=Path,
                        help="prebuilt dorado_rust path (default: build fresh)")
    args = parser.parse_args(argv)

    check_environment()
    fixture = args.input if args.input.is_absolute() else Path.cwd() / args.input
    if not fixture.is_dir() and (ROOT / args.input).is_dir():
        fixture = ROOT / args.input
    if not fixture.is_dir():
        parser.error(f"fixture directory not found: {args.input}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    params, manifest = ref.load_inputs(fixture)
    particles = args.particles if args.particles is not None else manifest["Np_tracer"]
    iterations = args.iterations if args.iterations is not None else manifest["iterations"]
    seed = args.seed if args.seed is not None else manifest["seed"]
    if not 0 <= seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if particles <= 0 or iterations <= 0:
        parser.error("--particles and --iterations must be positive")
    field = fixture.resolve().name.removeprefix("fixture_")
    try:
        seed_x, seed_y = ref.resolve_seed_box(field, args.seed_x, args.seed_y)
    except ValueError as err:
        parser.error(str(err))

    c_binary = (Path(args.c_binary) if args.c_binary
                else build_c_port(args.out / "build-c"))
    rust_binary = (Path(args.rust_binary) if args.rust_binary
                   else build_rust_port())

    out_dir = args.out if args.out.is_absolute() else Path.cwd() / args.out
    warm_n = min(particles, WARMUP_PARTICLES)
    warm_it = min(iterations, WARMUP_ITERATIONS)
    # Untimed warmup per implementation: same code paths, bounded cost.
    ref.run_reference(params, seed, warm_n, warm_it, seed_x, seed_y)
    run_port_rep(c_binary, fixture, warm_n, warm_it, seed, out_dir / "c" / "warmup")
    run_port_rep(rust_binary, fixture, warm_n, warm_it, seed,
                 out_dir / "rust" / "warmup")

    reps: dict[str, list[dict]] = {"python": [], "c": [], "rust": []}
    walk_bytes: dict[str, bytes] = {}
    for name in ("python", "c", "rust"):
        for k in range(1, args.repeats + 1):
            rep_dir = out_dir / name / f"rep{k}"
            if name == "python":
                rep = run_python_rep(params, seed, particles, iterations,
                                     seed_x, seed_y)
                rep_dir.mkdir(parents=True, exist_ok=True)
                if k == 1:
                    # The reference output defines the golden for this workload.
                    walk_bytes[name] = ref.walk_data_bytes(rep["walk_data"])
                    (rep_dir / "walk_data.json").write_bytes(walk_bytes[name])
                else:
                    rep.pop("walk_data")  # keep memory bounded on large walks
                timing = {"schema_version": 1, "implementation": "python",
                          "seconds": rep["seconds"], "wall_seconds": rep["wall_seconds"],
                          "appended_steps": rep["appended_steps"]}
                (rep_dir / "timing.json").write_bytes(
                    (json.dumps(timing, indent=2, allow_nan=False) + "\n").encode())
            else:
                binary = c_binary if name == "c" else rust_binary
                rep = run_port_rep(binary, fixture, particles, iterations, seed,
                                   rep_dir)
                if k == 1:
                    walk_bytes[name] = (rep_dir / "walk_data.json").read_bytes()
                else:
                    # Timing jsons stay as evidence; large walk files do not.
                    (rep_dir / "walk_data.json").unlink()
            reps[name].append(rep)

    port_meta = {"c": reps["c"][0]["timing_meta"],
                 "rust": reps["rust"][0]["timing_meta"]}
    workload = {
        "fixture_dir": str(fixture), "model": manifest["model"],
        "particles": particles, "iterations": iterations, "seed": seed,
        "seed_x": seed_x, "seed_y": seed_y, "repetitions": args.repeats,
        "warmup": {"particles": warm_n, "iterations": warm_it},
        "c_binary": str(c_binary), "rust_binary": str(rust_binary),
    }
    provenance = build_provenance(fixture, workload, port_meta)

    parity = {
        "python": {"status": "reference",
                   "detail": "python output defines the golden for this workload"},
        "c": parity_check(walk_bytes["python"], walk_bytes["c"]),
        "rust": parity_check(walk_bytes["python"], walk_bytes["rust"]),
    }
    summaries = {name: summarize_reps(rep_list, iterations)
                 for name, rep_list in reps.items()}
    results = build_results(workload, provenance, summaries, parity)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    results_path.write_bytes(
        (json.dumps(results, indent=2, allow_nan=False) + "\n").encode())

    print_summary(results)
    print(f"results.json: {results_path}")

    failed = [name for name in ("c", "rust")
              if results["results"][name]["parity"]["status"] != "pass"]
    if failed:
        print(f"PARITY GATE FAILED for {', '.join(failed)} — see "
              f"results.json; timing above is not valid evidence",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
