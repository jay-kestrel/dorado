"""Generate Julia golden fixtures from the Python reference (dorado master).

Everything the Julia equivalence suite asserts comes from here:

  goldens/init_parity/fixture_{rcm,anuga}/   inputs + derived state of
                                             ``Particles.__init__`` (depth/
                                             stage/qx/qy cleanup, the
                                             non-DeltaRCM axis swap, velocity
                                             floors, velocity angles, cell
                                             types, routing weights)
  goldens/pick_goldens.json                  inverse-CDF picker parity rows
                                             (fixed uniform inputs -> cell)
  goldens/deterministic/                     a draw-free (steepest-descent,
                                             diffusion-free, exact seeding)
                                             workload; the walk data must be
                                             reproduced exactly
  goldens/stochastic/                        per-seed summary statistics of
                                             the random-pick workloads
                                             (rcm/anuga x seeds 0..4); Julia
                                             must match their cross-seed
                                             spread, not any single stream
  goldens/stochastic/{name}_seed{seed}_walk_data.json
                                             one full walk for the schema test

Run from the repo root (uses this checkout, not an installed distribution):

    python3 julia/goldens/generate_goldens.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest.mock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dorado.lagrangian_walker import random_pick, random_pick_seed, steep_descent  # noqa: E402
from dorado.particle_track import Particles, modelParams  # noqa: E402

GOLD = Path(__file__).resolve().parent / "goldens"

SEED_BOXES = {
    "rcm": (list(range(15, 17)), list(range(137, 140))),
    "anuga": (list(range(20, 30)), list(range(48, 53))),
}
STOCH_SEEDS = [0, 1, 2, 3, 4]
STOCH_PARTICLES = 500
STOCH_ITERS = {"rcm": 20, "anuga": 50}


def load_inputs(fixture_dir: Path) -> tuple[modelParams, dict]:
    """Mirror benchmarks.run_reference.load_inputs exactly."""
    manifest = json.loads((fixture_dir / "params.json").read_text())
    params = modelParams()
    for key in ("depth", "stage", "qx", "qy", "cell_type"):
        setattr(params, key, np.load(fixture_dir / manifest["arrays"][key]))
    for key in ("dx", "theta", "gamma", "diff_coeff", "dry_depth",
                "steepest_descent", "model"):
        setattr(params, key, manifest[key])
    return params, manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, np.ascontiguousarray(array))


def dump_walk(path: Path, walk_data: dict) -> None:
    """Walk data as plain JSON (ints stay ints, floats stay floats)."""
    def jsonable(value):
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

    path.write_text(json.dumps(jsonable(walk_data), allow_nan=False))


class UniformQueue:
    """Patch of numpy.random.uniform/random that replays a fixed queue.

    Used to drive the pickers with known uniforms so Julia can replay the
    identical inputs and assert the identical outputs (picker parity),
    independent of any RNG stream.
    """

    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self, *args, **kwargs):
        value = self.values[self.i]
        self.i += 1
        return value


def gen_init_parity() -> None:
    """Inputs + derived init state for both bench workloads."""
    for name in ("rcm", "anuga"):
        fixture = ROOT / "bench" / f"fixture_{name}"
        params, _manifest = load_inputs(fixture)
        particle = Particles(params)

        out = GOLD / "init_parity" / f"fixture_{name}"
        out.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((fixture / "params.json").read_text())
        for key, fname in manifest["arrays"].items():
            shutil.copyfile(fixture / fname, out / fname)
        shutil.copyfile(fixture / "params.json", out / "params.json")

        np.save(out / "u.f64.npy", particle.u)
        np.save(out / "v.f64.npy", particle.v)
        np.save(out / "velocity.f64.npy", particle.velocity)
        np.save(out / "velocity_angle.f64.npy", particle.velocity_angle)
        np.save(out / "cell_type.i64.npy", particle.cell_type)
        np.save(out / "weight.f64.npy", particle.weight)

        hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
        (out / "hashes.json").write_text(json.dumps(hashes, indent=1) + "\n")
        print(f"init_parity/fixture_{name}: {sorted(hashes)}")


def gen_pick_goldens() -> None:
    """Picker parity rows: fixed uniforms in, cell indices out."""
    fixture = ROOT / "bench" / "fixture_rcm"
    params, _ = load_inputs(fixture)
    particle = Particles(params)
    weight = particle.weight  # (L, W, 9)

    L, W = weight.shape[:2]
    # Weight rows: the seed region, a sample of wet interior cells, plus
    # synthetic rows with NaNs, ties, single maxima, and all-zero rows.
    cells = [(15, 137), (15, 138), (16, 139), (60, 120), (30, 40), (90, 200)]
    rows = [list(weight[x, y, :]) for x, y in cells]
    rows += [
        [np.nan, 0.1, 0.2, 0.0, 0.3, np.nan, 0.0, 0.4, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # tie: slots 0 and 8
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # unique max = center
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # dead end
        [np.nan] * 9,
        [0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2],  # three-way tie
    ]

    fracs = [0.0, 0.25, 0.5, 0.75, 0.999]
    pick_rows = []
    for row in rows:
        clean = np.nan_to_num(np.asarray(row, dtype=float), nan=0.0)
        total = float(np.cumsum(clean)[-1])
        for frac in fracs:
            u = frac * total
            with unittest.mock.patch.object(np.random, "uniform", UniformQueue([u])):
                got = random_pick(np.array(row, dtype=float))
            pick_rows.append({
                "kind": "random_pick", "probs": [float(v) for v in row],
                "u": u, "expected": int(got),
            })
            with unittest.mock.patch.object(np.random, "uniform", UniformQueue([u])):
                got = steep_descent(np.array(row, dtype=float))
            pick_rows.append({
                "kind": "steep_descent", "probs": [float(v) for v in row],
                "u": u, "expected": int(got),
            })

    choices = list(range(15, 17)) + list(range(137, 140))
    for total_choice in (len(choices),):
        for frac in fracs:
            u = frac * total_choice
            with unittest.mock.patch.object(np.random, "uniform", UniformQueue([u])):
                got = random_pick_seed(choices)
            pick_rows.append({
                "kind": "random_pick_seed", "choices": choices,
                "u": u, "expected": int(got),
            })

    GOLD.mkdir(parents=True, exist_ok=True)
    (GOLD / "pick_goldens.json").write_text(json.dumps(pick_rows, indent=1) + "\n")
    print(f"pick_goldens: {len(pick_rows)} rows")


def gen_deterministic() -> None:
    """Replay-gated workload: bit-exact Julia reproduction required.

    A 12x16 synthetic domain: smooth stage whose steepest descent OCCASIONALLY
    TIES (the bilinear stage produces equal-gradient neighbors), a dry blob
    that exercises park stranding, estimated cell types (no explicit
    cell_type array), and exact seeding that includes the (0, 0) sentinel.

    Every np.random.uniform call the walk makes (all of them steep_descent
    tie-breaks: diff_coeff is 0 and seeding is exact) is recorded with
    numpy's exact sampling formula low + (high - low) * random(), and the
    recorded sequence is replayed by the Julia test — a bit-exact gate for
    the deterministic path that also exercises the tie-break branches.
    """
    L, W = 12, 16
    xs = np.arange(L, dtype=float)[:, None]
    ys = np.arange(W, dtype=float)[None, :]
    depth = 0.5 + 0.5 * np.sin(0.9 * xs + 0.3 * ys) * np.cos(0.7 * ys + 0.2 * xs)
    depth = np.broadcast_to(depth, (L, W)).copy()
    depth[4:6, 6:9] = 0.0  # dry blob -> park stranding + land estimation
    stage = 10.0 + 0.1 * xs + 0.037 * ys + 0.011 * xs * ys
    stage = np.broadcast_to(stage, (L, W)).copy()

    params = modelParams()
    params.dx = 2.0
    params.depth = depth
    params.stage = stage
    params.qx = np.zeros((L, W))
    params.qy = np.zeros((L, W))
    params.theta = 1.0
    params.gamma = 0.05
    params.diff_coeff = 0.0
    params.dry_depth = 0.1
    params.steepest_descent = True
    params.model = "DeltaRCM"

    drawn_uniforms = []

    def recording_uniform(*args, **kwargs):
        # Replicate numpy.random.uniform's sampling formula exactly. The RAW
        # random() draw is recorded (not the scaled value): the Julia replay
        # re-derives u = raw * high with the same multiplication, so the
        # rounding behavior (u may round up to `high`, making searchsorted
        # return index 9 — a reference defect we replicate, not fix) is
        # bit-identical.
        if len(args) >= 2:
            low, high = float(args[0]), float(args[1])
        elif len(args) == 1:
            low, high = float(args[0]), float(kwargs.get("high", 1.0))
        else:
            low = float(kwargs.get("low", 0.0))
            high = float(kwargs.get("high", 1.0))
        raw = float(np.random.random())
        drawn_uniforms.append(raw)
        return low + (high - low) * raw

    seed_xloc = [0, 2, 4, 6, 8, 10]
    # Equal lengths required: the Python exact-seeding method extends BOTH
    # axes with the X axis's divmod, so unequal box lengths raise IndexError
    # (a known defect we replicate, not fix — see julia/README.md).
    seed_yloc = [0, 3, 7, 11, 14, 15]

    np.random.seed(0)
    particle = Particles(params)
    with unittest.mock.patch.object(np.random, "uniform", recording_uniform):
        particle.generate_particles(20, seed_xloc, seed_yloc, method="exact")
        for _ in range(30):
            particle.run_iteration()

    out = GOLD / "deterministic"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dx": 2.0, "theta": 1.0, "gamma": 0.05, "diff_coeff": 0.0,
        "dry_depth": 0.1, "steepest_descent": True, "model": "DeltaRCM",
        "seed": 0, "Np_tracer": 20, "iterations": 30,
        "seed_xloc": seed_xloc, "seed_yloc": seed_yloc, "method": "exact",
        "uniforms": drawn_uniforms,
        "arrays": {
            "depth": "depth.f64.npy", "stage": "stage.f64.npy",
            "qx": "qx.f64.npy", "qy": "qy.f64.npy",
        },
    }
    (out / "params.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write_npy(out / "depth.f64.npy", depth)
    write_npy(out / "stage.f64.npy", stage)
    write_npy(out / "qx.f64.npy", params.qx)
    write_npy(out / "qy.f64.npy", params.qy)
    dump_walk(out / "walk_data.json", particle.walk_data)

    hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "hashes.json").write_text(json.dumps(hashes, indent=1) + "\n")
    print(f"deterministic: {len(json.loads((out / 'walk_data.json').read_text())['xinds'])} particles")


def visit_stats(walk_data: dict, shape: tuple[int, int]) -> dict:
    """Summary statistics of a walk (mirrored exactly by the Julia tests)."""
    L, W = shape
    hist = np.zeros((L, W))
    dx_counts = {k: 0 for k in (-1, 0, 1)}
    dy_counts = {k: 0 for k in (-1, 0, 1)}
    n_parked = 0
    n_entries = 0
    for px, py, times, stranded in zip(
            walk_data["xinds"], walk_data["yinds"],
            walk_data["travel_times"], walk_data["stranded"]):
        for t, (x, y) in enumerate(zip(px, py)):
            hist[x, y] += 1.0
            n_entries += 1
            n_parked += int(stranded[t] == 1)
            if t >= 1:
                ddx = x - px[t - 1]
                ddy = y - py[t - 1]
                if ddx in dx_counts:
                    dx_counts[ddx] += 1
                if ddy in dy_counts:
                    dy_counts[ddy] += 1
    final_times = sorted(float(path[-1]) for path in walk_data["travel_times"])
    quantiles = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        n = len(final_times)
        h = (n - 1) * q
        lo = int(np.floor(h))
        hi = int(np.ceil(h))
        quantiles[str(q)] = (final_times[lo] + (h - lo) * (final_times[hi] - final_times[lo])
                             if hi > lo else final_times[lo])
    return {
        "visit_hist": hist.tolist(),
        "n_entries": n_entries,
        "parked_fraction": n_parked / n_entries,
        "quantiles": quantiles,
        "dx_counts": {str(k): v for k, v in dx_counts.items()},
        "dy_counts": {str(k): v for k, v in dy_counts.items()},
        "appended_steps": sum(len(path) - 1 for path in walk_data["xinds"]),
    }


def gen_stochastic() -> None:
    """Per-seed summary statistics for the random-pick workloads."""
    (GOLD / "stochastic").mkdir(parents=True, exist_ok=True)
    for name in ("rcm", "anuga"):
        fixture = ROOT / "bench" / f"fixture_{name}"
        params, manifest = load_inputs(fixture)
        shape = params.depth.shape
        seed_x, seed_y = SEED_BOXES[name]
        iterations = STOCH_ITERS[name]
        for seed in STOCH_SEEDS:
            particle = Particles(params)
            np.random.seed(seed)
            particle.generate_particles(STOCH_PARTICLES, seed_x, seed_y)
            for _ in range(iterations):
                particle.run_iteration()
            walk = particle.walk_data
            stats = visit_stats(walk, shape)
            stats.update({"fixture": name, "seed": seed,
                          "particles": STOCH_PARTICLES, "iterations": iterations})
            out = GOLD / "stochastic" / f"{name}_seed{seed}_stats.json"
            out.write_text(json.dumps(stats, indent=1) + "\n")
            if seed == 0:
                dump_walk(GOLD / "stochastic" / f"{name}_seed0_walk_data.json", walk)
            print(f"stochastic {name} seed {seed}: "
                  f"appended={stats['appended_steps']} "
                  f"parked={stats['parked_fraction']:.4f}")


if __name__ == "__main__":
    gen_init_parity()
    gen_pick_goldens()
    gen_deterministic()
    gen_stochastic()
