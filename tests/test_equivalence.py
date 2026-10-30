"""Fixed-ensemble gates for the default ChaCha8 Rust API vs NumPy Python.

Protocol (not a hypothesis-test p-value): seeds 0..19, explicitly listed below,
1,000 particles per seed/backend = 20,000 terminal observations per dataset.
RCM (120x240) and ANUGA (100x100) use the full bundled grids, their example
parameters and seed regions, and 20 iterations. The 15x15 synthetic grid runs
12 iterations: equal cohorts start wet, dry, on permanent land, and beside a
boundary. After four steps its dry cell rewets; probability release (p=0.1)
exercises stochastic stranding, while land stays parked and the boundary freezes.
The rebuilt wet phase uses seed+20 (20..39) on BOTH engines, so rebuilding does
not replay the native stream used in the dry phase while Python continues it.

Both engines receive identical parameters, exact initial positions, and seed
lists. NumPy's global RNG is seeded for the Python reference; the default API
receives seed= for its independent per-particle streams. Environment overrides
are removed and global RNG state is restored by a module-local fixture. No
MT19937 parity adapter, COBOL route, or reference trajectory pin is modified.

Each metric pools across seeds, NOT across repeated observations of a particle:
* final position: exact joint grid-cell histogram (no spatial coarsening), with
  two-sample KS on row-major cell IDs; row and column marginals are also tested;
* travel time: KS on raw terminal seconds, TVD on fixed-width bins (RCM 5e9 s,
  ANUGA 50 s, synthetic 1 s), including a separate exact-zero bin and all tails;
* walk length: number of recorded transitions (parked steps included), discrete;
* stranding: final flag, ever-stranded flag, and cumulative stranded time;
* displacement: each particle's first recorded displacement, encoded D8, with
  category 9 for no recorded transition. Using one observation per particle
  avoids overweighting long walks; this is not an all-steps independence claim.

KS = sup_x |F_python(x)-F_rust(x)|. TVD = 0.5 * sum_b |p_b-q_b| over the
union of occupied bins; both must be <=0.05 for EVERY metric/dataset, without
seed selection, retries, p-value gates, or tolerance adaptation. Continuous
empirical samples generally have disjoint support, hence time TVD is explicitly
histogram-resolution equivalence, not unbinned measure equivalence. Fixed seeds
make this a repeatable finite-ensemble regression, not proof for all flow fields.
Known schema divergence (master 7477df0): with RCM, seed=0, 100 particles and
20 steps, all-zero stranded_time infers float64 on Rust but int64 on Python.
Python's ragged lists retain integer zeros until stranding adds a float; native
outputs uniform float64. The clarified contract preserves numeric field types
and ragged shape semantics, not this data-dependent dtype accident. A named
regression pins the exact divergence, and all other inferred dtypes must match.
Strict parity would require a separate source/normalization change; no backend
source or TestPythonReference constants are changed here.

Run `python -m pytest -q -s tests/test_equivalence.py` to print achieved metrics.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.stats import ks_2samp

from benchmarks.normalize_fixtures import build_fixture
from dorado.particle_track import Particles


SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
PARTICLES_PER_SEED = 1000
KS_LIMIT = 0.05
TVD_LIMIT = 0.05
EXAMPLE_DATA = Path(__file__).resolve().parents[1] / "dorado" / "example_data"
WALK_KEYS = {"xinds", "yinds", "travel_times", "stranded", "stranded_time", "depth"}


@dataclass(frozen=True)
class Workload:
    name: str
    steps: int
    starts: tuple[tuple[int, int], ...]
    time_bin_width: float


WORKLOADS = (
    Workload("rcm", 20, tuple((x, y) for x in (15, 16) for y in (137, 138, 139)), 5e9),
    Workload(
        "anuga", 20, tuple((x, y) for x in range(20, 30) for y in range(48, 53)), 50.0
    ),
    Workload("synthetic", 12, ((4, 4), (6, 6), (6, 10), (1, 7)), 1.0),
)


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    """Do not let a developer's opt-in engine silently replace either oracle."""
    monkeypatch.delenv("DORADO_BACKEND", raising=False)
    monkeypatch.delenv("DORADO_COBOL", raising=False)
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def make_params(workload: Workload, *, rewetted: bool = False) -> SimpleNamespace:
    if workload.name != "synthetic":
        arrays, manifest = build_fixture(workload.name, EXAMPLE_DATA)
        names = ("dx", "theta", "gamma", "diff_coeff", "dry_depth", "model")
        params = SimpleNamespace(**arrays, **{key: manifest[key] for key in names})
    else:
        depth = np.ones((15, 15))
        depth[6, 6] = 1.0 if rewetted else 0.05
        cell_type = np.ones((15, 15), dtype=int)
        cell_type[[0, -1], :] = -1
        cell_type[:, [0, -1]] = -1
        cell_type[6, 10] = 2
        params = SimpleNamespace(
            depth=depth,
            stage=np.ones_like(depth),
            qx=np.ones_like(depth),
            qy=np.zeros_like(depth),
            cell_type=cell_type,
            dx=1.0,
            theta=1.0,
            model="DeltaRCM",
            stranding_release="probability",
            entrain_prob=0.1,
        )
    params.stranding_handling = "park"
    params.particle_variables = ["depth"]
    return params


def new_particles(params, backend: str, seed: int, threads: int) -> Particles:
    # Omit backend on the native side: a changed default must fail this suite.
    kwargs = {} if backend == "rust" else {"backend": "python"}
    particles = Particles(params, seed=seed, n_threads=threads, **kwargs)
    assert particles.backend == backend
    return particles


def run_walk(
    workload: Workload,
    backend: str,
    seed: int,
    count: int = PARTICLES_PER_SEED,
    threads: int = 1,
) -> dict:
    np.random.seed(seed)
    particles = new_particles(make_params(workload), backend, seed, threads)
    x, y = zip(*workload.starts)
    particles.generate_particles(count, list(x), list(y), method="exact")
    for step in range(workload.steps):
        if workload.name == "synthetic" and step == 4:
            previous = particles.walk_data
            wet_seed = seed + len(SEEDS)
            np.random.seed(wet_seed)
            particles = new_particles(
                make_params(workload, rewetted=True),
                backend,
                wet_seed,
                threads,
            )
            particles.generate_particles(0, [], [], previous_walk_data=previous)
        walk = particles.run_iteration()
    return walk


def assert_schema(walk: dict, workload: Workload, count: int) -> dict:
    """Validate ragged particle/step alignment; compare inferred numeric dtypes."""
    assert isinstance(walk, dict) and set(walk) == WALK_KEYS
    lengths = [len(row) for row in walk["xinds"]]
    assert all(1 <= length <= workload.steps + 1 for length in lengths)
    dtypes = {}
    for key, rows in walk.items():
        assert isinstance(rows, list) and len(rows) == count
        assert all(isinstance(row, list) for row in rows)
        assert [len(row) for row in rows] == lengths
        values = np.asarray([value for row in rows for value in row])
        assert np.all(np.isfinite(values)), key
        expected_kind = "i" if key in {"xinds", "yinds", "stranded"} else "f"
        # Python retains integer zeros until a particle actually strands.
        # The observed int/float divergence is pinned explicitly below.
        allowed_kinds = "if" if key == "stranded_time" else expected_kind
        assert values.dtype.kind in allowed_kinds, (key, values.dtype)
        dtypes[key] = values.dtype
    shape = (
        (15, 15)
        if workload.name == "synthetic"
        else ((120, 240) if workload.name == "rcm" else (100, 100))
    )
    for key, bound in zip(("xinds", "yinds"), shape):
        assert all(0 <= value < bound for row in walk[key] for value in row)
    for key in ("travel_times", "stranded_time"):
        assert all(np.all(np.diff(row) >= 0) and row[0] == 0 for row in walk[key])
    assert all(value in (0, 1) for row in walk["stranded"] for value in row)
    return dtypes


def observations(walk: dict, width: int) -> dict[str, np.ndarray]:
    x = np.asarray([row[-1] for row in walk["xinds"]])
    y = np.asarray([row[-1] for row in walk["yinds"]])
    displacement = [
        (xs[1] - xs[0] + 1) * 3 + ys[1] - ys[0] + 1 if len(xs) > 1 else 9
        for xs, ys in zip(walk["xinds"], walk["yinds"])
    ]
    return {
        "final_position": x * width + y,
        "final_row": x,
        "final_column": y,
        "travel_time": np.asarray([row[-1] for row in walk["travel_times"]]),
        "walk_length": np.asarray([len(row) - 1 for row in walk["xinds"]]),
        "stranded_final": np.asarray([row[-1] for row in walk["stranded"]]),
        "stranded_ever": np.asarray([max(row) for row in walk["stranded"]]),
        "stranded_time": np.asarray([row[-1] for row in walk["stranded_time"]]),
        "first_displacement": np.asarray(displacement),
    }


def time_bins(values: np.ndarray, width: float) -> np.ndarray:
    """Zero is its own atom; positive t maps to 1+floor(t/width), with no cap."""
    return np.where(values == 0, 0, 1 + np.floor(values / width)).astype(np.int64)


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = Counter(left.tolist()), Counter(right.tolist())
    return 0.5 * sum(
        abs(a[key] / len(left) - b[key] / len(right)) for key in a.keys() | b.keys()
    )


def distances(
    left: np.ndarray, right: np.ndarray, bin_width: float | None = None
) -> tuple[float, float]:
    ks = float(ks_2samp(left, right, method="asymp").statistic)
    if bin_width is not None:
        left, right = time_bins(left, bin_width), time_bins(right, bin_width)
    return ks, total_variation(left, right)


def assert_synthetic_cohorts(walk: dict) -> None:
    # Fixed cohorts make these edge checks independent of any distribution gate.
    for particle in range(0, len(walk["xinds"]), 4):
        dry, land, boundary = particle + 1, particle + 2, particle + 3
        assert walk["stranded"][dry][1:5] == [1] * 4
        assert walk["stranded"][land] == [0] + [1] * 12
        assert walk["xinds"][land] == [6] * 13
        assert walk["yinds"][land] == [10] * 13
        assert walk["xinds"][boundary] == [1]
        assert walk["yinds"][boundary] == [7]
        assert walk["travel_times"][boundary] == [0]
    still_parked = [row[-1] for row in walk["stranded"][1::4]]
    assert 0 < sum(still_parked) < len(still_parked)


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
def test_cross_backend_ensemble(workload, record_property):
    samples = {"rust": {}, "python": {}}
    width = make_params(workload).depth.shape[1]
    for seed in SEEDS:
        for backend in samples:
            walk = run_walk(workload, backend, seed)
            assert_schema(walk, workload, PARTICLES_PER_SEED)
            if workload.name == "synthetic":
                assert_synthetic_cohorts(walk)
            for metric, values in observations(walk, width).items():
                samples[backend].setdefault(metric, []).append(values)
    results = {}
    for metric in samples["rust"]:
        native = np.concatenate(samples["rust"][metric])
        reference = np.concatenate(samples["python"][metric])
        assert len(native) == len(reference) == len(SEEDS) * PARTICLES_PER_SEED
        ks, tvd = distances(
            reference,
            native,
            workload.time_bin_width if metric == "travel_time" else None,
        )
        results[metric] = {"ks": ks, "tvd": tvd}
    report = json.dumps({"dataset": workload.name, "metrics": results}, sort_keys=True)
    print("EQUIVALENCE " + report)
    record_property("equivalence", report)
    for metric, result in results.items():
        assert result["ks"] <= KS_LIMIT, (workload.name, metric, result)
        assert result["tvd"] <= TVD_LIMIT, (workload.name, metric, result)


@pytest.mark.parametrize("workload", WORKLOADS, ids=lambda w: w.name)
def test_walk_data_schema_matches(workload):
    schemas = [
        assert_schema(run_walk(workload, backend, SEEDS[0], count=100), workload, 100)
        for backend in ("rust", "python")
    ]
    for schema in schemas:
        assert schema.pop("stranded_time").kind in "if"
    assert schemas[0] == schemas[1]


def test_stranded_time_dtype_divergence_documented():
    inferred = {}
    for backend in ("rust", "python"):
        walk = run_walk(WORKLOADS[0], backend, SEEDS[0], count=100)
        values = np.asarray([value for row in walk["stranded_time"] for value in row])
        assert np.all(values == 0)
        inferred[backend] = values.dtype
    assert inferred == {"rust": np.dtype("float64"), "python": np.dtype("int64")}


def test_native_seed_repeatability_and_thread_independence():
    workload = WORKLOADS[2]
    one = run_walk(workload, "rust", SEEDS[0], count=100, threads=1)
    repeat = run_walk(workload, "rust", SEEDS[0], count=100, threads=1)
    eight = run_walk(workload, "rust", SEEDS[0], count=100, threads=8)
    assert one == repeat == eight
    assert_schema(one, workload, 100)
    assert_synthetic_cohorts(one)
    assert one != run_walk(workload, "rust", SEEDS[1], count=100)


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([0, 1], [0, 1], (0.0, 0.0)),
        ([0, 0], [1, 1], (1.0, 1.0)),
        ([0, 0, 1, 1], [0, 1, 1, 1], (0.25, 0.25)),
        ([0, 0, 1, 1], [0, 1], (0.0, 0.0)),
    ],
)
def test_distribution_distance_math(left, right, expected):
    assert distances(np.asarray(left), np.asarray(right)) == pytest.approx(expected)


def test_time_histogram_keeps_zero_and_tails():
    values = np.asarray([0, 0.1, 9.9, 10, 10000])
    np.testing.assert_array_equal(time_bins(values, 10), [0, 1, 1, 2, 1001])
    # Histogram matching must not replace raw-time KS.
    assert distances(np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0]), 10) == (1.0, 0.0)


def test_observations_preserve_joint_positions_and_no_move_category():
    walk = {
        "xinds": [[1, 2], [3]],
        "yinds": [[4, 3], [5]],
        "travel_times": [[0, 2.0], [0]],
        "stranded": [[0, 1], [0]],
        "stranded_time": [[0, 1.0], [0]],
    }
    result = observations(walk, 10)
    expected = {
        "final_position": [23, 35],
        "final_row": [2, 3],
        "final_column": [3, 5],
        "travel_time": [2.0, 0],
        "walk_length": [1, 0],
        "stranded_final": [1, 0],
        "stranded_ever": [1, 0],
        "stranded_time": [1.0, 0],
        "first_displacement": [6, 9],
    }
    assert result.keys() == expected.keys()
    for key, values in expected.items():
        np.testing.assert_array_equal(result[key], values)
