"""Reference runner: seeded golden gates and fixture-byte equivalence."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dorado.particle_track import Particles, modelParams

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_reference.py"

# Goldens from tests/test_examplecases.py (TestRCM.test_few_steps_RCM, seed 0).
GOLDEN_TRAVEL_TIMES = [0.0, 7007593448.337235, 10439964733.462337, 13698473384.876724]


def run_runner(tmp_path: Path, fixture: str, *extra: str) -> Path:
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--input", str(ROOT / "bench" / f"fixture_{fixture}"),
         "--out", str(out), *extra],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return out


def test_rcm_golden_trajectory(tmp_path):
    """One seeded particle at (16, 140) must reproduce the example goldens."""
    out = run_runner(tmp_path, "rcm", "--particles", "1", "--iterations", "3",
                     "--seed-x", "16", "--seed-y", "140", "--seed", "0")
    walk = json.loads((out / "walk_data.json").read_text())
    assert walk["xinds"][0] == [16, 17, 18, 19]
    assert walk["yinds"][0] == [140, 141, 141, 141]
    assert walk["travel_times"][0] == pytest.approx(GOLDEN_TRAVEL_TIMES, rel=1e-6)
    timing = json.loads((out / "timing.json").read_text())
    assert timing["seconds"] > 0
    assert timing["appended_steps"] == 3
    assert timing["implementation"] == "python"
    assert timing["particles"] == 1
    assert timing["iterations"] == 3
    assert timing["seed"] == 0
    assert "npy/params load" in timing["timed_region"]


def test_anuga_matches_npz_reference(tmp_path):
    """Runner output on fixture bytes must equal an in-process run on the npz."""
    out = run_runner(tmp_path, "anuga", "--particles", "1", "--iterations", "3",
                     "--seed-x", "5", "--seed-y", "10", "--seed", "0")
    walk = json.loads((out / "walk_data.json").read_text())

    # Mirror dorado/example_data/define_params.py on the original npz bytes.
    with np.load(ROOT / "dorado" / "example_data" / "ex_anuga_data.npz") as data:
        params = modelParams()
        params.depth = data["depth"]
        params.stage = np.copy(data["depth"])
        params.qx = data["qx"]
        params.qy = data["qy"]
        params.dx = 10.0
        params.theta = 1.0
        params.gamma = 0.05
        params.diff_coeff = 0.2
        params.dry_depth = 0.1
        params.steepest_descent = False
        params.model = "Anuga"
    np.random.seed(0)
    particle = Particles(params)
    particle.generate_particles(1, [5], [10])
    expected = None
    for _ in range(3):
        expected = particle.run_iteration()

    assert walk["xinds"][0] == expected["xinds"][0]
    assert walk["yinds"][0] == expected["yinds"][0]
    assert walk["travel_times"][0] == expected["travel_times"][0]


def test_default_workload_comes_from_manifest(tmp_path):
    """Without overrides, the runner must report the manifest workload."""
    out = run_runner(tmp_path, "rcm", "--iterations", "1")
    timing = json.loads((out / "timing.json").read_text())
    manifest = json.loads((ROOT / "bench" / "fixture_rcm" / "params.json").read_text())
    assert timing["particles"] == manifest["Np_tracer"]
    assert timing["iterations"] == 1
    assert timing["seed"] == manifest["seed"]
    walk = json.loads((out / "walk_data.json").read_text())
    assert len(walk["xinds"]) == manifest["Np_tracer"]
