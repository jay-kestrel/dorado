"""COBOL walk-core kernels: bit-equality vs the Python reference.

Guards the DORADO_COBOL=1 routing contract:
- a missing shared object raises (no silent fallback, contract C2/C5)
- the pick kernel reproduces random_pick exactly, given the same
  uniform stream
- the travel kernel reproduces calc_travel_times' moved branch
  exactly, including the Python int 0 for unmoved particles
- a full reference-runner walk is byte-identical to the python backend
  on the same fixture bytes and seed (the landed normalizer contract)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dorado import cobol_walk as cw
from dorado.lagrangian_walker import calc_travel_times, particle_stepper, random_pick

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "benchmarks" / "run_reference.py"

HAVE_LIB = cw.library_path().exists()

needs_lib = pytest.mark.skipif(
    not HAVE_LIB, reason="COBOL kernels not built (make -C cobol, gnucobol3)")


def test_missing_library_raises(monkeypatch):
    """DORADO_COBOL=1 with no built .so must raise, never fall back."""
    monkeypatch.setenv("DORADO_COBOL", "1")
    monkeypatch.setattr(cw, "_lib_cache", None)
    monkeypatch.setattr(cw, "_LIB_NAME", "no_such_library.so")
    with pytest.raises(FileNotFoundError, match="make -C cobol"):
        cw.load_library()


def test_cobol_active_env(monkeypatch):
    monkeypatch.delenv("DORADO_COBOL", raising=False)
    assert not cw.cobol_active()
    monkeypatch.setenv("DORADO_COBOL", "1")
    assert cw.cobol_active()
    monkeypatch.setenv("DORADO_COBOL", "0")
    assert not cw.cobol_active()


def test_steepest_descent_rejected_under_cobol(monkeypatch):
    """The COBOL route covers the weighted random walk only."""
    monkeypatch.setenv("DORADO_COBOL", "1")
    particles = SimpleNamespace(stranding_handling="ignore",
                                steepest_descent=True)
    with pytest.raises(NotImplementedError, match="steepest_descent"):
        particle_stepper(particles, [[1, 1]], [0.0], {})


def _synthetic_rows(k, seed):
    """Random D8 weight rows with NaNs and zero-total rows mixed in."""
    rng = np.random.default_rng(seed)
    rows = rng.uniform(0, 1, (k, 9))
    rows[rng.random((k, 9)) < 0.1] = 0.0
    nan_mask = rng.random((k, 9)) < 0.05
    rows[nan_mask] = np.nan
    dead = rng.random(k) < 0.1
    rows[dead] = 0.0  # all-zero rows exercise the fall-through path
    return rows


@needs_lib
def test_pick_matches_random_pick_stream():
    """Kernel picks == random_pick given the identical uniform stream."""
    k, seed = 500, 42
    rows = _synthetic_rows(k, seed)

    # adapter side: raw uniforms from the reference stream
    np.random.seed(seed)
    us = np.random.random(k)
    got = cw.pick_cells(rows, us)

    # reference side: random_pick consumes one uniform(0, total) per
    # row, which is bit-identical to random()*total (probe-validated)
    np.random.seed(seed)
    expected = [random_pick(rows[i]) for i in range(k)]

    assert [int(v) for v in got] == [int(v) for v in expected]


def _fake_particles(seed):
    """Minimal Particles stand-in for calc_travel_times parity."""
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(0.01, 2.0, (20, 20))
    velocity[0, :5] = 0.0  # zero velocities -> inf travel time
    velocity[1, :5] = np.nan  # NaN velocities -> NaN travel time
    angles = np.array([[0.7853981633974483, 1.5707963267948966,
                        0.7853981633974483],
                       [3.141592653589793, 0.0, 3.141592653589793],
                       [5.497787143782138, 4.71238898038469,
                        5.497787143782138]])
    velocity_angle = rng.uniform(-np.pi, np.pi, (20, 20))
    return SimpleNamespace(velocity=velocity, angles=angles,
                           velocity_angle=velocity_angle)


def _travel_case(k, seed, diff_coeff):
    """Per-particle inputs mixing moved/unmoved and multiplier values."""
    rng = np.random.default_rng(seed + 1)
    particles = _fake_particles(seed)
    inds = [(int(x), int(y)) for x, y in rng.integers(1, 19, (k, 2))]
    cells = rng.integers(0, 9, k)
    new_inds = []
    dist = []
    for i in range(k):
        if rng.random() < 0.3:
            # tuples: the real walker feeds calc_travel_times tuples for
            # both ind and new_ind; a list would never compare equal
            new_inds.append((inds[i][0], inds[i][1]))  # unmoved
            dist.append(0.0)
        else:
            new_inds.append((int(inds[i][0]) + int((cells[i] // 3) - 1),
                             int(inds[i][1]) + int((cells[i] % 3) - 1)))
            dist.append(float(1.0 if cells[i] in (1, 3, 5, 7) else 1.4142135623730951))
    vm = rng.uniform(0.5, 2.0, k).tolist()
    dm = rng.uniform(0.5, 2.0, k).tolist()
    return particles, inds, new_inds, [int(c) for c in cells], dist, vm, dm


def _same_bits(a, b):
    """Byte-equality with the int-0/float-0 distinction preserved."""
    if isinstance(a, int) and isinstance(b, int):
        return type(a) is type(b) and a == b
    fa = np.float64(a) if not isinstance(a, int) else np.float64(a)
    fb = np.float64(b) if not isinstance(b, int) else np.float64(b)
    return fa.view(np.uint64) == fb.view(np.uint64)


@needs_lib
@pytest.mark.parametrize("diff_coeff", [0.0, 0.15])
def test_travel_matches_reference_stream(diff_coeff):
    """Kernel travel times == calc_travel_times, byte for byte."""
    k, seed = 300, 7
    particles, inds, new_inds, cells, dist, vm, dm = _travel_case(
        k, seed, diff_coeff)
    particles.diff_coeff = diff_coeff

    # reference side: calc_travel_times draws its own diffusion uniforms
    np.random.seed(seed)
    expected = [calc_travel_times(particles, cells[i], inds[i], new_inds[i],
                                  dist[i], velocity_mult=vm[i],
                                  diff_mult=dm[i]) for i in range(k)]

    # adapter side: explicit uniforms from the same seeded stream, one
    # per moved particle in index order
    np.random.seed(seed)
    moved = np.array([0 if (new_inds[i][0] == inds[i][0]
                            and new_inds[i][1] == inds[i][1]) else 1
                      for i in range(k)], dtype=np.int32)
    moved_idx = [i for i in range(k) if moved[i] == 1]
    diff_u = np.zeros(k)
    diff_u[moved_idx] = np.random.random(len(moved_idx))
    old_vel = np.zeros(k)
    new_vel = np.zeros(k)
    ang_new = np.zeros(k)
    ang_old = np.zeros(k)
    for i in moved_idx:
        x, y = inds[i]
        nx, ny = new_inds[i]
        old_vel[i] = particles.velocity[x, y]
        new_vel[i] = particles.velocity[nx, ny]
        ang_new[i] = particles.angles.flat[cells[i]]
        ang_old[i] = particles.velocity_angle[x, y]
    got = cw.travel_times(np.array(dist), old_vel, new_vel, ang_new,
                          ang_old, diff_u, np.array(dm),
                          np.array(vm), moved, diff_coeff)

    assert len(got) == len(expected)
    for i, (a, b) in enumerate(zip(got, expected)):
        assert _same_bits(a, b), f"particle {i}: {a!r} != {b!r}"


def _run_runner(out_dir, env_extra):
    import os
    env = dict(os.environ)
    env.pop("DORADO_COBOL", None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(RUNNER), "--input",
         str(ROOT / "bench" / "fixture_rcm"), "--out", str(out_dir),
         "--particles", "4", "--iterations", "3", "--seed-x", "16",
         "--seed-y", "140", "--seed", "0"],
        cwd=ROOT, capture_output=True, text=True, env=env)


@needs_lib
def test_walk_bytes_identical_to_python(tmp_path):
    """Full walk via the reference runner: COBOL bytes == python bytes."""
    out_py = tmp_path / "out_py"
    out_cob = tmp_path / "out_cob"
    proc_py = _run_runner(out_py, {})
    proc_cob = _run_runner(out_cob, {"DORADO_COBOL": "1"})
    assert proc_py.returncode == 0, proc_py.stderr
    assert proc_cob.returncode == 0, proc_cob.stderr
    walk_py = (out_py / "walk_data.json").read_bytes()
    walk_cob = (out_cob / "walk_data.json").read_bytes()
    assert walk_cob == walk_py
    # and the walk actually moved particles (the case is not trivial)
    walk = json.loads(walk_py)
    assert walk["xinds"][0][0] == 16 and len(walk["xinds"][0]) == 4


@needs_lib
def test_runner_cobol_provenance(tmp_path):
    """Timing JSON identifies the COBOL implementation and GnuCOBOL."""
    out = tmp_path / "out"
    proc = _run_runner(out, {"DORADO_COBOL": "1"})
    assert proc.returncode == 0, proc.stderr
    timing = json.loads((out / "timing.json").read_text())
    assert timing["implementation"] == "cobol"
    assert timing["gnucobol_version"]
