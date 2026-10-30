"""Keep benchmark timing boundaries and the non-shipping batch baseline honest."""

import copy
import json
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks import bench_routing as bench
from benchmarks.numpy_stepper import boundary_mask, choose_cells, make_stepper
from dorado import lagrangian_walker as lw


@pytest.mark.parametrize("name,shape,dx,steps", [
    ("rcm", (120, 240), 50, 20), ("anuga", (100, 100), 10, 50),
])
def test_workload(name, shape, dx, steps):
    workload = bench.load_workload(name)
    assert workload.params.depth.shape == shape
    assert workload.params.dx == dx
    assert workload.steps == steps
    assert len(workload.data_sha256) == 64


@pytest.mark.parametrize("name", ["rcm", "anuga"])
def test_numpy_displacements_and_travel_times(name):
    staged = bench.stage_particles(bench.load_workload(name), 500, "python", 23)
    reference, batch = copy.deepcopy(staged), copy.deepcopy(staged)
    np.random.seed(34)
    bench.route(reference, 20)
    np.random.seed(34)
    with patch.object(lw, "particle_stepper", make_stepper(batch)):
        bench.route(batch, 20)
    # Small-case displacement distribution comparison; same legacy draw ordering
    # permits the stronger trajectory check too, but is not the native RNG gate.
    for axis in ("xinds", "yinds"):
        ref_deltas = np.concatenate([np.diff(path) for path in reference.walk_data[axis]])
        batch_deltas = np.concatenate([np.diff(path) for path in batch.walk_data[axis]])
        np.testing.assert_array_equal(
            np.histogram(ref_deltas, bins=[-1.5, -0.5, 0.5, 1.5])[0],
            np.histogram(batch_deltas, bins=[-1.5, -0.5, 0.5, 1.5])[0])
        assert reference.walk_data[axis] == batch.walk_data[axis]
    for actual, expected in zip(batch.walk_data["travel_times"], reference.walk_data["travel_times"]):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_boundary_mask_matches_reference_slices():
    grid = np.zeros((7, 8))
    grid[0, :] = grid[-1, :] = -1
    grid[:, 0] = grid[:, -1] = -1
    grid[3, 4] = -1
    expected = np.array([[(-1 in grid[i-1:i+2, j-1:j+2])
                          for j in range(8)] for i in range(7)])
    np.testing.assert_array_equal(boundary_mask(grid), expected)


def test_weight_selection_zero_nan_and_left_cutoff():
    weights = np.zeros((4, 9))
    weights[0] = np.nan
    weights[1, 2] = weights[1, 6] = 1
    weights[2, 2] = weights[2, 6] = 1
    weights[3, 3] = 1
    np.testing.assert_array_equal(choose_cells(weights, np.array([0.5, 0.5, 0.75, 0])),
                                  [0, 2, 6, 0])


@pytest.mark.parametrize("diffusion", [0.0, 0.2])
def test_numpy_sentinel_boundary_and_optional_sampling(diffusion):
    staged = bench.stage_particles(bench.load_workload("anuga"), 3, "python", 0)
    staged.diff_coeff = diffusion
    staged.cell_type[:] = 0
    staged.cell_type[9, 9] = -1
    staged.roi_grid = np.ones_like(staged.depth)
    staged.available_particle_vars["depth"] = staged.depth
    old = [[0, 0], [10, 10], [25, 50]]
    np.random.seed(8)
    expected = lw.particle_stepper(copy.deepcopy(staged), old, [0, 0, 0], {"depth": [0, 0, 0]})
    np.random.seed(8)
    actual = make_stepper(staged)(staged, old, [0, 0, 0], {"depth": [0, 0, 0]})
    assert actual[0] == expected[0]
    assert actual[0][:2] == old[:2]
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12)
    assert actual[2] == expected[2]


def test_numpy_rejects_steepest_descent():
    staged = bench.stage_particles(bench.load_workload("rcm"), 1, "python", 0)
    staged.steepest_descent = True
    with pytest.raises(ValueError, match="steepest_descent"):
        make_stepper(staged)


def test_measure_resets_and_excludes_preparation(monkeypatch):
    staged = bench.stage_particles(bench.load_workload("anuga"), 10, "python", 0)
    original = copy.deepcopy(staged.walk_data)
    real_copy = copy.deepcopy
    in_timing = False
    calls = 0

    def clock():
        nonlocal in_timing, calls
        in_timing = not in_timing
        calls += 1
        return calls * 0.5

    def checked_copy(value):
        assert not in_timing
        return real_copy(value)

    monkeypatch.setattr(bench.time, "perf_counter", clock)
    monkeypatch.setattr(bench.copy, "deepcopy", checked_copy)
    seconds, moved = bench.measure(staged, 3, 2, 0)
    assert seconds == [0.5, 0.5]
    assert moved[0] == moved[1] == 30
    assert staged.walk_data == original


def test_quick_cli_json_and_patch_cleanup(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DORADO_BACKEND", raising=False)
    monkeypatch.setattr(bench, "rust_available", lambda: False)
    output = tmp_path / "nested" / "results.json"
    original = lw.particle_stepper
    assert bench.main(["--quick", "--output", str(output)]) == 0
    assert lw.particle_stepper is original
    report = json.loads(output.read_text())
    assert report["schema_version"] == 1
    assert report["skipped"][0]["backend"] == "rust"
    assert [row["backend"] for row in report["results"]] == ["python", "numpy"]
    for row in report["results"]:
        assert row["particles"] == 500 and row["steps"] == 3
        assert row["microseconds_per_particle_step"] == pytest.approx(
            row["seconds"][0] * 1e6 / 1500)
    assert "| anuga | numpy | 500 |" in capsys.readouterr().out


@pytest.mark.parametrize("flags", [
    ["--steps", "0"], ["--repeats", "-1"], ["--seed", "-1"],
    ["--seed", str(2**32)], ["--backend", "rust"],
])
def test_invalid_cli(tmp_path, monkeypatch, flags):
    monkeypatch.delenv("DORADO_BACKEND", raising=False)
    monkeypatch.setattr(bench, "rust_available", lambda: False)
    output = tmp_path / "results.json"
    with pytest.raises(SystemExit) as exc:
        bench.main(["--quick", "--output", str(output), *flags])
    assert exc.value.code == 2
    assert not output.exists()


def test_reject_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DORADO_BACKEND", "python")
    with pytest.raises(SystemExit) as exc:
        bench.main(["--quick", "--output", str(tmp_path / "results.json")])
    assert exc.value.code == 2
