"""Native parity against master 1898034; Python remains the independent oracle.

Use controlled uniforms for primitives and single-particle comparisons. The
native stream is NOT NumPy's global stream; ensemble equivalence is a later gate.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from dorado import _native, lagrangian_walker as lw, particle_track as pt


def particles(**options):
    params = SimpleNamespace(
        depth=np.ones((9, 9)), stage=np.ones((9, 9)),
        qx=np.ones((9, 9)), qy=np.zeros((9, 9)),
        cell_type=np.ones((9, 9)), dx=2.0, model="DeltaRCM",
        theta=1.0, diff_coeff=0.2, **options,
    )
    p = pt.Particles(params, backend="python")
    p.weight[:] = 0.0
    p.weight[:, :, 5] = 1.0  # deterministic eastward proposal
    return p


def stage(p):
    """Test adapter; production staging lives in particle_track.py."""
    return _native.RoutingFields(
        np.ascontiguousarray(p.weight, dtype=np.float64),
        np.ascontiguousarray(p.velocity, dtype=np.float64),
        np.ascontiguousarray(p.velocity_angle, dtype=np.float64),
        np.ascontiguousarray(p.cell_type, dtype=np.int8),
        np.ascontiguousarray(p.depth, dtype=np.float64),
        np.asarray(p.distances, dtype=np.float64).ravel(),
        np.asarray(p.iwalk, dtype=np.int64).ravel(),
        np.asarray(p.jwalk, dtype=np.int64).ravel(),
        np.asarray(p.angles, dtype=np.float64).ravel(), p.dx, p.diff_coeff,
        steepest_descent=p.steepest_descent,
        stranding_handling=p.stranding_handling, dry_depth=p.dry_depth,
        stranding_dt=p.stranding_dt, stranding_release=p.stranding_release,
        rest_time=p.rest_time, entrain_prob=p.entrain_prob,
    )


def buffers(positions):
    pos = np.asarray(positions, dtype=np.int64).reshape(-1, 2)
    return pos, np.zeros(len(pos)), np.zeros((len(pos), 3))


def controlled(monkeypatch, values):
    draws = iter(values)
    monkeypatch.setattr(np.random, "uniform", lambda low=0, high=1: low + (high-low)*next(draws))
    monkeypatch.setattr(lw, "random", lambda: next(draws))


@pytest.mark.parametrize("steepest", [False, True])
@pytest.mark.parametrize("uniform", [0.0, 0.1, 0.5, 0.999999999999])
@pytest.mark.parametrize("weights", [
    [0.0]*9, [np.nan]*9, [0, 1, 2, np.nan, 99, 2, 0, 0, 0],
    [1.0]*9, [0, 0, 0, 0, 1, 0, 0, 0, 0],
])
def test_pick_matches_python(monkeypatch, weights, uniform, steepest):
    controlled(monkeypatch, [uniform])
    weights = np.asarray(weights, dtype=float)
    original = weights.copy()
    expected = (lw.steep_descent if steepest else lw.random_pick)(weights)
    assert _native._pick(weights, uniform, steepest) == expected
    np.testing.assert_array_equal(weights, original)


@pytest.mark.parametrize("cell", range(9))
@pytest.mark.parametrize("diff_coeff", [0.0, 0.2])
def test_d8_distance_and_travel_time_match_helpers(monkeypatch, cell, diff_coeff):
    p = particles()
    p.diff_coeff = diff_coeff
    p.velocity[:] = 2.0
    p.velocity[3, 3] = 4.0
    p.velocity_angle[:] = 0.13
    p.weight[:] = 0.0
    p.weight[:, :, cell] = 1.0
    pos, times, state = buffers([[4, 4]])
    draws = _native.RngStreams(91).uniforms(0, 2)
    controlled(monkeypatch, draws[1:])
    expected_pos = lw.calculate_new_ind((4, 4), cell, p.iwalk, p.jwalk)
    expected_dt = lw.calc_travel_times(p, cell, [4, 4], list(expected_pos),
                                     lw.step_update(cell, p.distances, p.dx))
    result = _native.step_particles(stage(p), pos, times, state, _native.RngStreams(91))
    assert pos.tolist() == [list(expected_pos)]
    assert times[0] == pytest.approx(expected_dt, rel=1e-14, abs=1e-14)
    assert result.recorded == [cell != 4]


@pytest.mark.parametrize("case", ["boundary", "sentinel", "offgrid", "zero_distance", "deposited"])
def test_freeze_and_zero_distance(case):
    p = particles(stranding_handling="none")
    start = [4, 4]
    if case == "boundary":
        p.cell_type[3, 3] = -1
    elif case == "sentinel":
        start = [0, 0]
    elif case == "offgrid":
        start = [4, 8]
    elif case == "zero_distance":
        p.distances.flat[5] = 0
    pos, times, state = buffers([start])
    if case == "deposited":
        state[0, 0] = 1
    rng = _native.RngStreams(17)
    result = _native.step_particles(stage(p), pos, times, state, rng)
    assert pos.tolist() == [start]  # zero distance must NOT map to (0, 0)
    assert times.tolist() == [0.0]
    assert result.recorded == [False]
    assert rng.draw_counts == [1 if case in ("offgrid", "zero_distance") else 0]
    if case != "deposited":
        expected, dt, _ = lw.particle_stepper(p, [start], [0.0], {})
        assert expected == pos.tolist()
        assert dt == times.tolist()


def test_clamped_top_edge_neighborhood():
    p = particles(stranding_handling="none")
    p.cell_type[0, 3] = -1
    pos, times, state = buffers([[0, 4]])
    rng = _native.RngStreams(0)
    _native.step_particles(stage(p), pos, times, state, rng)
    assert pos.tolist() == [[0, 4]]
    assert rng.draw_counts == [0]


@pytest.mark.parametrize("steepest", [False, True])
@pytest.mark.parametrize("row", [[0.0]*9, [np.nan]*9, [0, 0, 1, 0, 100, 1, 0, 0, 0]])
def test_nan_dead_end_and_center_exclusion(monkeypatch, steepest, row):
    p = particles(steepest_descent=steepest)
    p.weight[:] = row
    original = p.weight.copy()
    controlled(monkeypatch, _native.RngStreams(13).uniforms(0, 3))
    expected_pos, expected_time, _ = lw.particle_stepper(p, [[4, 4]], [0.0], {})
    pos, times, state = buffers([[4, 4]])
    _native.step_particles(stage(p), pos, times, state, _native.RngStreams(13))
    assert pos.tolist() == expected_pos
    np.testing.assert_allclose(times, expected_time, rtol=1e-14)
    np.testing.assert_array_equal(p.weight, original)


@pytest.mark.parametrize("kind", [None, "sand", "mud", "plankton"])
@pytest.mark.parametrize("size", [None, 0.5, 3.0])
def test_kinds_sizes_and_diffusion(monkeypatch, kind, size):
    p = particles()
    p.velocity_angle[:] = 0.0
    behavior = np.array([lw.effective_multipliers(size, kind)])
    controlled(monkeypatch, _native.RngStreams(7).uniforms(0, 5))
    optional = {"particle_size": [size], "particle_type": [kind], "deposited": [0]}
    expected_pos, expected_time, expected_optional = lw.particle_stepper(p, [[4, 4]], [0.0], optional)
    pos, times, state = buffers([[4, 4]])
    _native.step_particles(stage(p), pos, times, state, _native.RngStreams(7), behavior=behavior)
    assert pos.tolist() == expected_pos
    np.testing.assert_allclose(times, expected_time, rtol=1e-14)
    assert state[0, 0] == expected_optional["deposited"][0]


def test_deposition_persists_without_history_growth():
    p = particles()
    pos, times, state = buffers([[4, 4]])
    rng = _native.RngStreams(41)
    behavior = np.array([[1.0, 1.0, 1.0]])
    result = _native.step_particles(stage(p), pos, times, state, rng, behavior=behavior)
    assert result.recorded == [False]
    assert state[0, 0] == 1
    assert rng.draw_counts == [2]  # route and deposition, no diffusion
    _native.step_particles(stage(p), pos, times, state, rng, behavior=behavior)
    assert rng.draw_counts == [2]
    assert pos.tolist() == [[4, 4]]


@pytest.mark.parametrize("release", ["immediate", "rest", "probability"])
@pytest.mark.parametrize("site", ["wet", "dry", "land", "nan", "wet_boundary", "dry_boundary"])
def test_park_release_parity(monkeypatch, release, site):
    p = particles(stranding_release=release, rest_time=3.0, entrain_prob=0.3, stranding_dt=2.0)
    if site in ("dry", "dry_boundary"):
        p.depth[4, 4] = 0.0
    elif site == "land":
        p.cell_type[4, 4] = 2
    elif site == "nan":
        p.depth[4, 4] = np.nan
    if "boundary" in site:
        p.cell_type[3, 3] = -1
    pos, times, state = buffers([[4, 4]])
    state[0] = [0, 1, 2]
    rng = _native.RngStreams(51)
    draws = _native.RngStreams(51).uniforms(0, 5)
    controlled(monkeypatch, draws)
    expected_pos, expected_time, expected = lw.particle_stepper(
        p, [[4, 4]], [0.0], {"stranded": [1], "stranded_time": [2.0]})
    result = _native.step_particles(stage(p), pos, times, state, rng)
    assert pos.tolist() == expected_pos
    np.testing.assert_allclose(times, expected_time, rtol=1e-14)
    assert state[0, 1:].tolist() == [expected["stranded"][0], expected["stranded_time"][0]]
    assert result.recorded == [expected_pos != [[4, 4]] or bool(expected["stranded"][0])]
    if site == "wet_boundary":
        assert rng.draw_counts == [int(release == "probability")]
    if site in ("dry", "dry_boundary", "land", "nan"):
        assert rng.draw_counts == [0]
        assert times.tolist() == [2.0]


def test_release_on_rewetting_and_rest_clock():
    p = particles(stranding_release="rest", rest_time=2.0)
    p.depth[4, 4] = 0.0
    pos, times, state = buffers([[4, 4]])
    rng = _native.RngStreams(9)
    _native.step_particles(stage(p), pos, times, state, rng)
    assert state[0].tolist() == [0, 1, 1]
    p.depth[4, 4] = 1.0
    _native.step_particles(stage(p), pos, times, state, rng)
    assert state[0].tolist() == [0, 1, 2]
    _native.step_particles(stage(p), pos, times, state, rng)
    assert pos.tolist() == [[4, 5]]
    assert state[0].tolist() == [0, 0, 2]


@pytest.mark.parametrize("mode", ["moving", "parked", "boundary", "sentinel", "instantaneous"])
@pytest.mark.parametrize("target,max_iter", [(0.0, 20), (2.5, 20), (100.0, 2)])
def test_adaptive_matches_run_iteration(monkeypatch, mode, target, max_iter):
    p = particles()
    p.diff_coeff = 0.0
    p.velocity_angle[:] = 0.0
    start = [4, 4]
    if mode == "parked":
        p.depth[4, 4] = 0.0
    elif mode == "boundary":
        p.cell_type[3, 3] = -1
    elif mode == "sentinel":
        start = [0, 0]
    elif mode == "instantaneous":
        p.velocity_angle[:] = np.pi
    p.generate_particles(1, [start[0]], [start[1]], method="exact")
    controlled(monkeypatch, [0.5]*200)
    expected = p.run_iteration(target_time=target, max_iter=max_iter)
    pos, times, state = buffers([start])
    result = _native.step_particles(stage(p), pos, times, state, _native.RngStreams(3),
                                    target_time=target, max_iter=max_iter)
    history = result.histories[0]
    assert [start[0]] + [r[0] for r in history] == expected["xinds"][0]
    assert [start[1]] + [r[1] for r in history] == expected["yinds"][0]
    assert [0.0] + [r[2] for r in history] == pytest.approx(expected["travel_times"][0])
    assert len(history) <= max_iter
    assert result.hit_max_iter == [len(history) == max_iter]


def test_adaptive_deposition_updates_last_history_row(monkeypatch):
    p = particles()
    p.velocity_angle[:] = 0.0
    # Seed 2 moves once, then deposits on its next proposal.
    draws = _native.RngStreams(2).uniforms(0, 10)
    probability = (draws[1] + draws[4]) / 2
    monkeypatch.setitem(lw.KIND_PRESETS, "test_kind", {
        "travel_time_multiplier": 1.0, "diffusion_multiplier": 1.0,
        "deposition_probability": probability,
    })
    p.generate_particles(1, [4], [4], method="exact", particle_type="test_kind")
    controlled(monkeypatch, draws)
    expected = p.run_iteration(target_time=100.0)
    pos, times, state = buffers([[4, 4]])
    result = _native.step_particles(
        stage(p), pos, times, state, _native.RngStreams(2), target_time=100.0,
        behavior=np.array([[1.0, 1.0, probability]]))
    assert len(result.histories[0]) == 1
    assert result.histories[0][-1][3] is True
    assert expected["deposited"][0] == [0, 1]
    assert state[0, 0] == 1
    assert times[0] == pytest.approx(expected["travel_times"][0][-1])


def test_adaptive_release_without_motion_retains_previous_flag():
    p = particles()
    p.weight[:] = 0.0
    p.weight[:, :, 4] = 1.0
    pos, times, state = buffers([[4, 4]])
    state[0] = [0, 1, 2]
    result = _native.step_particles(stage(p), pos, times, state, _native.RngStreams(8), target_time=3.0)
    assert result.histories == [[]]
    assert state.tolist() == [[0, 1, 2]]


def test_adaptive_initial_estimate_and_noop():
    p = particles()
    pos, times, state = buffers([[4, 4]])
    rng = _native.RngStreams(7)
    result = _native.step_particles(stage(p), pos, times, state, rng, target_time=2.0,
                                    est_next_dt=np.array([10.0]))
    assert result.histories == [[]]
    assert rng.draw_counts == [0]


@pytest.mark.parametrize("target", [None, 20.0])
def test_one_and_eight_threads_identical_walk_data(target):
    p = particles(stranding_release="probability", entrain_prob=0.25)
    p.weight[:] = 1.0
    p.depth[3, 3] = 0.0
    fields = stage(p)
    runs = []
    for threads in (1, 8):
        pos, times, state = buffers([[4, 4], [3, 3], [0, 0]]*12)
        state[:, 1] = 1
        behavior = np.tile([2.0, 0.5, 0.03], (len(pos), 1))
        rng = _native.RngStreams(1234)
        walk_data = {key: [[v] for v in vals] for key, vals in {
            "xinds": pos[:, 0].tolist(), "yinds": pos[:, 1].tolist(),
            "travel_times": times.tolist(), "deposited": state[:, 0].tolist(),
            "stranded": state[:, 1].tolist(), "stranded_time": state[:, 2].tolist(),
        }.items()}
        for _ in range(8):
            result = _native.step_particles(fields, pos, times, state, rng,
                                            behavior=behavior, n_threads=threads,
                                            target_time=target, max_iter=30)
            for i, recorded in enumerate(result.recorded):
                rows = result.histories[i] if target is not None else (
                    [(int(pos[i, 0]), int(pos[i, 1]), times[i], *state[i])] if recorded else [])
                for row in rows:
                    for key, value in zip(walk_data, row):
                        walk_data[key][i].append(value)
                walk_data["deposited"][i][-1] = state[i, 0]
        runs.append((walk_data, rng.draw_counts))
    assert runs[0] == runs[1]


def test_stream_growth_and_seed_contract():
    rng = _native.RngStreams(123)
    first = rng.uniforms(0, 4)
    second = rng.uniforms(1, 4)
    assert first != second
    assert first + rng.uniforms(0, 4) == _native.RngStreams(123).uniforms(0, 8)
    entropy = _native.RngStreams()
    assert entropy.uniforms(0, 8) == _native.RngStreams(entropy.seed).uniforms(0, 8)
    np.random.seed(1)
    a = _native.RngStreams(123).uniforms(0, 4)
    np.random.seed(999)
    assert _native.RngStreams(123).uniforms(0, 4) == a


@pytest.mark.parametrize("case", ["position", "times", "state", "behavior", "estimate", "threads", "max_iter", "target"])
def test_invalid_batch_is_atomic(case):
    fields = stage(particles())
    pos, times, state = buffers([[4, 4], [3, 3]])
    kwargs = {}
    if case == "position":
        pos[1, 0] = -1
    elif case == "times":
        times[1] = np.nan
    elif case == "state":
        state[1, 1] = 2
    elif case == "behavior":
        kwargs["behavior"] = np.array([[1, 1, 0], [1, 1, 2]], dtype=float)
    elif case == "estimate":
        kwargs["est_next_dt"] = np.array([1.0, np.nan])
    elif case == "threads":
        kwargs["n_threads"] = 0
    elif case == "max_iter":
        kwargs["max_iter"] = 0
    else:
        kwargs["target_time"] = np.inf
    before = [a.copy() for a in (pos, times, state)]
    rng = _native.RngStreams(7)
    with pytest.raises(ValueError):
        _native.step_particles(fields, pos, times, state, rng, **kwargs)
    for a, b in zip((pos, times, state), before):
        np.testing.assert_array_equal(a, b)
    assert rng.draw_counts == []


def test_shapes_strides_aliases_and_empty_batch():
    fields = stage(particles())
    pos, times, state = buffers([[4, 4], [3, 3]])
    rng = _native.RngStreams(7)
    with pytest.raises(ValueError):
        _native.step_particles(fields, pos[:, :1], times, state, rng)
    with pytest.raises((ValueError, TypeError)):
        _native.step_particles(fields, pos[::-1], times, state, rng)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _native.step_particles(fields, pos, state[:, 0], state, rng)
    pos.flags.writeable = False
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        _native.step_particles(fields, pos, times, state, rng)
    for threads in (1, 8):
        empty = buffers([])
        result = _native.step_particles(fields, *empty, rng, n_threads=threads)
        assert result.recorded == result.hit_max_iter == result.histories == []


@pytest.mark.parametrize("case", ["negative_weights", "infinite_weights", "row_overflow", "velocity", "offsets", "distances", "shape"])
def test_invalid_fields_rejected(case):
    p = particles()
    if case == "negative_weights":
        p.weight[0, 0, 0] = -1
    elif case == "infinite_weights":
        p.weight[0, 0, 0] = np.inf
    elif case == "row_overflow":
        p.weight[0, 0, :] = np.finfo(float).max
    elif case == "velocity":
        p.velocity[0, 0] = 0
    elif case == "offsets":
        p.iwalk.flat[0] = 2
    elif case == "distances":
        p.distances.flat[0] = -1
    else:
        p.weight = p.weight[:, :, :8]
    with pytest.raises(ValueError):
        stage(p)


def test_fields_are_owned_snapshot():
    p = particles()
    fields = stage(p)
    p.weight[:] = 0
    p.depth[:] = 0
    pos, times, state = buffers([[4, 4]])
    _native.step_particles(fields, pos, times, state, _native.RngStreams(4))
    assert pos.tolist() == [[4, 5]]
