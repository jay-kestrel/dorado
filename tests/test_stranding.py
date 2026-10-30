"""Tests for park-while-dry stranding handling and re-entrainment release."""

import pytest
import numpy as np

from dorado import particle_track


class pobj():
    pass


def make_params(depth, cell_type=None, **stranding_params):
    """Build model params on a synthetic domain with unit discharge."""
    params = pobj()
    params.depth = depth
    params.stage = np.ones_like(depth)
    params.qx = np.ones_like(depth)
    params.qy = np.zeros_like(depth)
    params.theta = 1.0
    params.model = 'DeltaRCM'
    params.dx = 1.0
    if cell_type is not None:
        params.cell_type = cell_type
    for name, value in stranding_params.items():
        setattr(params, name, value)
    return params


def dry_center_domain():
    """5x5 all-wet domain with a dry (but not permanent-land) center."""
    depth = np.ones((5, 5))
    depth[2, 2] = 0.05
    cell_type = np.ones((5, 5), dtype='int')
    return depth, cell_type


def dead_end_domain():
    """7x7 wet pocket at (2,2) ringed by dry cells, with deep permanent
    land at (3,2) as the legacy deepest-cell fallback target."""
    depth = np.full((7, 7), 0.05)
    depth[2, 2] = 0.15
    depth[3, 2] = 3.0
    cell_type = np.ones((7, 7), dtype='int')
    cell_type[3, 2] = 2
    return depth, cell_type


def test_park_while_dry():
    # a particle on a dry cell, or on permanent land with wet-depth, parks
    # in place: no movement, stranded=1, travel time advanced by exactly
    # stranding_dt
    depth, cell_type = dry_center_domain()
    depth[4, 4] = 1.0
    cell_type[4, 4] = 2  # permanent land at a wet depth
    particles = particle_track.Particles(make_params(depth, cell_type))
    particles.generate_particles(2, [2, 4], [2, 4], method='exact')
    for _ in range(3):
        particles.run_iteration()
    assert particles.walk_data['xinds'] == [[2, 2, 2, 2], [4, 4, 4, 4]]
    assert particles.walk_data['yinds'] == [[2, 2, 2, 2], [4, 4, 4, 4]]
    assert particles.walk_data['stranded'] == [[0, 1, 1, 1], [0, 1, 1, 1]]
    assert particles.walk_data['travel_times'] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert particles.walk_data['stranded_time'] == [[0, 1, 2, 3],
                                                    [0, 1, 2, 3]]


def test_none_matches_legacy():
    # with stranding_handling='none' the deepest-cell fallback ignores the
    # land mask exactly as on master, and no stranded state is tracked
    depth, cell_type = dead_end_domain()
    legacy = particle_track.Particles(
        make_params(depth, cell_type, stranding_handling='none'))
    legacy.generate_particles(1, [2], [2], method='exact')
    for _ in range(4):
        legacy.run_iteration()
    visited = list(zip(legacy.walk_data['xinds'][0],
                       legacy.walk_data['yinds'][0]))
    # the fallback hops the particle onto permanent land
    assert (3, 2) in visited
    assert 'stranded' not in legacy.walk_data
    assert 'stranded_time' not in legacy.walk_data


def test_immediate_release_on_rewetting():
    # a particle parked on a dry cell is released on the first step after
    # the Particles object is rebuilt on a re-wetted field
    depth, cell_type = dry_center_domain()
    stranded_run = particle_track.Particles(make_params(depth, cell_type))
    stranded_run.generate_particles(1, [2], [2], method='exact')
    stranded_run.run_iteration()
    assert stranded_run.walk_data['stranded'][0][-1] == 1

    rewetted = particle_track.Particles(make_params(np.ones((5, 5))))
    rewetted.generate_particles(0, [], [],
                                previous_walk_data=stranded_run.walk_data)
    rewetted.run_iteration()
    assert rewetted.walk_data['stranded'][0][-1] == 0


def test_rest_release():
    # release happens on the first parked step where the accumulated
    # stranded time reaches rest_time, counting both dry and held-wet
    # steps, with the clock carried across the rebuild
    depth, cell_type = dry_center_domain()
    dry_phase = particle_track.Particles(
        make_params(depth, cell_type, stranding_release='rest',
                    rest_time=15.0, stranding_dt=1.0))
    dry_phase.generate_particles(1, [2], [2], method='exact')
    for _ in range(7):
        dry_phase.run_iteration()
    assert dry_phase.walk_data['stranded_time'][0][-1] == 7.0

    wet_phase = particle_track.Particles(
        make_params(np.ones((21, 21)), stranding_release='rest',
                    rest_time=15.0, stranding_dt=1.0))
    wet_phase.generate_particles(0, [], [],
                                 previous_walk_data=dry_phase.walk_data)
    for _ in range(9):
        wet_phase.run_iteration()
    stranded = wet_phase.walk_data['stranded'][0][8:]
    times = wet_phase.walk_data['stranded_time'][0][8:]
    # 7 steps of dry stranding plus 8 more held-wet steps, then release
    # on the 9th wet step
    assert list(stranded[:8]) == [1] * 8
    assert stranded[8] == 0
    assert times[:8] == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


def test_probability_release():
    # per-step Bernoulli re-entrainment: exact against the replicated
    # draws under a fixed seed, and ensemble fraction ~ entrain_prob
    # (pinned to the python reference: the expected flags replicate the
    # legacy np.random draw sequence, which the native streams do not)
    N = 2000
    depth, cell_type = dry_center_domain()
    dry_phase = particle_track.Particles(
        make_params(depth, cell_type, stranding_release='probability',
                    entrain_prob=0.25), backend='python')
    dry_phase.generate_particles(N, [2] * N, [2] * N, method='exact')
    for _ in range(2):
        dry_phase.run_iteration()
    assert all(rec == 1 for path in dry_phase.walk_data['stranded']
               for rec in path[1:])

    wet_phase = particle_track.Particles(
        make_params(np.ones((21, 21)), stranding_release='probability',
                    entrain_prob=0.25), backend='python')
    wet_phase.generate_particles(0, [], [],
                                 previous_walk_data=dry_phase.walk_data)
    np.random.seed(42)
    trials = np.random.random(N)
    np.random.seed(42)
    wet_phase.run_iteration()
    stranded_after = [path[-1] for path in wet_phase.walk_data['stranded']]
    expected = [0 if trial < 0.25 else 1 for trial in trials]
    np.testing.assert_array_equal(stranded_after, expected)
    released_fraction = np.mean(np.asarray(stranded_after) == 0)
    assert abs(released_fraction - 0.25) < 0.05


def test_no_land_entry():
    # under 'park' no recorded position ever lands on a permanent-land
    # cell: the wet dead-end center is the deepest non-land cell in its
    # own window, so the particle is kept there instead of hopping to land
    depth, cell_type = dead_end_domain()
    particles = particle_track.Particles(make_params(depth, cell_type))
    particles.generate_particles(1, [2], [2], method='exact')
    for _ in range(30):
        particles.run_iteration()
    visited = list(zip(particles.walk_data['xinds'][0],
                       particles.walk_data['yinds'][0]))
    assert (3, 2) not in visited
    assert all(cell_type[x, y] != 2 for x, y in visited)
    assert set(visited) == {(2, 2)}


def test_travel_time_sanity():
    # across a run with dry periods every travel time stays finite and
    # bounded - the floored-velocity ~1e8 s dry hop is unreachable
    depth = np.full((7, 7), 0.05)
    depth[3, 3] = 1.0
    depth[5, 5] = 1.0
    cell_type = np.ones((7, 7), dtype='int')
    cell_type[5, 5] = 2
    particles = particle_track.Particles(make_params(depth, cell_type))
    particles.generate_particles(3, [2, 5, 3], [2, 5, 3], method='exact')
    for _ in range(50):
        particles.run_iteration()
    times = np.asarray([t for path in particles.walk_data['travel_times']
                        for t in path])
    assert np.all(np.isfinite(times))
    assert np.max(times) < 1e4
    # the dry-flat particle parks for the whole run on the stranding clock
    assert particles.walk_data['stranded'][0][-1] == 1
    assert abs(particles.walk_data['stranded_time'][0][-1] - 50.0) < 1e-9


def test_stranding_param_validation():
    depth, cell_type = dry_center_domain()
    bad_configs = [
        {'stranding_handling': 'loosen'},
        {'stranding_handling': 'release'},
        {'stranding_release': 'shields'},
        {'stranding_release': 'wait'},
        {'rest_time': -1.0},
        {'entrain_prob': 0.0},
        {'entrain_prob': 1.5},
        {'entrain_prob': -0.25},
        {'stranding_dt': 0.0},
        {'stranding_dt': -3.0},
    ]
    for bad in bad_configs:
        name = next(iter(bad))
        with pytest.raises(ValueError, match=name):
            particle_track.Particles(make_params(depth, cell_type, **bad))
    # defaults apply when the stranding parameters are unspecified
    defaults = particle_track.Particles(make_params(depth, cell_type))
    assert defaults.stranding_handling == 'park'
    assert defaults.stranding_release == 'immediate'
    assert defaults.rest_time == 0.0
    assert defaults.entrain_prob == 1.0
    assert defaults.stranding_dt == 1.0


def test_analyses_with_stranding():
    # exposure_time and nourishment_time run on a walk that contains
    # parked records
    depth, cell_type = dry_center_domain()
    particles = particle_track.Particles(make_params(depth, cell_type))
    particles.generate_particles(3, [2, 2, 2], [2, 2, 2], method='exact')
    for _ in range(10):
        particles.run_iteration()
    walk = particles.walk_data
    etd = particle_track.exposure_time(walk, np.ones((5, 5)))
    assert np.all(np.isfinite(np.asarray(etd)))
    # nourishment_time NaNs the smoothed-out minimum plateau by design;
    # the compatibility claim is that it runs on parked records and the
    # visited values stay sane (no floored-velocity ~1e8 s poison)
    ntime = particle_track.nourishment_time(walk, (5, 5), sigma=0)
    assert np.nanmax(ntime) > 0
    assert np.nanmax(ntime) < 1e4


def test_legacy_walk_data_pads_stranded():
    # a walk started under 'none' carries no stranded state; rebuilding
    # under 'park' pads the missing state with zeros so each particle's
    # list stays aligned with its recorded steps. The domain is large
    # enough that the legacy drift stays off the padded -1 edge ring,
    # where boundary-stuck particles correctly record no move.
    depth = np.ones((9, 9))
    depth[2, 2] = 0.05
    cell_type = np.ones((9, 9), dtype='int')
    np.random.seed(11)
    legacy = particle_track.Particles(
        make_params(depth, cell_type, stranding_handling='none'))
    legacy.generate_particles(1, [2], [4], method='exact')
    for _ in range(2):
        legacy.run_iteration()
    assert 'stranded' not in legacy.walk_data

    rewetted = particle_track.Particles(make_params(np.ones((9, 9))))
    rewetted.generate_particles(0, [], [],
                                previous_walk_data=legacy.walk_data)
    rewetted.run_iteration()
    # 3 legacy records plus the released step, all zero-padded
    assert rewetted.walk_data['stranded'] == [[0, 0, 0, 0]]
    assert rewetted.walk_data['stranded_time'] == [[0.0, 0.0, 0.0, 0.0]]
