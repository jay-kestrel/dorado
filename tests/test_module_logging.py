# -*- coding: utf-8 -*-
"""Tests for run-level INFO logging in the core modules.

Asserts that dorado.lagrangian_walker, dorado.parallel_routing, and
dorado.spatial carry child loggers propagating to the 'dorado' parent
(so DORADO_LOG_LEVEL controls them) and emit their run-level summaries.
"""
import logging

import numpy as np
import pytest

import dorado.particle_track as particle_track
import dorado.spatial as spatial
from dorado import lagrangian_walker as lw
from dorado import parallel_routing


@pytest.fixture
def small_particles():
    """Small 5x5 DeltaRCM particles object for a quick walk fixture."""
    params = particle_track.modelParams()
    params.dx = 1
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    params.model = 'DeltaRCM'
    return particle_track.Particles(params)


@pytest.fixture
def spatial_inputs():
    """Small 6x2 grid with ROI cells in the top half, plus walk data."""
    elevation = np.arange(12, dtype=float).reshape(6, 2)
    celltype = np.zeros((6, 2), dtype=int)
    celltype[0:3, 0:1] = 1
    walk_data = {
        'xinds': [[0, 1, 2], [2, 3, 3], [4, 4, 4]],
        'yinds': [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        'travel_times': [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    }
    return elevation, celltype, walk_data


def records_from(caplog, logger_name, fragment):
    return [r for r in caplog.records
            if r.name == logger_name and fragment in r.getMessage()]


def test_core_module_loggers_propagate_to_dorado_parent():
    parent = logging.getLogger('dorado')
    for name in ('dorado.lagrangian_walker',
                 'dorado.parallel_routing',
                 'dorado.spatial'):
        assert logging.getLogger(name).parent is parent


def test_run_walk_logs_start_and_completion(small_particles, caplog):
    np.random.seed(0)
    small_particles.generate_particles(2, [1], [1])

    with caplog.at_level(logging.INFO, logger='dorado.lagrangian_walker'):
        lw.run_walk(small_particles, 3, 2)

    start = records_from(caplog, 'dorado.lagrangian_walker',
                         'Walk run starting')
    complete = records_from(caplog, 'dorado.lagrangian_walker',
                            'Walk run complete')
    assert len(start) == 1
    assert '2 particles' in start[0].getMessage()
    assert 'weighted random walk' in start[0].getMessage()
    assert '3 iterations' in start[0].getMessage()
    assert len(complete) == 1
    assert '3 iterations' in complete[0].getMessage()


def test_run_walk_logs_steepest_descent_mode(small_particles, caplog):
    np.random.seed(0)
    small_particles.generate_particles(1, [1], [1])
    small_particles.steepest_descent = True

    with caplog.at_level(logging.INFO, logger='dorado.lagrangian_walker'):
        lw.run_walk(small_particles, 1, 1)

    start = records_from(caplog, 'dorado.lagrangian_walker',
                         'Walk run starting')
    assert len(start) == 1
    assert 'steepest descent' in start[0].getMessage()


def test_parallel_routing_logs_worker_summary(small_particles, caplog):
    np.random.seed(0)
    small_particles.generate_particles(2, [1], [1])

    with caplog.at_level(logging.INFO, logger='dorado.parallel_routing'):
        parallel_routing.parallel_routing(small_particles, 2, 2, [1], [1], 2)

    summary = records_from(caplog, 'dorado.parallel_routing',
                           'Parallel routing complete')
    assert len(summary) == 1
    message = summary[0].getMessage()
    assert '2 workers' in message
    assert '2 particles per worker' in message
    assert '2 iterations each' in message


def test_spatial_logs_analysis_summaries(spatial_inputs, caplog):
    elevation, celltype, walk_data = spatial_inputs

    with caplog.at_level(logging.INFO, logger='dorado.spatial'):
        sys_exp = spatial.systemwide(walk_data, elevation, celltype, 2)
        spatial.localized(walk_data, elevation, celltype, 2)
        spatial.compute_thresholds(sys_exp, elevation, 1, 2)

    systemwide_logs = records_from(caplog, 'dorado.spatial',
                                   'Systemwide exposure analysis')
    localized_logs = records_from(caplog, 'dorado.spatial',
                                  'Localized exposure analysis')
    thresholds_logs = records_from(caplog, 'dorado.spatial',
                                   'Exposure thresholds computed')
    assert len(systemwide_logs) == 1
    assert '6x2 grid' in systemwide_logs[0].getMessage()
    assert '2 of 3 chunks analyzed' in systemwide_logs[0].getMessage()
    assert len(localized_logs) == 1
    assert len(thresholds_logs) == 1
    assert '3x1 chunk grid' in thresholds_logs[0].getMessage()
