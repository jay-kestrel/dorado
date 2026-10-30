# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:18:10 2025

@author: cturn
"""
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pytest

dorado = pytest.importorskip("dorado")
spat = pytest.importorskip("dorado.spatial")
pt = pytest.importorskip("dorado.particle_track")

@pytest.fixture
def test_grid():
    elevation = np.arange(12, dtype=float).reshape(6, 2)
    celltype = np.zeros((6, 2), dtype=int)
    celltype[0:3, 0:1] = 1
    resolution_factor = 2
    timedelta = 1
    return elevation, celltype, resolution_factor, timedelta

@pytest.fixture
# The walk_data defined here is mock data used to test functionality of dorado.spatial(). 
def walk_data():
    return {
        "yinds": [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
        "xinds": [
            [0, 1, 2, 3, 4, 5, 5, 5],
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [0, 0, 0, 1, 1, 1, 1, 5],
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5],
            [1, 1, 1, 2, 2, 3, 3, 5],
            [1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 5],
            [2, 2, 2, 1, 2, 2, 2, 5, 5],
            [2, 2, 2, 1, 4, 4, 4, 4, 4, 4, 5],
            [2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5],
            [1, 2, 3, 4],
            [0, 1, 2, 2, 3, 4],
            [0, 1, 1, 1],
            [2, 3, 4],
            [2, 2, 3, 4],
            [2, 2, 2, 2]],
        "travel_times": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3],
            [0, 1, 2],
            [0, 1, 2, 3],
            [0, 1, 2, 3]]}

def test_exposure_time_and_threshold_plots(test_grid, walk_data):
    # calulate exposure-time 
    elevation, celltype, resolution_factor, timedelta = test_grid
    exposure_times = pt.exposure_time(walk_data, celltype)
    assert exposure_times is not None

    # plotting exposure plot at thresholds
    dorado.routines.plot_exposure_time_thresholds(
        walk_data, exposure_times, timedelta=timedelta, uniform_timesteps=True)
    dorado.routines.plot_exposure_time_thresholds(
        walk_data, exposure_times, timedelta=timedelta, uniform_timesteps=False)

def test_compute_thresholds(test_grid, walk_data):
    elevation, celltype, resolution_factor, timedelta = test_grid

    sys_exp = spat.systemwide(walk_data, elevation, celltype, resolution_factor)
    E50, E75, E90 = spat.compute_thresholds(sys_exp, elevation, timedelta, resolution_factor)

    rows, cols = elevation.shape
    target_shape = (rows // resolution_factor, cols // resolution_factor)
    for E in (E50, E75, E90):
        assert isinstance(E, np.ndarray)
        assert E.shape == target_shape
        assert np.nanmin(E) >= 0.0

def test_plot_spatial(test_grid, walk_data):
    elevation, celltype, resolution_factor, timedelta = test_grid
    # systemwide
    sys_exp = spat.systemwide(walk_data, elevation, celltype, resolution_factor)
    E50, E75, E90 = spat.compute_thresholds(sys_exp, elevation, timedelta, resolution_factor)
    spat.plot_spatial(
        exposure_maps=[E50, E75, E90],
        elevation=elevation,
        max_exposure=7,
        cbar_levels=7,
        cmap=matplotlib.colormaps["plasma_r"] if hasattr(matplotlib, "colormaps") else plt.get_cmap("plasma_r"),
        title="Test System-Wide (s)")
    plt.close("all")

    # localized
    loc_exp = spat.localized(walk_data, elevation, celltype, resolution_factor)
    e50, e75, e90 = spat.compute_thresholds(loc_exp, elevation, timedelta, resolution_factor)
    spat.plot_spatial(
        exposure_maps=[e50, e75, e90],
        elevation=elevation,
        max_exposure=7,
        cbar_levels=7,
        cmap=matplotlib.colormaps["viridis_r"] if hasattr(matplotlib, "colormaps") else plt.get_cmap("viridis_r"),
        title="Test Localized (s)")
    plt.close("all")


def test_boundary_particle_attributed_to_exactly_one_chunk():
    """BUG-4: the inclusive upper bound double-counted particles whose
    initial index equals a chunk's exclusive end (x0 == r1)."""
    boundary = {
        'xinds': [[2, 3, 4]],
        'yinds': [[0, 0, 0]],
        'travel_times': [[0, 1, 2]],
    }
    # 6x2 domain, factor 2 -> row chunks [0,2), [2,4), [4,6); x0=2 is the
    # first row of the middle chunk and the old code also claimed it for
    # the lower chunk
    lower = spat._find_common_indices_info(boundary, (0, 2), (0, 2))
    middle = spat._find_common_indices_info(boundary, (2, 4), (0, 2))
    upper = spat._find_common_indices_info(boundary, (4, 6), (0, 2))
    assert len(lower['xinds']) == 0
    assert len(middle['xinds']) == 1
    assert len(upper['xinds']) == 0


def test_plot_spatial_returns_non_empty_axes(test_grid, walk_data):
    """F-23: plot_spatial created the per-chunk axes but returned a
    permanently empty list."""
    elevation, celltype, resolution_factor, timedelta = test_grid
    sys_exp = spat.systemwide(walk_data, elevation, celltype, resolution_factor)
    E50, E75, E90 = spat.compute_thresholds(sys_exp, elevation, timedelta,
                                            resolution_factor)
    fig, axes, cax, cbar = spat.plot_spatial(
        exposure_maps=[E50, E75, E90],
        elevation=elevation,
        max_exposure=7,
        cbar_levels=7,
        cmap=matplotlib.colormaps['plasma_r'],
        title='Axes Return Test')
    assert len(axes) == 3
    plt.close('all')
