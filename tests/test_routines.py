import pytest
import warnings
from dorado import routines

class DummyParams:
    def __init__(self):
        self.dx = 1
        self.depth = [[1]]
        self.qx = [[0]]
        self.qy = [[1]]
        self.theta = 1
        self.model = 'DeltaRCM'

def test_routines_verbose_deprecation():
    # Test steady_plots
    walk_data = {'x': [], 'y': []}
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        try:
            routines.steady_plots(walk_data, 1, verbose=True)
        except Exception:
            pass # We just care about the warning
            
    with pytest.warns(DeprecationWarning, match="The 'verbose' parameter is deprecated"):
        try:
            routines.time_plots(walk_data, 1, verbose=True)
        except Exception:
            pass


import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _walk_data(optional_var=None):
    """Synthetic walks (no routing needed): four particles in a 10x10
    domain, each ending at travel time 10 so the ETD end_time is well
    defined."""
    walk_data = {
        'xinds': [[1, 2, 4, 6], [2, 3, 5, 7], [1, 1, 3, 5], [2, 2, 4, 6]],
        'yinds': [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        'travel_times': [[0, 1, 2, 10], [0, 2, 4, 10],
                         [0, 1, 3, 10], [0, 2, 5, 10]],
    }
    if optional_var is not None:
        walk_data[optional_var] = [[0.5] * len(w) for w in walk_data['xinds']]
    return walk_data


def test_get_state_contract_is_always_4_tuple():
    """BUG-2: get_state returned 3 values for base walk_data and 4 values
    when optional variables were tracked; the contract is now stable."""
    state = routines.get_state(_walk_data())
    assert len(state) == 4
    assert state[3] == {}

    x, y, end_time, optional_data = routines.get_state(
        _walk_data(optional_var='depth'))
    assert 'depth' in optional_data

    # get_time_state shares the stable 4-tuple contract on base walk_data
    # (no optional variables tracked): the optional dict is always present
    tstate = routines.get_time_state(_walk_data(), 10)
    assert len(tstate) == 4
    assert tstate[3] == {}


def test_plot_exposure_time_runs_on_default_and_optional_walk_data():
    """BUG-2: plot_exposure_time crashed with a 3-value unpack error on
    walk_data from a default run_iteration (no optional variables)."""
    exposure_times = [2.5, 4.5, 6.5, 0.0]

    routines.plot_exposure_time(_walk_data(), exposure_times,
                                save_output=False)
    routines.plot_exposure_time(_walk_data(optional_var='depth'),
                                exposure_times, save_output=False)
    plt.close('all')


def test_plot_exposure_time_thresholds_runs_on_both_walk_data_shapes():
    """BUG-2: plot_exposure_time_thresholds crashed when walk_data tracked
    optional variables (4-value unpack into 3 names)."""
    exposure_times = [2.5, 4.5, 6.5, 0.0]

    routines.plot_exposure_time_thresholds(_walk_data(), exposure_times,
                                           save_output=False)
    routines.plot_exposure_time_thresholds(
        _walk_data(optional_var='depth'), exposure_times, save_output=False)
    plt.close('all')


def test_thresholds_report_not_reached_when_cdf_below_percentiles(caplog):
    """BUG-3: an unclamped searchsorted raised IndexError when the CDF
    never reached a percentile (1 of 5 particles included here, so the
    CDF tops out at 0.2)."""
    walk_data = {
        'xinds': [[1, 2], [2, 3], [1, 1], [2, 2], [1, 2]],
        'yinds': [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
        'travel_times': [[0, 10]] * 5,
    }
    exposure_times = [5.0, 0.0, 0.0, 0.0, 0.0]

    with caplog.at_level(logging.INFO, logger='dorado.routines'):
        routines.plot_exposure_time_thresholds(walk_data, exposure_times,
                                               save_output=False)
    assert 'not reached' in caplog.text
    plt.close('all')


class _UnusedParticle:
    """Zero iterations never touch the particle object, so a stub avoids
    building a full routing domain."""


def test_steady_plots_zero_iterations_does_not_raise(tmp_path):
    """BUG-5: num_iter=0 with save_output=True crashed with an
    UnboundLocalError at the json dump."""
    assert routines.steady_plots(_UnusedParticle(), 0,
                                 folder_name=str(tmp_path)) is None
    assert not (tmp_path / 'data' / 'data.txt').exists()


def test_time_plots_zero_iterations_does_not_raise(tmp_path):
    assert routines.time_plots(_UnusedParticle(), 0,
                               folder_name=str(tmp_path)) is None


def test_unsteady_plots_zero_steps_does_not_raise(tmp_path):
    base = tmp_path / 'outputs'
    base.mkdir()
    grid = np.ones((3, 3))
    for name in ['depth1', 'stage1', 'qx1', 'qy1']:
        np.savetxt(base / (name + '.csv'), grid, delimiter=',')

    result = routines.unsteady_plots(dx=1, Np_tracer=1, seed_xloc=[1],
                                     seed_yloc=[1], num_steps=0, timestep=1.0,
                                     output_base=str(base), output_type='csv',
                                     folder_name=str(tmp_path / 'out'))
    assert result is None


def test_unsteady_plots_mismatched_file_lists_raise_named_error(tmp_path):
    """F-7: a depth series with one more file than the qx series used to
    pass the old max()-based guard and crash mid-run with an IndexError."""
    base = tmp_path / 'outputs'
    base.mkdir()
    grid = np.ones((3, 3))
    for name in ['depth1', 'depth2', 'stage1', 'qx1', 'qy1']:
        np.savetxt(base / (name + '.csv'), grid, delimiter=',')

    with pytest.raises(ValueError,
                       match='Mismatched model output file lists'):
        routines.unsteady_plots(dx=1, Np_tracer=1, seed_xloc=[1],
                                seed_yloc=[1], num_steps=2, timestep=1.0,
                                output_base=str(base), output_type='csv',
                                folder_name=str(tmp_path / 'out'))


def test_nourishment_plots_run_on_current_matplotlib_with_nan_grid():
    """F-10/F-11: the nourishment plots kept a plt.cm.get_cmap fallback
    (removed in matplotlib 3.9) and NaN-unsafe np.max grid scaling; both
    must work on a modern matplotlib with NaN land cells in the grid."""
    grid = np.ones((10, 10))
    grid[0, 0] = np.nan
    visit_freq = np.linspace(0, 1, 100).reshape(10, 10)
    mean_times = np.linspace(0, 10, 100).reshape(10, 10)
    seed_walk = {'xinds': [[1, 2]], 'yinds': [[1, 2]],
                 'travel_times': [[0.0, 1.0]]}

    ax = routines.show_nourishment_area(visit_freq, grid=grid,
                                        walk_data=seed_walk)
    assert np.isfinite(ax.images[0].norm.vmax)

    ax = routines.show_nourishment_time(mean_times, grid=grid,
                                        walk_data=seed_walk,
                                        show_colorbar=True)
    assert np.isfinite(ax.images[0].norm.vmax)
    plt.close('all')

