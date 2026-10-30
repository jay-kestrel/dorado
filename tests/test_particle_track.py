from __future__ import division, print_function, absolute_import
from builtins import range, map
from math import floor, sqrt, pi
import copy
import logging
import pytest
import io
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from dorado import particle_track
from dorado import lagrangian_walker
from dorado import parallel_routing
import numpy as np

# init some parameters
class pobj():
    pass

# instantiate so the deep-copied goodparams below is an actual isolated
# copy — deepcopy of a class object returns the same class
params = pobj()
seed_xloc = [1]
seed_yloc = [1]
Np_tracer = 1
params.dx = 1
params.depth = np.ones((3,3))
params.stage = np.ones((3,3))
params.qx = np.zeros((3,3))
params.qy = np.ones((3,3))
params.theta = 1
params.model = 'DeltaRCM'
params.roi_grid = np.ones((3,3))
params.particle_variables = ['depth', 'qx'] # optional variables to track with particles
goodparams = copy.deepcopy(params)  # don't corrupt these good parameters
particle = particle_track.Particles(params)

# testing of the Particle __init__ functionality
def test_dx():
    assert particle.dx == params.dx

def test_depth():
    assert np.all(particle.depth) == np.all(params.depth)

def test_stage():
    assert np.all(particle.stage) == np.all(params.stage)

def test_qx():
    assert np.all(particle.qx) == np.all(params.qx)

def test_qy():
    assert np.all(particle.qy) == np.all(params.qy)

def test_velocity():
    vel = np.sqrt(params.qx**2+params.qy**2)/params.depth
    assert np.all(particle.velocity) == np.all(vel)

def test_theta():
    assert particle.theta == params.theta

def test_neg_diffcoeff(caplog):
    params.diff_coeff = -1.0
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle = particle_track.Particles(params)
    assert any(
        record.levelno == logging.WARNING
        and "Specified diffusion coefficient is negative" in record.getMessage()
        for record in caplog.records
    )
    assert particle.diff_coeff == 0.0

def test_big_diffcoeff(caplog):
    params.diff_coeff = 3.0
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle = particle_track.Particles(params)
    assert any(
        record.levelno == logging.WARNING
        and "Diffusion behaves non-physically when" in record.getMessage()
        for record in caplog.records
    )
    assert particle.diff_coeff == 3.0

def test_dry_depth():
    assert particle.dry_depth == 0.1

def test_gamma():
    assert particle.gamma == 0.05

def test_cell_type():
    assert particle.cell_type[1,1] == 0
    assert np.all(particle.cell_type[0,:] == [-1,-1,-1])
    assert np.all(particle.cell_type[-1,:] == [-1,-1,-1])
    assert np.all(particle.cell_type[:,0] == [-1,-1,-1])
    assert np.all(particle.cell_type[:,-1] == [-1,-1,-1])

def test_distances():
    assert particle.distances[1,1] == 1

def test_ivec():
    assert particle.ivec[1,1] == 0

def test_jvec():
    assert particle.jvec[1,1] == 0

def test_iwalk():
    assert particle.iwalk[1,1] == 0

def test_jwalk():
    assert particle.jwalk[1,1] == 0

def test_steep():
    assert particle.steepest_descent == False

def test_steep_true():
    params.diff_coeff = 'invalid'
    params.steepest_descent = True
    particle = particle_track.Particles(params)
    assert particle.steepest_descent == True

def test_steep_other():
    params.steepest_descent = 'other'
    particle = particle_track.Particles(params)
    assert particle.steepest_descent == False

def test_roi_grid():
    if getattr(params, 'roi_grid', None) is not None:
        assert particle.roi_grid is not None
    else:
        assert particle.roi_grid is None

def test_optional_particle_variables():
    for var in getattr(params, 'particle_variables', []):
        assert var in particle.particle_variables

    if getattr(params, 'roi_grid', None) is not None:
        assert 'roi_flag' in particle.particle_variables
    else:
        assert 'roi_flag' not in particle.particle_variables

    # optional_outputs should exactly match particle_variables
    assert set(particle.optional_outputs) == set(particle.particle_variables)

# testing of the run_iteration function
def test_start_pairs_X():
    particle = particle_track.Particles(params)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0][0] == seed_xloc[0]

def test_start_pairs_Y():
    particle = particle_track.Particles(params)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    all_walk_data = particle.run_iteration()
    assert all_walk_data['yinds'][0][0] == seed_yloc[0]

def test_exact_locations():
    num_ps = 2
    particle = particle_track.Particles(params)
    particle.generate_particles(num_ps, seed_xloc, seed_yloc, method='exact')
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0] == seed_xloc
    assert all_walk_data['yinds'][0] == seed_yloc
    assert all_walk_data['travel_times'][0] == [0]

def test_seed_time_int(caplog):
    num_ps = 2
    particle = particle_track.Particles(params)
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle.generate_particles(num_ps, seed_xloc, seed_yloc,
                                    seed_time=10, method='exact')
    assert any(
        record.levelno == logging.WARNING
        and "Particle seed time is nonzero" in record.getMessage()
        for record in caplog.records
    )
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0] == seed_xloc
    assert all_walk_data['yinds'][0] == seed_yloc
    assert all_walk_data['travel_times'][0] == [10]
    assert all_walk_data['travel_times'][1] == [10]

def test_seed_time_float(caplog):
    num_ps = 2
    particle = particle_track.Particles(params)
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle.generate_particles(num_ps, seed_xloc, seed_yloc,
                                    seed_time=11.1, method='exact')
    assert any(
        record.levelno == logging.WARNING
        and "Particle seed time is nonzero" in record.getMessage()
        for record in caplog.records
    )
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0] == seed_xloc
    assert all_walk_data['yinds'][0] == seed_yloc
    assert all_walk_data['travel_times'][0] == [11.1]
    assert all_walk_data['travel_times'][1] == [11.1]

def test_exact_locations_overflow():
    """Test where number of particles exceeds number of seed locations."""
    num_ps = 3
    particle = particle_track.Particles(params)
    particle.generate_particles(num_ps, [0, 1], [0, 1], method='exact')
    all_walk_data = particle.run_iteration()
    assert len(all_walk_data['xinds']) == num_ps
    assert len(all_walk_data['yinds']) == num_ps

def test_exact_locations_multidims():
    """Test case where seed locs given as multi-dimensional arrays."""
    num_ps = 5
    particle = particle_track.Particles(params)
    x_seed = np.array(((0, 0), (1, 1)))
    y_seed = np.array(((0, 0), (1, 1)))
    particle.generate_particles(num_ps, x_seed, y_seed, method='exact')
    all_walk_data = particle.run_iteration()
    assert len(all_walk_data['xinds']) == num_ps
    assert len(all_walk_data['yinds']) == num_ps

def test_no_explicit_generation():
    """Test reading of Np_tracer from self if some walk_data exists."""
    # create some walk data (would exist from a previous run)
    walk_data = {'xinds': [[1], [1]], 'yinds': [[1], [1]],
                 'travel_times': [[0.0], [0.0]]}
    # init particle class and stick walk data in it
    # init defines Np_tracer as 0, but it is 2 in walk_data
    particle = particle_track.Particles(params)
    particle.walk_data = walk_data
    # assert that Np_tracer is 0
    assert particle.Np_tracer == 0
    # run an iteration and see if Np_tracer is corrected
    particle.run_iteration()
    # assert that number of particles has been correctly identified
    assert particle.Np_tracer == 2

def test_travel_time():
    # Particle doesn't travel in the 3x3 space due to the 'sticky' edge
    # conditions so check that travel time is 0 and particle hasn't moved
    particle = particle_track.Particles(params)
    np.random.seed(0)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0][0] == 1
    assert all_walk_data['yinds'][0][0] == 1
    assert all_walk_data['travel_times'][0][0] == 0.0

def test_init_params():
    # test initialization of params class
    params = particle_track.modelParams()
    # make assertions
    assert params.dx is None
    assert params.depth is None
    assert params.stage is None
    assert params.qx is None
    assert params.qy is None
    assert params.u is None
    assert params.v is None


def test_previous_walk_data():
    # test of loading previously defined walk data
    particle = particle_track.Particles(params)
    np.random.seed(0)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    old_walk_data = particle.run_iteration()
    # try to do another walk - test just makes sure code doesn't break
    np.random.seed(0)
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0][0] == 1
    assert all_walk_data['yinds'][0][0] == 1
    assert all_walk_data['travel_times'][0][0] == 0.0


def test_generate_twice():
    # test ability to generate particles multiple times
    particle = particle_track.Particles(params)
    np.random.seed(0)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    old_walk_data = particle.run_iteration()
    # try to do another walk - test just makes sure code doesn't break
    np.random.seed(0)
    all_walk_data = particle.run_iteration()
    assert all_walk_data['xinds'][0][0] == 1
    assert all_walk_data['yinds'][0][0] == 1
    assert all_walk_data['travel_times'][0][0] == 0.0
    assert len(all_walk_data['xinds']) == 2


def test_use_walk_data():
    # use walk data without using the generator function
    particle1 = particle_track.Particles(params)
    np.random.seed(0)
    particle1.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    particle1.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    old_walk_data = particle1.run_iteration()
    # create new particles class and try to use that old walk data
    particle2 = particle_track.Particles(params)
    particle2.generate_particles(0, [], [], previous_walk_data=old_walk_data)
    all_walk_data = particle2.run_iteration()
    assert all_walk_data['xinds'][0][0] == 1
    assert all_walk_data['yinds'][0][0] == 1
    assert all_walk_data['travel_times'][0][0] == 0.0
    assert len(all_walk_data['xinds']) == 2


def test_manual_reset():
    # test resetting walk data
    particle = particle_track.Particles(params)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    assert particle.walk_data['xinds'][0][0] == 1
    particle.clear_walk_data()
    assert (particle.walk_data is None) is True

def test_run_iteration_optional_variables():
    # test that optional variables are being tracked in walk data
    particle = particle_track.Particles(params)
    particle.generate_particles(Np_tracer, seed_xloc, seed_yloc)
    all_walk_data = particle.run_iteration()
    # Check that all optional particle variables are in the returned walk_data
    for var in getattr(params, 'particle_variables', []):
        assert var in all_walk_data, f"{var} missing from walk_data"
    # roi_flag is included only if roi_grid exists
    if getattr(params, 'roi_grid', None) is not None:
        assert 'roi_flag' in all_walk_data, "roi_flag missing from walk_data"
    else:
        assert 'roi_flag' not in all_walk_data

class TestValueErrors:
    """
    Catching the ValueErrors and edge cases
    """

    def test_dx(self):
        params = particle_track.modelParams()
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_broken_depth(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = 'badstring'
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_missing_depth(self):
        params = particle_track.modelParams()
        params.dx = 1
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_depth_via_stage_topo(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.stage = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        particle = particle_track.Particles(params)
        # should work so make assertion
        assert np.all(particle.depth == params.stage-params.topography)

    def test_depth_via_stage_topo_broken(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.stage = 'badstring'
        params.topography = np.zeros((3,3))
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_stage_broken(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.stage = 'badstring'
        params.depth = np.ones((3,3))
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_stage_via_depth_topo(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.topography = np.zeros((3,3))
        params.depth = np.ones((3,3))
        params.depth[0,0] = 0
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        particle = particle_track.Particles(params)
        # should work so make assertion
        assert np.isnan(particle.stage[0,0]) == True
        assert particle.stage[1,1] == 1.

    def test_stage_via_depth_topo_broken(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.topography = 'badstring'
        params.depth = np.ones((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_missing_stage(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)
    
    def test_roi_grid_shape_mismatch(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        params.roi_grid = np.ones((4,4))  # Mismatched shape
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_roi_with_only_stage(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.stage = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        params.roi_grid = np.ones((3,3))
        particle = particle_track.Particles(params)
        # should work so make assertion
        assert particle.roi_grid.shape == (3,3)

    def test_rcm_model_uv(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        params.v = np.ones((3,3))
        params.model = 'DeltaRCM'
        particle = particle_track.Particles(params)
        # should work so make assertion
        assert np.all(particle.v == params.v)
        assert np.all(particle.u == params.u)
        assert np.all(particle.qx == params.u*params.depth)
        assert np.all(particle.qy == params.v*params.depth)

    def test_rcm_model_uv_broken(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        params.model = 'DeltaRCM'
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_model_q(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.qx = np.ones((3,3))
        params.qy = np.ones((3,3))
        particle = particle_track.Particles(params)
        # should work so make assertion
        assert np.all(particle.v == particle.qy*particle.depth/(particle.depth**2+1e-8))
        assert np.all(particle.u == particle.qx*particle.depth/(particle.depth**2+1e-8))
        assert np.all(particle.qx == -1*params.qy)
        assert np.all(particle.qy == params.qx)

    def test_model_uv_broken(self):
        params = particle_track.modelParams()
        params.dx = 1
        params.depth = np.ones((3,3))
        params.topography = np.zeros((3,3))
        params.u = np.ones((3,3))
        with pytest.raises(ValueError):
            particle = particle_track.Particles(params)

    def test_bad_Np_tracer(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(TypeError):
            particle.generate_particles('invalid', seed_xloc, seed_yloc)

    def test_bad_seed_xloc(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(TypeError):
            particle.generate_particles(1, 'invalid', seed_yloc)

    def test_bad_seed_yloc(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(TypeError):
            particle.generate_particles(1, seed_xloc, 'invalid')

    def test_invalid_seedtime(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(TypeError):
            particle.generate_particles(1, seed_xloc, seed_yloc,
                                        seed_time='invalid')

    def test_bad_method(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(ValueError):
            particle.generate_particles(1, seed_xloc, seed_yloc, method='bad')

    def test_invalid_method(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(TypeError):
            particle.generate_particles(1, seed_xloc, seed_yloc, method=5)

    def test_unstruct2grid_break(self):
        coords = [(10.5, 10.1),
                  (10.1, 15.1),
                  (15.2, 20.2)]
        quantity = [1, 2]
        cellsize = 1.0
        with pytest.raises(ValueError):
            particle_track.unstruct2grid(coords, quantity, cellsize,
                                         k_nearest_neighbors=1)

    def test_no_initialized_particles(self):
        particle = particle_track.Particles(goodparams)
        with pytest.raises(ValueError):
            particle.run_iteration()


def test_coord2ind():
    coords = [(10, 10),
              (10, 15),
              (15, 20)]
    raster_origin = (2, 5)
    raster_size = (25, 25)
    cellsize = 1.0
    inds = particle_track.coord2ind(coords, raster_origin,
                                    raster_size, cellsize)
    assert inds[0] == (20, 8)
    assert inds[1] == (15, 8)
    assert inds[2] == (10, 13)

def test_ind2coord():
    walk_data = dict()
    walk_data['xinds'] = [[20], [15, 10]]
    walk_data['yinds'] = [[8], [8, 13]]
    raster_origin = (2, 5)
    raster_size = (25, 25)
    cellsize = 1.0
    new_data = particle_track.ind2coord(walk_data, raster_origin,
                                        raster_size, cellsize)
    assert new_data['xcoord'][0] == [10.0]
    assert new_data['ycoord'][0] == [10.0]
    assert new_data['xcoord'][1] == [10.0, 15.0]
    assert new_data['ycoord'][1] == [15.0, 20.0]

def test_exposure_time_y():
    walk_data = dict()
    walk_data['xinds'] = [[1, 1, 1, 1, 1]]
    walk_data['yinds'] = [[1, 2, 3, 4, 5]]
    walk_data['travel_times'] = [[2, 4, 6, 8, 10]]
    roi = np.zeros((6, 6))
    roi[:, 2:4] = 1
    exp_times = particle_track.exposure_time(walk_data, roi)
    assert exp_times[0] == 4.0

def test_exposure_time_x():
    walk_data = dict()
    walk_data['yinds'] = [[1, 1, 1, 1, 1]]
    walk_data['xinds'] = [[1, 2, 3, 4, 5]]
    walk_data['travel_times'] = [[2, 4, 6, 8, 10]]
    roi = np.zeros((6, 6))
    roi[2:4, :] = 1
    exp_times = particle_track.exposure_time(walk_data, roi)
    assert exp_times[0] == 4.0

def test_exposure_reenter():
    walk_data = dict()
    walk_data['yinds'] = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    walk_data['xinds'] = [[1, 2, 3, 4, 5, 4, 3, 2, 3, 4, 5]]
    walk_data['travel_times'] = [[2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16]]
    roi = np.zeros((6, 6))
    roi[2:3, :] = 1
    exp_times = particle_track.exposure_time(walk_data, roi)
    assert exp_times[0] == 3.0

def test_nourishment_area():
    # seeded off the reserved (0, 0) cell: (0, 0) entries in walk_data are
    # the walker's parked-particle sentinel state, not real visits
    walk_data = dict()
    walk_data['yinds'] = [[1, 2, 3]]
    walk_data['xinds'] = [[1, 2, 3]]
    answer = np.diag([0., 1., 1., 1.])
    answer[answer == 0.] = np.nan
    vf = particle_track.nourishment_area(walk_data, (4,4),
                                         sigma=0, clip=100)
    assert ((vf == answer) | (np.isnan(vf) & np.isnan(answer))).all()

def test_nourishment_area_sigma():
    # same shifted-seed fixture; expected values pinned from
    # nourishment_area on this fixture (scipy gaussian smoothing)
    walk_data = dict()
    walk_data['yinds'] = [[1, 2, 3]]
    walk_data['xinds'] = [[1, 2, 3]]
    answer = np.array([[0.0141, 0.1038, 0.0143, 0.],
                       [0.1038, 0.7789, 0.2071, 0.0146],
                       [0.0143, 0.2071, 0.7930, 0.2216],
                       [0., 0.0146, 0.2216, 1.]])
    vf = particle_track.nourishment_area(walk_data, (4,4),
                                         sigma=0.5, clip=100)
    vf[np.isnan(vf)] = 0
    assert vf == pytest.approx(answer, rel=1e-2)

def test_nourishment_area_ignores_parked_sentinels():
    """(0, 0) entries in walk_data are the walker's parked-particle
    sentinel state (particle_stepper refuses to route a particle whose
    current index is (0, 0)), so they are not real cell visits and must
    not pollute the corner-cell visit count."""
    # a parked particle: its whole history is (0, 0) sentinel entries
    parked = {'xinds': [[0, 0, 0]], 'yinds': [[0, 0, 0]],
              'travel_times': [[1., 1., 1.]]}
    vf = particle_track.nourishment_area(parked, (3, 3), sigma=0, clip=100)
    assert np.isnan(vf).all()

    # a real walker plus a parked one: the corner stays uncounted and the
    # real visits normalize to 1
    mixed = {'xinds': [[0, 0, 0], [2, 2]],
             'yinds': [[0, 0, 0], [2, 2]],
             'travel_times': [[1., 1., 1.], [0., 1.]]}
    vf = particle_track.nourishment_area(mixed, (3, 3), sigma=0, clip=100)
    assert np.isnan(vf[0, 0])
    assert vf[2, 2] == pytest.approx(1.0)

def test_nourishment_time():
    walk_data = dict()
    walk_data['yinds'] = [[0, 1, 2, 3, 4, 5]]
    walk_data['xinds'] = [[0, 1, 2, 3, 4, 5]]
    walk_data['travel_times'] = [[2, 4, 6, 8, 15, 20]]
    answer = np.diag([0., 2., 2., 4.5, 6., 0.])
    answer[answer == 0.] = np.nan
    nt = particle_track.nourishment_time(walk_data, (6, 6),
                                         sigma=0, clip=100)
    assert ((nt == answer) | (np.isnan(nt) & np.isnan(answer))).all()

def test_nourishment_time_sigma():
    walk_data = dict()
    walk_data['yinds'] = [[0, 1, 2, 3, 4, 5]]
    walk_data['xinds'] = [[0, 1, 2, 3, 4, 5]]
    walk_data['travel_times'] = [[2, 4, 6, 8, 15, 20]]
    answer = np.array([[0.023, 0.168, 0.023, 0, 0, 0],
                       [0.168, 1.26, 0.335, 0.024, 0, 0],
                       [0.023, 0.335, 1.311, 0.544, 0.053, 0],
                       [0, 0.024, 0.544, 2.875, 0.879, 0.069],
                       [0, 0, 0.053, 0.879, 3.763, 0.504],
                       [0, 0, 0, 0.069, 0.504, 0.068]])
    nt = particle_track.nourishment_time(walk_data, (6, 6),
                                         sigma=0.5, clip=100)
    nt[np.isnan(nt)] = 0
    assert nt == pytest.approx(answer, abs=1e-2)

def test_unstruct2grid_k1():
    coords = [(10.5, 10.1),
              (10.1, 15.1),
              (15.2, 20.2)]
    quantity = [1, 2, 3]
    cellsize = 1.0
    interp_func, gridd = particle_track.unstruct2grid(coords, quantity,
                                                      cellsize,
                                                      k_nearest_neighbors=1)
    assert np.all(gridd == np.array([[3., 3., 3., 3., 3., 3., 3.],
                                     [2., 3., 3., 3., 3., 3., 3.],
                                     [2., 2., 3., 3., 3., 3., 3.],
                                     [2., 2., 2., 3., 3., 3., 3.],
                                     [2., 2., 2., 2., 3., 3., 3.],
                                     [2., 2., 2., 2., 2., 3., 3.],
                                     [2., 2., 2., 2., 2., 2., 3.],
                                     [2., 2., 2., 2., 2., 2., 2.],
                                     [2., 2., 2., 2., 2., 2., 2.],
                                     [1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1.],
                                     [1., 1., 1., 1., 1., 1., 1.]]))

def test_unstruct2grid_k3():
    coords = [(1.5, 1.1),
              (0.1, 2.1),
              (1.2, 2.2)]
    quantity = [1, 2, 3]
    cellsize = 1.0
    interp_func, gridd = particle_track.unstruct2grid(coords, quantity,
                                                      cellsize,
                                                      k_nearest_neighbors=3)
    assert pytest.approx(gridd) == np.array([[2.13911589, 2.26676874,
                                             2.1792037],
                                            [2., 2.68254467, 2.10026059],
                                            [1.96968263, 1.6122416,
                                             1.65818041]])

def test_unstruct2grid_bounds():
    coords = [(10.5, 10.1),
              (10.1, 15.1),
              (15.2, 20.2)]
    quantity = [1, 2, 3]
    cellsize = 1.0
    boundary = [[12, 12], [12, 19], [15, 19], [15, 12]]
    interp_func, gridd = particle_track.unstruct2grid(coords, quantity,
                                                      cellsize,
                                                      k_nearest_neighbors=1,
                                                      boundary=boundary,
                                                      crop=True)
    assert np.all(gridd == np.array([[2., 3., 3.],
                                     [2., 2., 3.],
                                     [2., 2., 2.],
                                     [2., 2., 2.],
                                     [2., 2., 2.],
                                     [2., 2., 2.],
                                     [1., 1., 1.]]))
    
def test_flux_proportional_seeding():
    num_total_particles = 1000
    num_seeding_steps = 10
    q = np.ones((10,1))*200
    q[0] = 400  # Increase flux in the first cell
    q[1] = 100  # Decrease flux in the second cell
    q[9] = 100  # Decrease flux in the last cell (q sums to 1000)
    flux_normalized_particle_counts = particle_track.flux_proportional_seeding(q, num_total_particles, num_seeding_steps)
    assert len(flux_normalized_particle_counts) == num_seeding_steps
    assert sum(flux_normalized_particle_counts) == num_total_particles
    assert flux_normalized_particle_counts[0] > flux_normalized_particle_counts[1]
    assert flux_normalized_particle_counts[1] == flux_normalized_particle_counts[9]


def test_parallel_routing():
    """Test the parallel routing functionality of the particle tracking model with optional outputs."""
    # Set up parameters for the particle tracking
    params = particle_track.modelParams()
    params.dx = 1
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    params.particle_variables = ['depth', 'qx']  # Optional variables to track
    params.seed_xloc = [1, 2]
    params.seed_yloc = [1, 2]
    params.Np_tracer = 2  # Number of particles to generate
    params.model = 'DeltaRCM'

    # Create the particle object
    particle = particle_track.Particles(params)

    # Run parallel routing with a specified number of cores
    num_cores = 2  # Adjust based on your machine's capabilities
    num_iter = 5   # Number of iterations to run
    par_result = parallel_routing.parallel_routing(particle, num_iter, params.Np_tracer, params.seed_xloc, params.seed_yloc, num_cores)

    assert len(par_result['xinds']) == params.Np_tracer * num_cores
    assert 'depth' in par_result
    assert 'qx' in par_result


def test_negative_diff_coeff_logs_warning(caplog):
    # misconfiguration notices are logger warnings, not warnings.warn
    params.diff_coeff = -1.0
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle_track.Particles(params)
    assert any(
        record.levelno == logging.WARNING
        and "Specified diffusion coefficient is negative" in record.getMessage()
        for record in caplog.records
    )

    caplog.clear()
    params.diff_coeff = 2.5
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle_track.Particles(params)
    assert any(
        record.levelno == logging.WARNING
        and "Diffusion behaves non-physically when" in record.getMessage()
        for record in caplog.records
    )

def test_seed_time_logs_warning(caplog):
    params.diff_coeff = 0.0
    pt = particle_track.Particles(params)
    with caplog.at_level(logging.WARNING, logger="dorado"):
        pt.generate_particles(1, [1], [1], seed_time=1)
    assert any(
        record.levelno == logging.WARNING
        and "Particle seed time is nonzero" in record.getMessage()
        for record in caplog.records
    )
# ---------------------------------------------------------------------------
# per-particle size/type attributes
# ---------------------------------------------------------------------------

def attribute_params(shape=(20, 15)):
    """Uniform +y-flow domain used by the per-particle attribute tests."""
    params = particle_track.modelParams()
    params.dx = 1
    params.depth = np.ones(shape)
    params.stage = np.ones(shape)
    params.qx = np.zeros(shape)
    params.qy = np.ones(shape)
    params.theta = 1
    params.model = 'DeltaRCM'
    return params


def seeded_attribute_run(nparticles=3, niter=6, **gen_kwargs):
    """Run a seeded generate_particles + run_iteration session."""
    np.random.seed(42)
    particle = particle_track.Particles(attribute_params(),
                                        backend='python')
    particle.generate_particles(nparticles, [2, 3], [2, 3], method='exact',
                                **gen_kwargs)
    walk_data = None
    for _ in range(niter):
        walk_data = particle.run_iteration()
    return walk_data


def test_dorado_backend_env_overrides_constructor(monkeypatch):
    """DORADO_BACKEND forces the backend regardless of the constructor."""
    params = attribute_params()
    monkeypatch.setenv('DORADO_BACKEND', 'python')
    forced = particle_track.Particles(params)
    assert forced.backend == 'python'
    # even an explicit constructor choice cannot beat the env override
    assert particle_track.Particles(params, backend='rust').backend == \
        'python'
    monkeypatch.setenv('DORADO_BACKEND', 'rust')
    assert particle_track.Particles(params, backend='python').backend == \
        'rust'


def test_dorado_backend_python_forces_reference_path(monkeypatch):
    """DORADO_BACKEND=python routes through the legacy walker verbatim."""
    monkeypatch.setenv('DORADO_BACKEND', 'python')
    # same seed as seeded_attribute_run so the two runs draw from the
    # same stream position and must land on identical trajectories
    np.random.seed(42)
    particle = particle_track.Particles(attribute_params())
    particle.generate_particles(3, [2, 3], [2, 3], method='exact')
    for _ in range(6):
        walk_data = particle.run_iteration()
    # identical to the seeded_attribute_run python reference (same seed
    # stream position, same generate call, same iteration count)
    reference = seeded_attribute_run()
    assert walk_data['xinds'] == reference['xinds']
    assert walk_data['travel_times'] == reference['travel_times']


def test_size_length_mismatch_raises():
    """A particle_size list shorter than Np_tracer must raise a ValueError."""
    with pytest.raises(ValueError, match='does not match the expected length'):
        seeded_attribute_run(particle_size=[1.0, 2.0])


def test_type_length_mismatch_raises():
    """A particle_type list shorter than Np_tracer must raise a ValueError."""
    with pytest.raises(ValueError, match='does not match the expected length'):
        seeded_attribute_run(particle_type=['sand', 'mud'])


def test_unknown_type_names_known_kinds():
    """An unknown kind must raise and name the known kinds in the message."""
    with pytest.raises(ValueError, match='Known kinds are:'):
        seeded_attribute_run(particle_type='boulder')


def test_nonstr_type_entries_raise():
    """Non-string entries in a particle_type list must raise a ValueError."""
    with pytest.raises(ValueError, match='must be strings'):
        seeded_attribute_run(2, particle_type=['sand', 5])


def test_attribute_histories_persist_across_iterations():
    """Sizes/types stay constant per particle; deposited starts at 0."""
    walk_data = seeded_attribute_run(
        particle_size=[0.5, 2.0, 1.0],
        particle_type=['plankton', None, None])
    sizes = walk_data['particle_size']
    types = walk_data['particle_type']
    deposited = walk_data['deposited']
    assert len(sizes) == 3
    assert len(types) == 3
    assert len(deposited) == 3
    for i in range(3):
        # one attribute entry appended per move, aligned with the walk
        assert len(sizes[i]) == len(walk_data['xinds'][i])
        assert set(sizes[i]) == {sizes[i][0]}
        assert set(types[i]) == {types[i][0]}
    assert sizes[0][0] == 0.5
    assert sizes[1][0] == 2.0
    assert sizes[2][0] == 1.0
    assert types[0][0] == 'plankton'
    assert types[1][0] is None
    assert types[2][0] is None
    # no deposition probability in play: the flag stays 0 everywhere
    assert all(flag == 0 for particle in deposited for flag in particle)


def test_mixed_batches_are_padded_with_none():
    """Batches without attributes get None padding, lengths stay aligned."""
    np.random.seed(0)
    particle = particle_track.Particles(attribute_params())
    particle.generate_particles(2, [2, 3], [2, 3], method='exact',
                                particle_size=[1.0, 2.0])
    particle.generate_particles(1, [2], [2], method='exact')
    walk_data = particle.run_iteration()
    assert len(walk_data['particle_size']) == 3
    assert walk_data['particle_size'][0][0] == 1.0
    assert walk_data['particle_size'][1][0] == 2.0
    assert all(value is None for value in walk_data['particle_size'][2])
    for i in range(3):
        assert len(walk_data['particle_size'][i]) == len(walk_data['xinds'][i])
        assert len(walk_data['deposited'][i]) == len(walk_data['xinds'][i])


def test_larger_size_slows_particles_same_seed():
    """With the same seed, size=2 moves identically but takes longer."""
    legacy = seeded_attribute_run()
    sized = seeded_attribute_run(particle_size=2.0)
    # identical draw sequence -> identical trajectories
    assert sized['xinds'] == legacy['xinds']
    assert sized['yinds'] == legacy['yinds']
    for new_times, old_times in zip(sized['travel_times'],
                                    legacy['travel_times']):
        assert len(new_times) == len(old_times)
        # every actual move (past the 0 travel-time seed entry) is slower
        assert all(t2 > t1 for t2, t1 in zip(new_times[1:], old_times[1:]))


def test_smaller_size_diffuses_more():
    """Smaller particles get a larger diffusion multiplier (1/size).

    With the same seed all size runs follow identical paths (directions
    do not depend on size), so the size-normalized per-particle total is
    n_p(s) = B_p + E_p/s where B_p is the deterministic path cost and
    E_p the size-invariant accumulated diffusion noise. The gap between
    adjacent sizes therefore isolates the diffusion term: D12 = E_p and
    D23 = E_p/2, so their variances must stand in a 4:1 ratio -- direct
    evidence that halving the size doubles the relative diffusion.
    (Raw travel-time spread cannot show this: it is dominated by the
    size-squared scaling of the deterministic path cost.)
    """
    runs = {size: seeded_attribute_run(20, 8, particle_size=size)
            for size in (0.5, 1.0, 2.0)}
    assert (runs[0.5]['xinds'] == runs[1.0]['xinds'] == runs[2.0]['xinds'])
    normalized = {size: [sum(times)/size for times in runs[size]['travel_times']]
                  for size in (0.5, 1.0, 2.0)}
    d12 = [a - b for a, b in zip(normalized[0.5], normalized[1.0])]
    d23 = [a - b for a, b in zip(normalized[1.0], normalized[2.0])]
    assert np.var(d12) > np.var(d23)


@pytest.fixture()
def full_deposition_kind():
    """Register a test-only kind with deposition probability 1.0."""
    import dorado.lagrangian_walker as lw
    lw.KIND_PRESETS['test_settler'] = {'travel_time_multiplier': 1.0,
                                       'diffusion_multiplier': 1.0,
                                       'deposition_probability': 1.0}
    yield 'test_settler'
    del lw.KIND_PRESETS['test_settler']


def test_deposition_stops_all_particles_first_step(full_deposition_kind):
    """A deposition probability of 1 freezes particles on their first move."""
    kind = full_deposition_kind
    np.random.seed(7)
    particle = particle_track.Particles(attribute_params())
    particle.generate_particles(3, [2, 3], [2, 3], method='exact',
                                particle_type=kind)
    start_positions = [particle.walk_data['xinds'][i][0] for i in range(3)]
    walk_data = particle.run_iteration()
    # no particle moved away from its seed location
    assert [walk_data['xinds'][i][-1] for i in range(3)] == start_positions
    # every particle is now deposited
    assert all(walk_data['deposited'][i][-1] == 1 for i in range(3))
    # deposition is permanent: a second iteration changes nothing
    second = particle.run_iteration()
    assert second['xinds'] == walk_data['xinds']
    assert all(second['deposited'][i][-1] == 1 for i in range(3))


def test_no_attrs_run_keeps_legacy_walk_data():
    """Runs without attributes gain no new keys and match the legacy walk."""
    walk_data = seeded_attribute_run()
    # no attribute keys appear on a run that passed no attributes; the
    # only additions are the stranding columns tracked by the park
    # default (a documented behavior change, docs/source/background)
    assert sorted(walk_data.keys()) == ['stranded', 'stranded_time',
                                        'travel_times', 'xinds', 'yinds']
    # and the trajectories match the pre-change seeded result exactly
    assert walk_data['xinds'] == [[2, 2, 1],
                                  [3, 4, 5, 4, 4, 4, 3],
                                  [2, 2, 2, 1]]
    assert walk_data['yinds'] == [[2, 3, 4],
                                  [3, 4, 5, 6, 7, 8, 9],
                                  [2, 3, 4, 5]]
    assert walk_data['travel_times'][0][1] == pytest.approx(0.9802683129632757)


def test_size_one_matches_legacy_exactly():
    """A size of exactly 1.0 reduces to the legacy formula bit-for-bit."""
    legacy = seeded_attribute_run()
    sized = seeded_attribute_run(particle_size=1.0)
    assert legacy['xinds'] == sized['xinds']
    assert legacy['yinds'] == sized['yinds']
    assert legacy['travel_times'] == sized['travel_times']


def test_routines_handle_attribute_walk_data():
    """get_state and get_time_state work on walk_data with new keys."""
    from dorado.routines import get_state, get_time_state
    walk_data = seeded_attribute_run(particle_size=1.0)
    state_out = get_state(walk_data)
    assert len(state_out) == 4
    assert state_out[3]['particle_size'] == [1.0, 1.0, 1.0]
    time_out = get_time_state(walk_data, 1.0)
    assert len(time_out) == 4
    assert time_out[3]['deposited'] == [0, 0, 0]


def test_parallel_routing_with_attributed_particles():
    """Smoke test: parallel_routing runs with attributed particles."""
    params = attribute_params((5, 5))
    params.seed_xloc = [1, 2]
    params.seed_yloc = [1, 2]
    params.Np_tracer = 2
    particle = particle_track.Particles(params)
    par_result = parallel_routing.parallel_routing(particle, 3, 2, [1, 2],
                                                   [1, 2], 2,
                                                   particle_size=[0.5, 2.0],
                                                   particle_type=['sand',
                                                                  'plankton'])
    assert len(par_result['xinds']) == 4
    assert 'particle_size' in par_result
    assert 'deposited' in par_result


def test_verbose_run_iteration_warning(caplog):
    """A particle that exceeds the iteration limit before reaching the
    target time must trip the warning, whatever its position in the
    per-particle iteration order."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((60, 60))
    params.stage = np.ones((60, 60))
    params.qx = 1e9 * np.ones((60, 60))  # fast flow: tiny per-step times
    # moderate-flow column: velocity ~5, so particle 1 takes ~0.2 s per
    # step and reaches the 0.3 s target cleanly within max_iter
    params.qx[:, 35] = 5.0
    params.qy = np.zeros((60, 60))
    particle = particle_track.Particles(params)
    np.random.seed(0)
    # 'exact' pairs the x and y seed lists positionally ('random' would
    # pair them independently): particle 0 travels in the fast flow and
    # cannot reach the target time within max_iter; particle 1 sits in
    # the moderate column and finishes cleanly after it -- pre-fix, its
    # per-particle re-initialization of _iter_particles erased particle
    # 0's capped entry and the warning never fired
    particle.generate_particles(2, [2, 2], [30, 35], method='exact')
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle.run_iteration(target_time=0.3, max_iter=20)
    assert any(
        record.levelno == logging.WARNING
        and "exceeded iteration limit" in record.getMessage()
        for record in caplog.records
    )


# --- regression tests for confirmed routing bugs ---

def test_seed_on_row0_stays_in_domain():
    """Particles seeded on row 0 must not wrap to negative indices."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    particle = particle_track.Particles(params)
    np.random.seed(0)
    particle.generate_particles(1, [0], [2])
    walk_data = particle.run_iteration()
    assert min(walk_data['xinds'][0]) >= 0
    assert max(walk_data['xinds'][0]) < params.depth.shape[0]


def test_generate_particles_rejects_out_of_bounds_seeds():
    """Out-of-domain seed indices must raise instead of wrapping."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    particle = particle_track.Particles(params)
    with pytest.raises(ValueError, match='[Ss]eed location'):
        particle.generate_particles(1, [7], [2])
    with pytest.raises(ValueError, match='[Ss]eed location'):
        particle.generate_particles(1, [2], [-1])


def test_params_arrays_not_mutated():
    """Particles() must not modify the caller's params arrays."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.depth[0, 0] = np.nan
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    depth_snapshot = params.depth.copy()
    stage_snapshot = params.stage.copy()
    qx_snapshot = params.qx.copy()
    qy_snapshot = params.qy.copy()

    particle = particle_track.Particles(params)

    assert np.array_equal(params.depth, depth_snapshot, equal_nan=True)
    assert np.array_equal(params.stage, stage_snapshot, equal_nan=True)
    assert np.array_equal(params.qx, qx_snapshot)
    assert np.array_equal(params.qy, qy_snapshot)
    # the particle's arrays must be independent copies, not aliases
    assert not np.shares_memory(params.depth, particle.depth)
    assert not np.shares_memory(params.stage, particle.stage)
    assert not np.shares_memory(params.qx, particle.qx)
    assert not np.shares_memory(params.qy, particle.qy)
    # NaN depth is still treated as dry inside the particle's own copy
    assert particle.depth[0, 0] == 0


def test_params_uv_and_lists_not_mutated():
    """u/v params and the particle_variables list must not be modified."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.depth[0, 0] = 0.0  # dry cell: stage gains NaN only in the copy
    params.stage = np.ones((5, 5))
    params.u = np.ones((5, 5))
    params.u[0, 1] = np.nan
    params.v = np.ones((5, 5))
    params.particle_variables = ['depth']
    params.roi_grid = np.ones((5, 5))
    u_snapshot = params.u.copy()
    v_snapshot = params.v.copy()
    stage_snapshot = params.stage.copy()
    pv_snapshot = list(params.particle_variables)

    particle = particle_track.Particles(params)

    assert np.array_equal(params.u, u_snapshot, equal_nan=True)
    assert np.array_equal(params.v, v_snapshot)
    assert np.array_equal(params.stage, stage_snapshot, equal_nan=True)
    assert params.particle_variables == pv_snapshot
    # roi_flag is appended to the particle's own list, not the caller's
    assert 'roi_flag' in particle.particle_variables


def test_gamma_out_of_range_raises():
    """gamma outside [0, 1] must raise ValueError instead of corrupting picks.

    Review probe (F3): gamma = 1.5 produced negative weights, a
    non-monotonic cumulative distribution, and 43.5% of picks landing
    on a single invalid cell with no warning of any kind.
    """
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    # sloped stage so the routing weights are actually populated
    params.stage = np.tile(np.linspace(1.0, 0.9, 5), (5, 1))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))

    params.gamma = 1.5
    with pytest.raises(ValueError, match='[Gg]amma'):
        particle_track.Particles(params)

    params.gamma = -0.2
    with pytest.raises(ValueError, match='[Gg]amma'):
        particle_track.Particles(params)


def test_cell_type_none_falls_back_to_estimation():
    """Optional parameters set explicitly to None must use the fallback."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.ones((5, 5))
    params.cell_type = None
    particle = particle_track.Particles(params)
    # cell types were estimated from depth, including the -1 border ring
    assert particle.cell_type.shape == (5, 5)
    assert -1 in particle.cell_type


def test_velocity_floor_preserves_sign():
    """The tiny-velocity floor must not flip a negative velocity."""
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((5, 5))
    params.stage = np.ones((5, 5))
    params.qx = np.zeros((5, 5))
    params.qy = np.zeros((5, 5))
    # tiny westward + northward discharge at the center cell
    params.qx[2, 2] = -1e-9
    params.qy[2, 2] = 1e-3
    particle = particle_track.Particles(params)
    # the floored u must stay negative so the velocity angle keeps
    # pointing with the flow instead of rotating ~126 degrees
    assert particle.u[2, 2] < 0
    assert particle.velocity_angle[2, 2] > 0


def test_stepper_freezes_out_of_domain_step():
    """A step off an all-water domain must freeze, not record a wrap.

    Review probe (F1): a particle sitting on a border cell of a domain
    with no -1 edge markers has an all-zero weight row, so random_pick
    deterministically returns slot 0 (NW) and steps to a negative row;
    nothing bounds-checked the result.
    """
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((6, 6))
    params.stage = np.ones((6, 6))
    params.qx = np.zeros((6, 6))
    params.qy = np.ones((6, 6))
    params.cell_type = np.zeros((6, 6))  # water everywhere: no -1 edges
    particle = particle_track.Particles(params)
    new_inds, travel_times, _ = lagrangian_walker.particle_stepper(
        particle, [[0, 2]], [0.], {})
    assert new_inds[0] == [0, 2]
    assert 0 <= new_inds[0][0] < 6
    assert 0 <= new_inds[0][1] < 6


def test_water_bordered_domain_no_escape():
    """Particles must never walk off a water-bordered domain.

    Regression for review finding F1 (the 498/500 seeded-escape probe):
    on a 6x6 all-water domain with stage sloped downhill-outward,
    unpatched code let particles march through the rim into negative
    indices, where every downstream lookup wrapped silently and
    corrupted the spatial statistics.
    """
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((6, 6))
    params.qx = np.zeros((6, 6))
    params.qy = np.ones((6, 6))
    params.cell_type = np.zeros((6, 6))  # all water: no -1 edge markers
    params.gamma = 1.0  # pure stage-gradient walk, drifts outward
    ii, jj = np.meshgrid(np.arange(6), np.arange(6), indexing='ij')
    # high center, falling off toward every edge: downhill-outward
    params.stage = 2.0 - 0.25 * (np.abs(ii - 2.5) + np.abs(jj - 2.5))
    particle = particle_track.Particles(params)
    np.random.seed(0)
    particle.generate_particles(500, [3] * 500, [3] * 500)
    walk = {}
    for _ in range(8):
        walk = particle.run_iteration()
    for path in walk['xinds'] + walk['yinds']:
        assert min(path) >= 0
        assert max(path) < 6


def test_over_limit_warning_message(caplog):
    """The over-limit warning fires once with a correct, readable count.

    Review probe (F10): the warning read '1Particles' (missing space)
    and the count could miss earlier overflowers entirely.
    """
    params = particle_track.modelParams()
    params.dx = 1.
    params.model = 'DeltaRCM'
    params.depth = np.ones((60, 60))
    params.stage = np.ones((60, 60))
    params.qx = np.zeros((60, 60))
    params.qy = np.ones((60, 60))
    # gamma = 0: the walk follows the discharge field only, so the
    # particle always steps E/NE/SE (stay and uphill slots carry zero
    # weight) and drifts east from the center. The rim sits 29 cells
    # away while max_iter caps the run at 26 steps: the particle can
    # neither reach the target time nor park, so it must overflow.
    params.gamma = 0.0
    particle = particle_track.Particles(params)
    np.random.seed(3)
    particle.generate_particles(1, [30], [30])
    with caplog.at_level(logging.WARNING, logger="dorado"):
        particle.run_iteration(target_time=1e9, max_iter=25)
    over_limit = [r for r in caplog.records
                  if "particles exceeded" in r.getMessage()]
    assert len(over_limit) == 1
    assert over_limit[0].getMessage().startswith("1 particles exceeded")



# ---- Exception diagnostics -------------------------------------------
# Chained re-raises: the original cause must survive on __cause__, and the
# exception type/message must stay what they were before chaining.

def test_depth_re_raise_chains_cause():
    # a depth array without .copy() triggers the AttributeError that the
    # except block converts to ValueError; the cause must survive
    bad = copy.deepcopy(goodparams)
    bad.depth = 5  # int has no .copy() -- re-raise path, not a None guard
    with pytest.raises(ValueError) as excinfo:
        particle_track.Particles(bad)
    assert excinfo.value.__cause__ is not None
    assert "Water depth array incorrectly defined." in str(excinfo.value)


def test_gen_input_check_chains_cause():
    # Np_tracer that int() cannot coerce trips the chained TypeError
    with pytest.raises(TypeError) as excinfo:
        particle_track.gen_input_check(object(), [1], [1], 0.0, 'random')
    assert excinfo.value.__cause__ is not None
    assert "Np_tracer input type was not int." in str(excinfo.value)


def test_stranding_re_raise_chains_cause():
    # narrow (TypeError, ValueError) blocks chain too
    bad = copy.deepcopy(goodparams)
    bad.rest_time = 'not-a-number'
    with pytest.raises(ValueError) as excinfo:
        particle_track.check_stranding_params(bad)
    assert excinfo.value.__cause__ is not None


# Default-parameter substitution: omitting each D8 geometry parameter
# must emit one info record naming the parameter

def test_default_parameter_info_records(caplog):
    minimal = pobj()
    minimal.dx = 1
    minimal.depth = np.ones((3, 3))
    minimal.stage = np.ones((3, 3))
    minimal.qx = np.zeros((3, 3))
    minimal.qy = np.ones((3, 3))
    minimal.model = 'DeltaRCM'
    with caplog.at_level(logging.INFO, logger="dorado"):
        particle_track.Particles(minimal)
    text = caplog.text
    for name in ['distances', 'ivec', 'jvec', 'iwalk', 'jwalk', 'angles']:
        assert "params.%s not provided" % name in text

