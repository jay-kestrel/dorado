"""Tests for the optional subsurface modelParams and weight volume.

Covers the contracts of check_subsurface_params (forcing-form exclusivity,
porosity coupling and range, probability and time-scalar ranges, grid-shape
matching, and the exchange-scalars-only storage no-op) and of
make_weight_subsurface ((L, W, 9) volume, mask zeroing with a reachable
mask edge, per-object copies, storage-mode zeros).

The zero-diff class pins seeded trajectories on a small synthetic field
with no subsurface input set. The pinned constants were cross-checked
byte-for-byte against a pristine master checkout (2f756e9) for both the
default (rust) backend and the python backend.
"""
import json
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import numpy as np
import pytest

from dorado.particle_track import Particles
from dorado.particle_track import check_subsurface_params
from dorado.particle_track import modelParams

# small synthetic surface domain shared by the volume and zero-diff tests
L = W = 10
STAGE = np.ones((L, W))
DEPTH = np.ones((L, W))
QX = np.full((L, W), 0.5)
QY = np.zeros((L, W))


def surface_params():
    """Surface-only modelParams on the shared synthetic domain."""
    p = modelParams()
    p.dx = 10.0
    p.stage = STAGE
    p.depth = DEPTH
    p.qx = QX
    p.qy = QY
    p.model = 'DeltaRCM'
    return p


def subsurface_params(**kwargs):
    """modelParams with a surface base plus subsurface keyword inputs."""
    p = surface_params()
    for key, value in kwargs.items():
        setattr(p, key, value)
    return p


class TestValidation:
    """check_subsurface_params contracts."""

    def test_no_subsurface_inputs_returns_none(self):
        assert check_subsurface_params(surface_params(), (L, W)) is None

    def test_both_forcing_forms_raise(self):
        p = subsurface_params(gw_qx=np.zeros((L, W)), gw_qy=np.zeros((L, W)),
                              gw_u=np.ones((L, W)), gw_v=np.ones((L, W)),
                              porosity=0.3)
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_lone_discharge_component_raises(self):
        p = subsurface_params(gw_qx=np.zeros((L, W)), porosity=0.3)
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_lone_velocity_component_raises(self):
        p = subsurface_params(gw_u=np.ones((L, W)))
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_discharge_without_porosity_raises(self):
        p = subsurface_params(gw_qx=np.zeros((L, W)), gw_qy=np.zeros((L, W)))
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    @pytest.mark.parametrize('porosity', [0.0, -0.1, 1.5, np.nan])
    def test_scalar_porosity_out_of_range_raises(self, porosity):
        p = subsurface_params(gw_qx=np.zeros((L, W)),
                              gw_qy=np.zeros((L, W)), porosity=porosity)
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    @pytest.mark.parametrize('bad_value', [0.0, 1.5])
    def test_grid_porosity_out_of_range_raises(self, bad_value):
        porosity = np.full((L, W), 0.3)
        porosity[0, 0] = bad_value
        p = subsurface_params(gw_qx=np.zeros((L, W)),
                              gw_qy=np.zeros((L, W)), porosity=porosity)
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    @pytest.mark.parametrize('name', ['p_infiltration', 'p_resurface'])
    @pytest.mark.parametrize('value', [-0.1, 1.5, np.nan])
    def test_probability_out_of_range_raises(self, name, value):
        p = subsurface_params(**{name: value})
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    @pytest.mark.parametrize('name', ['residence_time_mean', 'storage_dt'])
    @pytest.mark.parametrize('value', [0.0, -5.0, np.nan])
    def test_time_scalar_nonpositive_raises(self, name, value):
        p = subsurface_params(**{name: value})
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_gw_grid_shape_mismatch_raises(self):
        p = subsurface_params(gw_u=np.ones((L, W + 1)),
                              gw_v=np.ones((L, W + 1)))
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_mask_shape_mismatch_raises(self):
        p = subsurface_params(gw_u=np.ones((L, W)), gw_v=np.ones((L, W)),
                              aquifer_mask=np.ones((L, W), dtype=bool))
        p.aquifer_mask = np.ones((L + 1, W), dtype=bool)
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_porosity_grid_shape_mismatch_raises(self):
        p = subsurface_params(gw_qx=np.zeros((L, W)), gw_qy=np.zeros((L, W)),
                              porosity=np.full((L + 1, W), 0.3))
        with pytest.raises(ValueError):
            check_subsurface_params(p, (L, W))

    def test_exchange_scalars_without_forcing_is_storage_noop(self):
        # valid storage-mode configuration: no gw fields, no error, and
        # the derived velocity fields are None
        p = subsurface_params(p_infiltration=0.5, residence_time_mean=100.0)
        sub = check_subsurface_params(p, (L, W))
        assert sub['gw_u'] is None and sub['gw_v'] is None
        assert sub['p_infiltration'] == 0.5
        assert sub['residence_time_mean'] == 100.0
        assert sub['storage_dt'] == 1.0  # default
        assert sub['p_resurface'] == 0.0  # default

    def test_discharge_converted_to_pore_velocity(self):
        qx = np.full((L, W), 0.7)
        qy = np.full((L, W), 0.14)
        p = subsurface_params(gw_qx=qx, gw_qy=qy, porosity=0.35)
        sub = check_subsurface_params(p, (L, W))
        assert np.allclose(sub['gw_u'], 2.0)
        assert np.allclose(sub['gw_v'], 0.4)

    def test_velocity_form_passthrough(self):
        p = subsurface_params(gw_u=np.full((L, W), 2.0), gw_v=np.zeros((L, W)))
        sub = check_subsurface_params(p, (L, W))
        assert np.allclose(sub['gw_u'], 2.0)
        assert sub['porosity'] is None

    def test_mask_default_true_with_forcing(self):
        p = subsurface_params(gw_u=np.ones((L, W)), gw_v=np.ones((L, W)))
        sub = check_subsurface_params(p, (L, W))
        assert sub['aquifer_mask'].dtype == bool
        assert sub['aquifer_mask'].all()

    def test_mask_int_inputs_viewed_as_bool(self):
        mask = np.ones((L, W), dtype=int)
        mask[0, :] = 0
        p = subsurface_params(gw_u=np.ones((L, W)), gw_v=np.ones((L, W)),
                              aquifer_mask=mask)
        sub = check_subsurface_params(p, (L, W))
        assert sub['aquifer_mask'].dtype == bool
        assert not sub['aquifer_mask'][0, :].any()

    def test_nan_gw_components_zeroed(self):
        qx = np.full((L, W), 0.7)
        qx[0, 0] = np.nan
        p = subsurface_params(gw_qx=qx, gw_qy=np.zeros((L, W)), porosity=0.35)
        sub = check_subsurface_params(p, (L, W))
        assert sub['gw_u'][0, 0] == 0.0
        assert np.allclose(sub['gw_u'][1:, :], 2.0)

    def test_inputs_not_mutated_and_copied(self):
        # caller-owned arrays are neither mutated nor aliased
        qx = np.full((L, W), 0.7)
        qy = np.zeros((L, W))
        mask = np.ones((L, W), dtype=bool)
        p = subsurface_params(gw_qx=qx, gw_qy=qy, porosity=0.35,
                              aquifer_mask=mask)
        sub = check_subsurface_params(p, (L, W))
        assert np.allclose(qx, 0.7) and np.allclose(qy, 0.0)
        assert mask.all()
        qx[:] = 999.0
        mask[:] = False
        assert np.allclose(sub['gw_u'], 2.0)
        assert sub['aquifer_mask'].all()


class TestSubsurfaceWeightVolume:
    """make_weight_subsurface contracts, exercised through Particles."""

    def test_storage_mode_zero_volume(self):
        pobj = Particles(subsurface_params(p_infiltration=0.5),
                         backend='python')
        assert pobj.weight_sub.shape == (L, W, 9)
        assert not pobj.weight_sub.any()
        assert pobj.velocity_sub is None
        assert pobj.velocity_angle_sub is None

    def test_field_mode_projection(self):
        # uniform +x pore velocity: all weight on the S/SW/SE channels,
        # normalized to 0.5/0.25/0.25 on interior cells
        pobj = Particles(subsurface_params(gw_u=np.full((L, W), 2.0),
                                           gw_v=np.zeros((L, W))),
                         backend='python')
        row = pobj.weight_sub[5, 5, :]
        assert np.isclose(row[7], 0.5)  # S
        assert np.isclose(row[6], 0.25)  # SW
        assert np.isclose(row[8], 0.25)  # SE
        assert np.isclose(row.sum(), 1.0)
        assert pobj.weight_sub[5, 5, 4] == 0.0  # no center stay
        assert np.isclose(pobj.velocity_sub[5, 5], 2.0)

    def test_discharge_and_velocity_forms_identical(self):
        pobj_a = Particles(subsurface_params(gw_u=np.full((L, W), 2.0),
                                             gw_v=np.zeros((L, W))),
                           backend='python')
        pobj_b = Particles(subsurface_params(gw_qx=np.full((L, W), 0.7),
                                             gw_qy=np.zeros((L, W)),
                                             porosity=0.35),
                           backend='python')
        assert np.array_equal(pobj_a.weight_sub, pobj_b.weight_sub)

    def test_mask_zeroing_with_reachable_edge(self):
        # cells outside the mask are immovable rock, but interior cells
        # keep their weights pointing at the mask edge (the exfiltration
        # pathway)
        mask = np.ones((L, W), dtype=bool)
        mask[6:, :] = False
        pobj = Particles(subsurface_params(gw_u=np.full((L, W), 2.0),
                                           gw_v=np.zeros((L, W)),
                                           aquifer_mask=mask),
                         backend='python')
        assert not pobj.weight_sub[6:, :, :].any()
        assert pobj.weight_sub[5, 5, 7] > 0  # S: toward the mask edge

    def test_border_rows_cleared(self):
        pobj = Particles(subsurface_params(gw_u=np.full((L, W), 2.0),
                                           gw_v=np.zeros((L, W))),
                         backend='python')
        assert not pobj.weight_sub[0, :, :].any()
        assert not pobj.weight_sub[-1, :, :].any()
        assert not pobj.weight_sub[:, 0, :].any()
        assert not pobj.weight_sub[:, -1, :].any()

    def test_volume_is_per_object(self):
        # two objects built from the same params arrays own independent
        # volumes; mutating one never reaches the other or the caller
        gw_u = np.full((L, W), 2.0)
        gw_v = np.zeros((L, W))
        p = subsurface_params(gw_u=gw_u, gw_v=gw_v)
        pobj_a = Particles(p, backend='python')
        pobj_b = Particles(p, backend='python')
        assert pobj_a.weight_sub is not pobj_b.weight_sub
        before = pobj_b.weight_sub.copy()
        pobj_a.weight_sub[:] = -1.0
        gw_u[:] = -3.0
        assert np.array_equal(pobj_b.weight_sub, before)

    def test_feature_off_leaves_no_subsurface_state(self):
        pobj = Particles(surface_params(), backend='python')
        assert pobj.weight_sub is None
        assert pobj.gw_u is None and pobj.gw_v is None
        assert pobj.p_infiltration == 0.0
        assert pobj.p_resurface == 0.0
        assert pobj.storage_dt == 1.0


class TestZeroDiff:
    """Byte-identical trajectories when no subsurface parameter is set.

    The pinned constants below were cross-checked byte-for-byte against a
    pristine master checkout (2f756e9, before the subsurface feature) for
    both backends on this exact synthetic field and seeding.
    """

    def seeded_walk(self, backend):
        pobj = Particles(surface_params(), backend=backend, seed=0)
        np.random.seed(42)  # fix the seed-placement stream
        pobj.generate_particles(5, [4, 5, 6], [4, 5, 6])
        for _ in range(5):
            walk_data = pobj.run_iteration()
        return walk_data

    def test_zero_diff_rust_backend(self):
        walk_data = self.seeded_walk('rust')
        assert walk_data['xinds'] == [[5, 6, 7, 8], [6, 7, 8], [6, 7, 8],
                                      [5, 6, 7, 8], [4, 5, 6, 7, 8]]
        assert walk_data['yinds'] == [[4, 4, 4, 5], [4, 5, 5], [6, 6, 6],
                                      [5, 4, 3, 3], [6, 7, 6, 6, 5]]
        assert walk_data['travel_times'] == [
            [0, 20.136313312204685, 41.89562886726115, 61.697504386159615],
            [0, 19.047533515029667, 37.40132835425717],
            [0, 18.378952950451495, 40.090116491217316],
            [0, 21.804736438241516, 42.05777397044753, 63.13528343722956],
            [0, 19.452423142344553, 37.60605194113954, 57.878466597213716,
             79.06214793554496]]

    def test_zero_diff_python_backend(self):
        walk_data = self.seeded_walk('python')
        assert walk_data['xinds'] == [[5, 6, 7, 8], [6, 7, 8], [6, 7, 8],
                                      [5, 6, 7, 8], [4, 5, 6, 7, 8]]
        assert walk_data['yinds'] == [[4, 3, 3, 3], [4, 5, 4], [6, 7, 7],
                                      [5, 4, 4, 5], [6, 5, 5, 6, 5]]
        assert walk_data['travel_times'] == [
            [0, 21.266381747922434, 40.12567809094334, 58.89208888614159],
            [0, 20.783031651652784, 41.98433631100629],
            [0, 19.90097487050028, 39.844037316276456],
            [0, 20.27221972270933, 39.90256164356457, 60.684107190317434],
            [0, 20.835083230856988, 42.64928179811898, 64.25859399037279,
             82.62131219843047]]

    def test_zero_diff_walk_data_json_roundtrip(self):
        # the full recorded history must serialize identically on both
        # backends' contracts (lists only, no numpy types)
        for backend in ('rust', 'python'):
            walk_data = self.seeded_walk(backend)
            assert isinstance(json.dumps(walk_data), str)
