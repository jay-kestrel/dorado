# -*- coding: utf-8 -*-
"""Reference (pre-fusion) make_weight implementation and memory-probe CLI.

This module preserves the temporaries-heavy NumPy weight construction exactly
as it existed in ``dorado/lagrangian_walker.py`` at master ``914c386`` (the
commit before the fused rewrite) so the parity and memory-gate tests can
compare the fused implementation against it. The helpers and
:obj:`reference_make_weight` are copied verbatim from that revision; the only
intentional deltas are the ``return weight`` line (so callers can grab the
array without reaching through the namespace) and the synthetic-field
builders plus the peak-RSS probe CLI at the bottom.

Run as a script to measure the peak-RSS transient of one implementation on
the synthetic spec-D4 workload (values printed as JSON)::

    python tests/_weight_reference.py reference 1000 2000
    python tests/_weight_reference.py fused 1000 2000

The probe brackets only the construction call with the ``VmHWM`` process
high-water mark, so the reported delta is the memory the call itself
transiently materializes above everything imported or built beforehand. The
fused side loads ``lagrangian_walker`` directly from its file (its imports
are numpy-only) instead of importing the ``dorado`` package, whose
``__init__`` pulls matplotlib and would inflate the baseline high-water
mark.
"""
from __future__ import division, print_function, absolute_import
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def big_sliding_window(raster):
    """Creates 3D array organizing local neighbors at every index

    Returns a raster of shape (L,W,9) which organizes (along the third
    dimension) all of the neighbors in the original raster at a given
    index, in the order [NW, N, NE, W, 0, E, SW, S, SE]. Outputs are
    ordered to match np.ravel(), so it functions similarly to a loop
    applying ravel to the elements around each index.
    For example, the neighboring values in raster indexed at (i,j) are
    raster(i-1:i+2, j-1:j+2).ravel(). These 9 values have been mapped to
    big_ravel(i,j,:) for ease of computations. Helper function for make_weight.

    **Inputs** :

        raster : `ndarray`
            2D array of values (e.g. stage, qx)

    **Outputs** :

        big_ravel : `ndarray`
            3D array which sorts the D8 neighbors at index (i,j) in
            raster into the 3rd dimension at (i,j,:)

    """
    big_ravel = np.zeros((raster.shape[0],raster.shape[1],9))
    big_ravel[1:-1,1:-1,0] = raster[0:-2,0:-2]
    big_ravel[1:-1,1:-1,1] = raster[0:-2,1:-1]
    big_ravel[1:-1,1:-1,2] = raster[0:-2,2:]
    big_ravel[1:-1,1:-1,3] = raster[1:-1,0:-2]
    big_ravel[1:-1,1:-1,4] = raster[1:-1,1:-1]
    big_ravel[1:-1,1:-1,5] = raster[1:-1,2:]
    big_ravel[1:-1,1:-1,6] = raster[2:,0:-2]
    big_ravel[1:-1,1:-1,7] = raster[2:,1:-1]
    big_ravel[1:-1,1:-1,8] = raster[2:,2:]

    return big_ravel


def tile_local_array(local_array, L, W):
    """Take a local array [[NW, N, NE], [W, 0, E], [SW, S, SE]]
    and repeat it into an array of shape (L,W,9), where L, W are
    the shape of the domain, and the original elements are ordered
    along the third axis. Helper function for make_weight.

    **Inputs** :

        local_array : `ndarray`
            2D array of represnting the D8 neighbors around
            some index (e.g. ivec, jvec)

        L : `int`
            Length of the domain

        W : `int`
            Width of the domain

    **Outputs** :

        tiled_array : `ndarray`
            3D array repeating local_array.ravel() LxW times

    """
    return np.tile(local_array.ravel(), (L, W, 1))


def tile_domain_array(raster):
    """Repeat a large 2D array 9 times along the third axis.
    Helper function for make_weight.

    **Inputs** :

        raster : `ndarray`
            2D array of values (e.g. stage, qx)

    **Outputs** :

        tiled_array : `ndarray`
            3D array repeating raster 9 times

    """
    return np.repeat(raster[:, :, np.newaxis], 9, axis=2)


def clear_borders(tiled_array):
    """Set to zero all the edge elements of a vertical stack
    of 2D arrays. Helper function for make_weight.

    **Inputs** :

        tiled_array : `ndarray`
            3D array repeating raster (e.g. stage, qx) 9 times
            along the third axis

    **Outputs** :

        None :
            The input `tiled_array` is modified in place, with the
            borders in the first and second dimension set to 0.

    """
    tiled_array[0,:,:] = 0.
    tiled_array[:,0,:] = 0.
    tiled_array[-1,:,:] = 0.
    tiled_array[:,-1,:] = 0.
    return


def reference_make_weight(Particles):
    """Create the weighting array for particle routing (pre-fusion reference)

    Verbatim copy of ``dorado.lagrangian_walker.make_weight`` as of master
    ``914c386``, used as the parity and memory reference for the fused
    rewrite. The only delta is the trailing ``return weight``.

    **Inputs** :

        Particles : :obj:`dorado.particle_track.Particles` or namespace
            Object exposing the routing fields (stage, depth, qx, qy,
            cell_type, distances, ivec, jvec, dry_depth, theta, gamma,
            stranding_handling)

    **Outputs** :

        Updates the weights array inside the passed object and returns it

    """
    L, W = Particles.stage.shape

    # calculate surface slope weights
    weight_sfc = (tile_domain_array(Particles.stage) \
                  - big_sliding_window(Particles.stage))
    weight_sfc /= tile_local_array(Particles.distances, L, W)
    weight_sfc[weight_sfc <= 0] = 0
    clear_borders(weight_sfc)

    # calculate inertial component weights
    weight_int = (tile_domain_array(Particles.qx)*tile_local_array(Particles.jvec, L, W)) \
                 + (tile_domain_array(Particles.qy)*tile_local_array(Particles.ivec, L, W))
    weight_int /= tile_local_array(Particles.distances, L, W)
    weight_int[weight_int <= 0] = 0
    clear_borders(weight_int)

    # get depth and cell types for neighboring cells
    depth_ind = big_sliding_window(Particles.depth)
    ct_ind = big_sliding_window(Particles.cell_type)

    # set weights for cells that are too shallow, or invalid 0
    weight_sfc[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0
    weight_int[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0

    # if sum of weights is above 0 normalize by sum of weights
    norm_sfc = np.nansum(weight_sfc, axis=2)
    idx_sfc = tile_domain_array((norm_sfc > 0))
    weight_sfc[idx_sfc] /= tile_domain_array(norm_sfc)[idx_sfc]

    norm_int = np.nansum(weight_int, axis=2)
    idx_int = tile_domain_array((norm_int > 0))
    weight_int[idx_int] /= tile_domain_array(norm_int)[idx_int]

    # define actual weight by using gamma, and weight components
    weight = Particles.gamma * weight_sfc + \
             (1 - Particles.gamma) * weight_int

    # modify the weight by the depth and theta weighting parameter
    weight = depth_ind ** Particles.theta * weight

    # if the depth is below the minimum depth then set weight to 0
    weight[(depth_ind <= Particles.dry_depth) | (ct_ind == 2)] = 0

    # if it's a dead end with only nans and 0's, choose deepest cell
    zero_weights = tile_domain_array((np.nansum(weight, axis=2) <= 0))
    deepest_cells = (depth_ind == tile_domain_array(np.max(depth_ind,axis=2)))
    # under stranding_handling = 'park', permanent land is not a valid
    # fallback destination while any non-land cell exists in the window:
    # pick the deepest non-land cell instead, and only fall back to the
    # legacy choice when the whole window is land
    if Particles.stranding_handling == 'park':
        not_land = (ct_ind != 2)
        any_not_land = np.any(not_land, axis=2)
        deepest_dry_aware = \
            (depth_ind == tile_domain_array(np.max(np.where(not_land,
                                                            depth_ind,
                                                            -np.inf),
                                                   axis=2)))
        deepest_cells = ((deepest_dry_aware & not_land) |
                         (deepest_cells &
                          tile_domain_array(~any_not_land)))
    choose_deep_cells = (zero_weights & deepest_cells)
    weight[choose_deep_cells] = 1.0

    # Final checks, eliminate invalid choices
    clear_borders(weight)

    # set weight in the true weight array
    Particles.weight = weight

    return weight


# Default D8 tables, mirroring the constructor defaults in
# dorado.particle_track (distances/ivec/jvec as 3x3 arrays in
# [NW, N, NE, W, center, E, SW, S, SE] raveled order).
_SQRT2 = float(np.sqrt(2))
_SQRT05 = float(np.sqrt(0.5))
_DEFAULT_DISTANCES = np.array([[_SQRT2, 1, _SQRT2],
                               [1, 1, 1],
                               [_SQRT2, 1, _SQRT2]])
_DEFAULT_IVEC = np.array([[-_SQRT05, 0, _SQRT05],
                          [-1, 0, 1],
                          [-_SQRT05, 0, _SQRT05]])
_DEFAULT_JVEC = np.array([[-_SQRT05, -1, -_SQRT05],
                          [0, 0, 0],
                          [_SQRT05, 1, _SQRT05]])


def build_synthetic_particles(L, W, seed=0, nan_patch=False):
    """Build a Particles-like namespace carrying every attribute make_weight reads.

    Fields are deterministic for a given (L, W, seed) and exercise the same
    masking paths as real data: dry cells (depth < dry_depth), land cells
    (cell_type == 2), signed discharges, border cells (cell_type == -1), and
    optionally NaN patches in stage/depth to drive the nansum paths.

    **Inputs** :

        L : `int`
            Domain length (rows)

        W : `int`
            Domain width (columns)

        seed : `int`
            Seed for the synthetic-field generator

        nan_patch : `bool`
            When True, inject NaN patches into stage and depth

    **Outputs** :

        params : :obj:`types.SimpleNamespace`
            Namespace with stage, depth, qx, qy, cell_type, distances,
            ivec, jvec, theta, gamma, dry_depth, stranding_handling

    """
    rng = np.random.default_rng(seed)

    stage = rng.random((L, W)) * 2.0
    depth = rng.random((L, W)) * 2.0
    depth[rng.random((L, W)) < 0.15] = 0.0
    qx = rng.standard_normal((L, W))
    qy = rng.standard_normal((L, W))

    cell_type = np.zeros((L, W), dtype='int')
    cell_type[depth < 0.1] = 2
    cell_type[0, :] = -1
    cell_type[-1, :] = -1
    cell_type[:, 0] = -1
    cell_type[:, -1] = -1

    if nan_patch:
        patch = max(1, min(L, W) // 20)
        stage[:patch, :patch] = np.nan
        depth[L - patch:, W - patch:] = np.nan

    return SimpleNamespace(stage=stage, depth=depth, qx=qx, qy=qy,
                           cell_type=cell_type,
                           distances=_DEFAULT_DISTANCES,
                           ivec=_DEFAULT_IVEC, jvec=_DEFAULT_JVEC,
                           theta=1.0, gamma=0.05, dry_depth=0.1,
                           stranding_handling='park')


def _vm_hwm_kib():
    """Process high-water resident set (VmHWM) in KiB."""
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    raise RuntimeError('VmHWM is not available on this platform')


def _load_lagrangian_walker():
    """Load ``dorado/lagrangian_walker.py`` standalone, bypassing the
    ``dorado`` package ``__init__`` (which imports matplotlib). The module
    is numpy-only, so the probe's import floor stays low enough to bracket
    the construction's own transient with the VmHWM delta."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / 'dorado' / 'lagrangian_walker.py'
    spec = importlib.util.spec_from_file_location('_lw_probe', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(impl, L, W, seed):
    """Run one weight implementation on a synthetic grid and report its peak-RSS delta.

    The delta is the ``VmHWM`` process high-water mark after the call minus
    the high-water mark before it, so it isolates the memory the construction
    itself transiently materializes: imports, the synthetic fields, and any
    earlier peak are outside the bracket. VmHWM (unlike VmRSS) never falls
    when memory is freed, so freed temporaries are still counted.
    """
    import gc
    import json
    import time

    if impl == 'fused':
        fn = _load_lagrangian_walker().make_weight
    elif impl == 'reference':
        fn = reference_make_weight
    else:
        raise SystemExit("impl must be 'fused' or 'reference', got: " + str(impl))

    ns = build_synthetic_particles(L, W, seed=seed, nan_patch=True)

    gc.collect()
    hwm_before = _vm_hwm_kib()
    t0 = time.perf_counter()
    fn(ns)
    seconds = time.perf_counter() - t0
    hwm_after = _vm_hwm_kib()

    print(json.dumps({'impl': impl, 'shape': [L, W], 'seed': seed,
                      'peak_rss_delta_mb': round((hwm_after - hwm_before) / 1024.0, 1),
                      'seconds': round(seconds, 3)}))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        raise SystemExit('usage: python tests/_weight_reference.py '
                         '{reference|fused} L W')
    _measure(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), seed=0)
