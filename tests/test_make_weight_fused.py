# -*- coding: utf-8 -*-
"""Parity and memory gates for the fused make_weight construction.

The fused, channel-wise weight construction in ``dorado.lagrangian_walker``
replaced the temporaries-heavy formulation that materialized ~8 full
(L, W, 9) stack temporaries (a ~1.1 GB peak-RSS transient on 1000x2000
grids; findings ``art_jiwouHPN`` section 2 #4). These tests pin its two
spec-D4 gates against the verbatim pre-fusion reference kept in
``_weight_reference``:

- parity: weight values match the reference within a 1e-12 max relative
  difference on the bundled RCM and ANUGA grids and on synthetic grids
  covering dry cells, land cells, NaN fields, both stranding modes, and
  the gamma extremes (measured: identical except for last-ulp summation
  order in the per-cell normalization sums, ~5e-16 max relative);
- memory: the peak-RSS delta of the fused construction stays under 300 MB
  on the synthetic 1000x2000 grid, and strictly below the reference's.

The memory probe runs each implementation in a fresh subprocess so the
process high-water mark isolates that implementation's own transient.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import dorado.lagrangian_walker as lw
from dorado.example_data import define_params
from dorado.particle_track import Particles

from _weight_reference import build_synthetic_particles, reference_make_weight

HERE = Path(__file__).parent
PARITY_TOL = 1e-12
MEMORY_GATE_MB = 300.0
LARGE_GRID = (1000, 2000)


def _assert_parity(fused, reference, context):
    """Assert weight parity and return the measured max relative difference."""
    assert fused.shape == reference.shape, f"{context}: shape mismatch"
    assert fused.dtype == reference.dtype, f"{context}: dtype mismatch"
    assert np.array_equal(np.isnan(fused), np.isnan(reference)), (
        f"{context}: NaN patterns differ"
    )
    if np.array_equal(fused, reference, equal_nan=True):
        return 0.0
    nan_mask = np.isnan(fused)
    raw = np.abs(fused - reference)
    raw = np.where(np.isnan(raw) & ~nan_mask, 0.0, raw)  # matching infinities
    diff = np.where(nan_mask, 0.0, raw)
    scale = np.where(nan_mask, 1.0, np.maximum(np.abs(fused), np.abs(reference)))
    scale = np.where(scale > 0, scale, 1.0)
    max_rel = float(np.max(diff / scale))
    assert max_rel <= PARITY_TOL, (
        f"{context}: max relative diff {max_rel:.3e} exceeds {PARITY_TOL:g}"
    )
    return max_rel


def test_parity_bundled_rcm():
    """Fused weights match the reference on the DeltaRCM example field."""
    particles = define_params.make_rcm_particles()
    ns = SimpleNamespace(**particles.__dict__)
    lw.make_weight(ns)
    fused = ns.weight
    reference = reference_make_weight(ns)
    _assert_parity(fused, reference, "rcm-bundled")
    assert fused.dtype == np.float64


def test_parity_bundled_anuga():
    """Fused weights match the reference on the ANUGA example field."""
    particles = Particles(define_params.make_anuga_params())
    ns = SimpleNamespace(**particles.__dict__)
    lw.make_weight(ns)
    fused = ns.weight
    reference = reference_make_weight(ns)
    _assert_parity(fused, reference, "anuga-bundled")
    assert fused.dtype == np.float64


@pytest.mark.parametrize("stranding", ["park", "none"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parity_synthetic_small(seed, stranding):
    """Fused weights match the reference on dry/land/NaN synthetic fields."""
    ns = build_synthetic_particles(60, 80, seed=seed, nan_patch=True)
    ns.stranding_handling = stranding
    lw.make_weight(ns)
    fused = ns.weight
    reference_make_weight(ns)
    _assert_parity(fused, ns.weight, f"synthetic-{seed}-{stranding}")


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_parity_gamma_extremes(gamma):
    """The gamma extremes exercise each blend term in isolation."""
    ns = build_synthetic_particles(50, 70, seed=3, nan_patch=True)
    ns.gamma = gamma
    lw.make_weight(ns)
    fused = ns.weight
    reference_make_weight(ns)
    _assert_parity(fused, ns.weight, f"gamma-{gamma}")


def test_parity_synthetic_1000x2000():
    """Fused weights match the reference on the spec-D4 workload grid."""
    ns = build_synthetic_particles(*LARGE_GRID, seed=7, nan_patch=True)
    lw.make_weight(ns)
    fused = ns.weight
    reference_make_weight(ns)
    _assert_parity(fused, ns.weight, "synthetic-1000x2000")


def _probe(impl, L, W):
    """Measure one implementation's peak-RSS delta in a fresh subprocess."""
    cmd = [sys.executable, str(HERE / "_weight_reference.py"), impl, str(L), str(W)]
    env = dict(os.environ, PYTHONPATH=str(HERE.parent))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, cwd=str(HERE.parent), env=env
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(
    sys.platform.startswith("win") or not Path("/proc/self/status").exists(),
    reason="peak-RSS probe reads Linux /proc and ru_maxrss",
)
def test_memory_fused_1000x2000_under_gate():
    """The fused construction stays under the 300 MB spec gate and strictly
    below the reference transient on the 1000x2000 workload."""
    fused = _probe("fused", *LARGE_GRID)
    assert fused["peak_rss_delta_mb"] <= MEMORY_GATE_MB, (
        f"fused make_weight peak RSS delta {fused['peak_rss_delta_mb']} MB "
        f"exceeds the {MEMORY_GATE_MB:g} MB gate"
    )
    reference = _probe("reference", *LARGE_GRID)
    assert reference["peak_rss_delta_mb"] > fused["peak_rss_delta_mb"], (
        "fused construction did not reduce the transient "
        f"(fused {fused['peak_rss_delta_mb']} MB vs reference "
        f"{reference['peak_rss_delta_mb']} MB)"
    )
