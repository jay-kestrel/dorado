"""Quick-mode coverage of the scenario suite, parametrized over the registry.

One quick run per scenario per pytest session — records are cached in
``_RECORDS`` so parametrized tests share runs instead of repeating them —
which keeps the whole suite well under the ~2-minute CI budget. Following
the ``benchmarks/`` rule, timings are recorded and never gated: no test in
this file asserts an absolute wall-time threshold.

The cases carry their own built-in assertions (determinism hashes, scaling
monotonicity, parallel return counts, fraction bounds); the tests here check
those properties at the record level, plus the schema and cross-run
invariants the cases cannot see.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

import numpy as np

import scenarios
from scenarios.core import RECORD_SCHEMA
from scenarios.cases import (
    determinism_repeat,
    domain_scaling,
    multi_inlet,
    parallel_speedup,
    particle_scaling,
    unsteady_flow,
)

# --- one quick run per scenario per session ---------------------------------

_RECORDS: dict[str, dict] = {}


def quick_record(name: str) -> dict:
    """Run the scenario once in quick mode and cache its metrics record."""
    if name not in _RECORDS:
        _RECORDS[name] = scenarios.run_scenario(name, quick=True, seed=0)
    return _RECORDS[name]


REGISTRY = scenarios.scenario_names()

# Scenario-specific sparse fields that carry a value; every other sparse
# field must be null on that scenario's record (the suite contract: sparse,
# not absent).
EFFICIENCY_SPARSE = ("scaling_exponent", "scaling_r2",
                     "parallel_speedup", "parallel_efficiency")
PROCESSED_SPARSE = ("per_inlet_seeded", "grids_loaded", "timing_cv")
FILLED = {
    "particle-scaling": {"scaling_exponent", "scaling_r2"},
    "domain-scaling": {"scaling_exponent", "scaling_r2"},
    "parallel-speedup": {"parallel_speedup", "parallel_efficiency"},
    "multi-inlet": {"per_inlet_seeded"},
    "unsteady-flow": {"grids_loaded"},
    "determinism-repeat": {"timing_cv"},
}


# --- registry ---------------------------------------------------------------

def test_registry_lists_nine_scenarios():
    """Discovery finds the full suite, keyed uniquely with descriptions."""
    assert len(REGISTRY) == 9
    assert len(set(REGISTRY)) == len(REGISTRY)
    for name, description in scenarios.list_scenarios():
        assert name in REGISTRY
        assert description.strip(), f"{name} has an empty description"


# --- schema and meta, for every case ----------------------------------------

@pytest.mark.parametrize("name", REGISTRY)
def test_schema_conformance(name):
    """Every record carries the full schema; timing fields are positive."""
    record = quick_record(name)
    assert set(record) == set(RECORD_SCHEMA)
    for section, fields in RECORD_SCHEMA.items():
        assert set(record[section]) == set(fields), section

    meta = record["meta"]
    assert meta["scenario"] == name
    assert meta["quick"] is True
    assert meta["seed"] == 0

    speed = record["speed"]
    assert speed["wall_seconds"] > 0.0
    assert speed["phases_seconds"]
    for phase, seconds in speed["phases_seconds"].items():
        assert seconds > 0.0, f"{name}: phase {phase}"
    for throughput in ("particles_per_second", "steps_per_second"):
        value = speed[throughput]
        assert value is None or value > 0.0, throughput


@pytest.mark.parametrize("name", REGISTRY)
def test_sparse_fields_null(name):
    """Fields a scenario does not measure are null, never absent."""
    record = quick_record(name)
    filled = FILLED.get(name, set())
    for field in EFFICIENCY_SPARSE:
        value = record["efficiency"][field]
        if field in filled:
            assert value is not None, field
        else:
            assert value is None, field
    for field in PROCESSED_SPARSE:
        value = record["processed"][field]
        if field in filled:
            assert value is not None, field
        else:
            assert value is None, field


@pytest.mark.parametrize("name", REGISTRY)
def test_internal_consistency(name):
    """Counting invariants hold on every run, whatever the workload."""
    processed = quick_record(name)["processed"]

    seeded = processed["particles_seeded"]
    routed = processed["particles_routed_to_boundary"]
    active = processed["particles_active_at_end"]
    assert seeded == routed + active
    assert seeded > 0

    # DeltaRCM-field cases may route zero parcels (diffusive displacement
    # cannot reach the domain edge in a quick iteration budget), so no
    # nonzero-exit assertion exists here — only the counting identity.
    exits = processed["exits_per_boundary_cell"]
    if name == "exposure-fractions":
        # fractions of seeded parcels, not counts (see the case docstring)
        values = list(exits.values())
        assert all(0.0 <= value <= 1.0 for value in values)
        assert sum(values) <= 1.0 + 1e-9
    else:
        assert sum(exits.values()) <= routed

    digest = processed["walk_data_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    int(digest, 16)  # hex-encodable

    assert processed["field_shape"]
    assert all(dim > 0 for dim in processed["field_shape"])
    assert processed["field_bytes"] > 0
    assert processed["iterations"] >= 1


# --- scenario-specific record-level criteria ---------------------------------

def test_particle_scaling_monotone():
    """Quick S3: median routing time grows with Np; the fit is reported."""
    record = quick_record(particle_scaling.NAME)
    phases = record["speed"]["phases_seconds"]
    repeats = particle_scaling.QUICK_REPEATS
    medians = {
        n: float(np.median(
            [phases[f"route_np{n}_r{r}"] for r in range(repeats)]))
        for n in particle_scaling.QUICK_PARTICLES
    }
    counts = sorted(medians)
    for lo, hi in zip(counts, counts[1:]):
        assert medians[hi] >= medians[lo], f"Np {lo} -> {hi}"
    assert medians[counts[-1]] > medians[counts[0]]

    efficiency = record["efficiency"]
    assert efficiency["scaling_exponent"] is not None
    assert efficiency["scaling_r2"] is not None


def test_domain_scaling_fit_reported():
    """Quick S4: the area fit is reported; the no-inversion gate held in-run."""
    record = quick_record(domain_scaling.NAME)
    efficiency = record["efficiency"]
    assert efficiency["scaling_exponent"] is not None
    assert efficiency["scaling_r2"] is not None


def test_parallel_single_core_operation():
    """Quick S7: both core counts returned full shares; speedup reported."""
    record = quick_record(parallel_speedup.NAME)
    # particles_seeded describes the headline run (the largest quick core
    # count): a merged walk of exactly the constant total means every
    # worker — including the 1-core baseline — returned its full share.
    assert record["processed"]["particles_seeded"] == \
        parallel_speedup.QUICK_TOTAL_PARCELS
    phases = record["speed"]["phases_seconds"]
    for cores in parallel_speedup.QUICK_CORES:
        assert phases[f"route_c{cores}"] > 0.0
    efficiency = record["efficiency"]
    assert efficiency["parallel_speedup"] is not None
    assert efficiency["parallel_speedup"] > 0.0
    assert efficiency["parallel_efficiency"] is not None


def test_determinism_hashes_equal_across_runs():
    """Quick S8: two fresh identical-seed runs hash identically.

    The case asserts hash identity across its internal repeats; this is the
    cross-run version — separate processes of routing, same seed, same bytes.
    """
    first = quick_record(determinism_repeat.NAME)
    second = scenarios.run_scenario(determinism_repeat.NAME, quick=True,
                                    seed=0)
    assert first["processed"]["walk_data_sha256"] == \
        second["processed"]["walk_data_sha256"]
    assert first["processed"]["timing_cv"] is not None


def test_unsteady_grids_loaded():
    """Quick S5: one grid per processed timestep."""
    record = quick_record(unsteady_flow.NAME)
    assert record["processed"]["grids_loaded"] == unsteady_flow.QUICK_TIMESTEPS


def test_missing_unsteady_data_raises(tmp_path, monkeypatch):
    """A missing unsteady data directory raises the named error."""
    monkeypatch.setattr(
        unsteady_flow, "UNSTEADY_DATA_DIR", tmp_path / "absent")
    with pytest.raises(unsteady_flow.MissingUnsteadyDataError,
                       match="unsteady_data"):
        unsteady_flow.run(quick=True, seed=0)


def test_multi_inlet_per_inlet_sums():
    """Quick S6: per-inlet seeded counts sum to the record's total."""
    record = quick_record(multi_inlet.NAME)
    per_inlet = record["processed"]["per_inlet_seeded"]
    assert per_inlet is not None
    assert set(per_inlet) == set(multi_inlet.INLET_NAMES)
    inlet_seeded = sum(
        int(stats["particles_seeded"]) for stats in per_inlet.values())
    assert inlet_seeded == record["processed"]["particles_seeded"]
    for stats in per_inlet.values():
        assert stats["particles_seeded"] == (
            stats["particles_routed_to_boundary"]
            + stats["particles_active_at_end"])
