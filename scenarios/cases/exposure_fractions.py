"""S9 — exposure-fractions: where routed parcels exit, and how long they took."""

from __future__ import annotations

import numpy as np

import dorado.particle_track as pt
from scenarios.core import (
    MemoryTracker,
    PhaseClock,
    all_on_boundary,
    analyze_walk,
    build_record,
    field_info,
    load_deltarcm_field,
)

NAME = "exposure-fractions"
DESCRIPTION = (
    "Steady DeltaRCM run with an exposure analysis layer: exit fractions"
    " per boundary cell and exposure-time percentiles, computed from"
    " walk data with numpy."
)

# Same canonical inlet as steady-rcm-baseline (x indexes rows, y columns).
INLET_SEED_X = list(range(15, 17))
INLET_SEED_Y = list(range(137, 140))

# Quick is CI-sized; full is the demo.
QUICK_PARCELS, QUICK_ITERATIONS = 50, 50
FULL_PARCELS, FULL_ITERATIONS = 500, 50

# Exit fractions sum to the boundary-routed share of the seeded parcels;
# this tolerance only absorbs float rounding of the per-cell divisions.
SUM_TOLERANCE = 1e-9


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Route parcels, then report exit fractions and exposure times."""
    np.random.seed(seed)
    parcels = QUICK_PARCELS if quick else FULL_PARCELS
    iterations = QUICK_ITERATIONS if quick else FULL_ITERATIONS

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_deltarcm_field()
    with clock.timed("seeding"), memory.sample():
        particles = pt.Particles(params)
        particles.generate_particles(parcels, INLET_SEED_X, INLET_SEED_Y)

    iterations_run = 0
    with clock.timed("routing"), memory.sample():
        for _ in range(iterations):
            particles.run_iteration()
            iterations_run += 1
            if all_on_boundary(particles.walk_data, params):
                break  # every particle is frozen at the boundary

    with clock.timed("analysis"):
        processed = analyze_walk(particles.walk_data, params)
        # This scenario's analysis layer: the shared analyzer's exit
        # counts become fractions of all seeded parcels, so per-cell exit
        # shares are directly comparable across run sizes. The record's
        # travel_time_seconds percentiles are the exposure times — the
        # final travel time of each parcel, i.e. how long it spent in the
        # flow before exiting or being parked.
        processed.update(
            field_info(params),
            iterations=iterations_run,
            exits_per_boundary_cell=_exit_fractions(
                processed["exits_per_boundary_cell"],
                int(processed["particles_seeded"]),
            ),
        )
        _assert_fraction_bounds(processed["exits_per_boundary_cell"])

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
        )


def _exit_fractions(
        exit_counts: dict[str, int],
        particles_seeded: int) -> dict[str, float]:
    """Exit counts as fractions of all seeded parcels.

    numpy over the shared analyzer's per-cell exit counts — dorado.spatial
    is avoided on purpose: master's systemwide/localized double-count
    chunk membership. Dividing by the seeded total (not the routed total)
    makes the per-cell fractions sum to the boundary-routed share, so
    they never exceed 1 and stay comparable across run sizes. Keys are
    the analyzer's "x,y" cell labels.
    """
    if particles_seeded <= 0:
        raise ValueError(
            f"cannot normalize exit counts {exit_counts} by"
            f" particles_seeded={particles_seeded}")
    counts = np.array(list(exit_counts.values()), dtype=np.float64)
    fractions = counts / float(particles_seeded)
    return {
        cell: float(fraction)
        for cell, fraction in zip(exit_counts, fractions)
    }


def _assert_fraction_bounds(fractions: dict[str, float]) -> None:
    """Built-in assertion: each fraction in [0, 1], the sum <= 1 + tol."""
    values = np.array(list(fractions.values()), dtype=np.float64)
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise AssertionError(
            f"exit fractions outside [0, 1]: {fractions}; the counter did"
            " not divide by the seeded total")
    total = float(values.sum()) if values.size else 0.0
    if total > 1.0 + SUM_TOLERANCE:
        raise AssertionError(
            f"exit fractions sum to {total}, above 1 + {SUM_TOLERANCE};"
            " some cell counted more exiting parcels than were seeded")
