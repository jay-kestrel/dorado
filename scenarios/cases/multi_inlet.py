"""S6 — multi-inlet: two seed patches routed on one Particles object."""

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

NAME = "multi-inlet"
DESCRIPTION = (
    "Two seed patches appended to one Particles object through"
    " successive generate_particles calls: per-inlet seeded, routed,"
    " exit, and travel counts keyed off seed order."
)

# Inlet 1 is the canonical DeltaRCM demo inlet (the
# define_params.make_rcm_particles patch); inlet 2 sits two rows
# downstream in the same channel, so both patches are wet on the bundled
# field. x indexes rows, y indexes columns.
INLET_1_SEED_X = list(range(15, 17))
INLET_1_SEED_Y = list(range(137, 140))
INLET_2_SEED_X = list(range(17, 19))
INLET_2_SEED_Y = list(range(137, 140))

# Parcels per inlet and routing iterations (quick is CI-sized, full is
# the demo).
QUICK_PARCELS, QUICK_ITERATIONS = 20, 20
FULL_PARCELS, FULL_ITERATIONS = 2500, 100

INLET_NAMES = ("inlet_1", "inlet_2")


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Route two inlet patches on one Particles object; return one record."""
    np.random.seed(seed)
    parcels = QUICK_PARCELS if quick else FULL_PARCELS
    iterations = QUICK_ITERATIONS if quick else FULL_ITERATIONS

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_deltarcm_field()
    with clock.timed("seeding"), memory.sample():
        particles = pt.Particles(params)
        # Successive generate_particles calls append each new patch to
        # the same walk_data (see the Particles.generate_particles
        # docstring), so inlet identity is the seed order: the first
        # ``parcels`` histories are inlet 1's.
        particles.generate_particles(parcels, INLET_1_SEED_X, INLET_1_SEED_Y)
        particles.generate_particles(parcels, INLET_2_SEED_X, INLET_2_SEED_Y)

    iterations_run = 0
    with clock.timed("routing"), memory.sample():
        for _ in range(iterations):
            particles.run_iteration()
            iterations_run += 1
            if all_on_boundary(particles.walk_data, params):
                break  # every particle is frozen at the boundary

    with clock.timed("analysis"):
        processed = analyze_walk(particles.walk_data, params)
        processed.update(
            field_info(params),
            iterations=iterations_run,
            per_inlet_seeded=_per_inlet_stats(
                particles.walk_data, params, parcels),
        )
        _assert_inlets_sum(processed)

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
        )


def _per_inlet_stats(
        walk_data: dict, params: pt.modelParams,
        parcels: int) -> dict[str, dict[str, object]]:
    """Per-inlet attribution keyed off seed order.

    The seeded histories are appended in call order, so slicing the
    per-particle lists at ``parcels`` splits the two inlets; each slice
    runs through the same shared analyzer as the whole-run counters. The
    record field is named ``per_inlet_seeded`` per the suite schema and
    carries the full per-inlet block (seeded, routed, exits, travel
    times) for each inlet.
    """
    slices = (
        _inlet_slice(walk_data, 0, parcels),
        _inlet_slice(walk_data, parcels, None),
    )
    return {
        name: analyze_walk(inlet_walk, params)
        for name, inlet_walk in zip(INLET_NAMES, slices)
    }


def _inlet_slice(walk_data: dict, start: int, stop: int | None) -> dict:
    """Copy of walk_data with only the per-particle histories [start:stop]."""
    return {key: list(value[start:stop]) for key, value in walk_data.items()}


def _assert_inlets_sum(processed: dict[str, object]) -> None:
    """Built-in assertion: per-inlet seeded counts sum to the total Np."""
    per_inlet = processed["per_inlet_seeded"]
    inlet_seed_counts = [
        int(stats["particles_seeded"]) for stats in per_inlet.values()]
    total_seeded = int(processed["particles_seeded"])
    if sum(inlet_seed_counts) != total_seeded:
        raise AssertionError(
            f"per-inlet seeded counts {inlet_seed_counts} sum to"
            f" {sum(inlet_seed_counts)}, not the {total_seeded} particles"
            " in walk_data; seed-order attribution is broken")
