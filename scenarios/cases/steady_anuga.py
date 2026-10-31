"""S2 — steady-anuga-baseline: the contract on a second field family."""

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
    load_anuga_field,
)

NAME = "steady-anuga-baseline"
DESCRIPTION = (
    "Steady routing of the bundled ANUGA field (real discharge): proves the"
    " metrics contract on a second field family."
)

# Mirrors examples/steady_anuga_particles.py: x is the row index (rows
# 20-29), y the column index (columns 48-52).
INLET_SEED_X = list(range(20, 30))
INLET_SEED_Y = list(range(48, 53))

# Quick mode is CI-sized (the whole quick suite must stay ~2 minutes); full
# mode is the demo.
QUICK_PARCELS, QUICK_ITERATIONS = 50, 10
FULL_PARCELS, FULL_ITERATIONS = 5000, 50


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Route parcels across the bundled ANUGA field; return one record."""
    np.random.seed(seed)
    parcels = QUICK_PARCELS if quick else FULL_PARCELS
    iterations = QUICK_ITERATIONS if quick else FULL_ITERATIONS

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_anuga_field()
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
        processed.update(field_info(params), iterations=iterations_run)

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
        )
