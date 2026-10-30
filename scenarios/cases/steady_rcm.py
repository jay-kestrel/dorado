"""S1 — steady-rcm-baseline: the reference workload every other scenario varies."""

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

NAME = "steady-rcm-baseline"
DESCRIPTION = (
    "Steady routing of the bundled DeltaRCM field (zero discharge): the"
    " reference workload every other scenario varies."
)

# Mirrors define_params.make_rcm_particles: x is the row index (rows 15-16),
# y the column index (columns 137-139).
INLET_SEED_X = list(range(15, 17))
INLET_SEED_Y = list(range(137, 140))

# Quick mode is CI-sized (the whole quick suite must stay ~2 minutes); full
# mode is the demo.
QUICK_PARCELS, QUICK_ITERATIONS = 50, 20
FULL_PARCELS, FULL_ITERATIONS = 5000, 100


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Route parcels across the bundled DeltaRCM field; return one record."""
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
