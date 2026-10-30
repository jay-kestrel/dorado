"""S8 — determinism-repeat: identical seeded runs must hash identically.

Repeats the S1 workload R times, re-seeding the global RNG before every
repeat (the repo's own convention: examples and tests seed the global
RNG before generating particles). Byte-identical walk-data hashes across
repeats are the assertion — and the tripwire: any future change to
routing behavior surfaces here first.

The record's processed counters describe repeat 0. Per-repeat routing
times live in the phases (routing_r0..rR); the timing_cv field is the
population std/mean of those times (mean and spread are recoverable
from the same phases).
"""

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
    walk_data_sha256,
)

NAME = "determinism-repeat"
DESCRIPTION = (
    "Identical seeded runs repeated R times: walk-data SHA-256 must be"
    " byte-identical (the routing tripwire); reports timing CV."
)

# S1's workload, repeated. Every repeat re-seeds the global RNG, so the
# walks — and the freeze iterations — are identical across repeats.
SEED_X = list(range(15, 17))
SEED_Y = list(range(137, 140))
QUICK_PARCELS, QUICK_ITERATIONS = 50, 20
FULL_PARCELS, FULL_ITERATIONS = 5000, 100
QUICK_REPEATS, FULL_REPEATS = 2, 3


def _route_to_boundary(
    particles: pt.Particles, params: pt.modelParams, max_iterations: int
) -> int:
    """S1's routing loop; returns the number of iterations run."""
    for iteration in range(max_iterations):
        particles.run_iteration()
        if all_on_boundary(particles.walk_data, params):
            return iteration + 1  # every particle is frozen at the boundary
    return max_iterations


def _assert_identical(hashes: list[str], repeats: int, seed: int) -> None:
    """Every repeat must produce the same walk-data digest."""
    if len(set(hashes)) != 1:
        raise AssertionError(
            f"walk-data sha256 differs across {repeats} identical-seed"
            f" repeats at seed {seed}: deterministic routing drifted —"
            " this scenario is the tripwire for that")


def _timing_cv(seconds: list[float]) -> float | None:
    """Population coefficient of variation of per-repeat routing times."""
    arr = np.asarray(seconds, dtype=np.float64)
    if arr.size == 0 or float(arr.mean()) <= 0.0:
        return None
    return float(arr.std() / arr.mean())


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Repeat an identical seeded workload R times; return one record."""
    np.random.seed(seed)
    repeats = QUICK_REPEATS if quick else FULL_REPEATS
    parcels = QUICK_PARCELS if quick else FULL_PARCELS
    max_iterations = QUICK_ITERATIONS if quick else FULL_ITERATIONS

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_deltarcm_field()

    hashes: list[str] = []
    route_seconds: list[float] = []
    first_walk = None
    for repeat in range(repeats):
        np.random.seed(seed)  # identical stream for every repeat
        with clock.timed("seeding"), memory.sample():
            particles = pt.Particles(params)
            particles.generate_particles(parcels, SEED_X, SEED_Y)
        phase = f"routing_r{repeat}"
        if repeat == 0:
            # plain 'routing' mirrors the run the record describes
            with clock.timed("routing"), memory.sample():
                with clock.timed(phase):
                    iterations_run = _route_to_boundary(
                        particles, params, max_iterations)
            first_walk = particles.walk_data
        else:
            with clock.timed(phase), memory.sample():
                iterations_run = _route_to_boundary(
                    particles, params, max_iterations)
        hashes.append(walk_data_sha256(particles.walk_data))
        route_seconds.append(clock.snapshot()[phase])

    _assert_identical(hashes, repeats, seed)

    with clock.timed("analysis"):
        processed = analyze_walk(first_walk, params)
        processed.update(field_info(params), iterations=iterations_run)
        processed["timing_cv"] = _timing_cv(route_seconds)

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
        )
