"""S7 — parallel-speedup: parallel_routing wall time vs worker count.

``parallel_routing`` hands EACH worker the full ``Np_tracer`` (its log
line: "%d particles per worker"; each worker's ``run_iter`` calls
``generate_particles`` itself), so constant total work across core
counts means passing ``Np_tracer = total // num_cores``. Workers are
reseeded independently on this revision (one seed per worker drawn from
the parent RNG), but this scenario claims SPEEDUP ONLY — trajectory
distinctness across workers is neither asserted nor implied.

Per-core results are observable only through the combined walk (the
function merges worker results before returning), and each worker seeds
exactly ``Np_tracer`` parcels — so a combined walk holding exactly
``total`` histories is the assertion that every core returned its share.

Speedup and parallel efficiency are reported at the largest core count
and never gated: at quick scale the process pool's startup cost can
dominate, which is a real property of small workloads, not a failure.
The parent passes a bare Particles object (its ``walk_data`` is None),
so workers generate their own parcels cleanly — ``generate_particles``
appends to any existing walk data.
"""

from __future__ import annotations

import numpy as np

import dorado.parallel_routing as pr
import dorado.particle_track as pt
from scenarios.core import (
    MemoryTracker,
    PhaseClock,
    analyze_walk,
    build_record,
    field_info,
    load_deltarcm_field,
)

NAME = "parallel-speedup"
DESCRIPTION = (
    "Parallel routing wall time vs core count at constant total"
    " parcels: speedup and parallel efficiency at the largest count."
)

# Inlet patch and iterations mirror S1's workload; the total parcel
# count is held constant across core counts (per-worker Np shrinks as
# cores grow), so wall comparisons isolate parallelism.
SEED_X = list(range(15, 17))
SEED_Y = list(range(137, 140))
FULL_CORES = [1, 2, 4, 8]
QUICK_CORES = [1, 2]
FULL_TOTAL_PARCELS, FULL_ITERATIONS = 5000, 100
QUICK_TOTAL_PARCELS, QUICK_ITERATIONS = 800, 20


def _assert_full_return(walk: dict, total: int, cores: int) -> None:
    """Combined walk must hold exactly ``total`` particle histories.

    Each worker seeds exactly its per-worker share, so a short combined
    walk means a worker returned nothing or bookkeeping dropped
    histories — the "all cores returned" check, observable only through
    the combined count.
    """
    returned = len(walk["xinds"])
    if returned != total:
        raise AssertionError(
            f"parallel_routing with {cores} core(s) returned {returned}"
            f" particle histories, expected {total}: every worker must"
            " return its share of the parcels")


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Sweep core counts at constant total parcels; return one record."""
    np.random.seed(seed)
    cores_list = QUICK_CORES if quick else FULL_CORES
    total = QUICK_TOTAL_PARCELS if quick else FULL_TOTAL_PARCELS
    iterations = QUICK_ITERATIONS if quick else FULL_ITERATIONS
    if cores_list[0] != 1:
        raise AssertionError(
            "speedup needs a 1-core baseline as the first sweep point")
    for cores in cores_list:
        if total % cores:
            raise AssertionError(
                f"total parcels {total} do not divide evenly across"
                f" {cores} core(s); per-worker shares must be integral")

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_deltarcm_field()
    with clock.timed("seeding"), memory.sample():
        # bare object: workers generate their own parcels
        particles = pt.Particles(params)

    headline_cores = cores_list[-1]
    headline_walk = None
    route_seconds: dict[int, float] = {}
    for cores in cores_list:
        phase = f"route_c{cores}"
        if cores == headline_cores:
            # plain 'routing' mirrors the run the record describes
            with clock.timed("routing"), memory.sample():
                with clock.timed(phase):
                    walk = pr.parallel_routing(
                        particles, iterations, total // cores,
                        SEED_X, SEED_Y, cores)
            headline_walk = walk
        else:
            with clock.timed(phase), memory.sample():
                walk = pr.parallel_routing(
                    particles, iterations, total // cores,
                    SEED_X, SEED_Y, cores)
        _assert_full_return(walk, total, cores)
        route_seconds[cores] = clock.snapshot()[phase]

    with clock.timed("analysis"):
        processed = analyze_walk(headline_walk, params)
        processed.update(field_info(params), iterations=iterations)

    with clock.timed("export"):
        baseline = route_seconds[1]
        speedup = baseline / route_seconds[headline_cores]
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
            efficiency_extras={
                "parallel_speedup": float(speedup),
                "parallel_efficiency": float(speedup / headline_cores),
            },
        )
