"""S4 — domain-scaling: routing cost as the domain grows at fixed parcels.

Routes a fixed parcel count for a fixed iteration budget on
right-anchored column crops of the bundled DeltaRCM field — same
parcels, same budget, growing area. Parcels are seeded domain-wide on
wet interior cells rather than at the inlet patch: at fixed parcels a
3-column patch samples one point of the field's texture, and parking on
dry cells scrambles the area signal; domain-wide sampling averages it.

Measured on this kernel (and reported via the fitted exponent), routing
wall time at fixed parcels is essentially FLAT in domain area: per-step
work is local, and longer transits in wider domains are offset by less
per-step dead time. The assertion therefore gates a material inversion,
not strict growth — each wider crop must take at least WALL_FLOOR times
the previous crop's routing time. A strict gate would flake on jitter;
an unbounded one would accept a smaller domain doing systematically
more work, which is the workload/bookkeeping bug this check exists for.
"""

from __future__ import annotations

import numpy as np

import dorado.particle_track as pt
from scenarios.core import (
    MemoryTracker,
    PhaseClock,
    analyze_walk,
    build_record,
    field_info,
    load_deltarcm_field,
    loglog_fit,
)

NAME = "domain-scaling"
DESCRIPTION = (
    "Wall time and peak memory as the domain grows at fixed parcels:"
    " right-anchored column crops, routing cost vs domain area."
)

# Fixed parcels and iteration budget across crops, so area is the only
# sweep variable. The budget is generous enough for transit and parking
# dynamics to play out; on this field not every parcel freezes, so
# there is no early exit.
SWEEP_ITERATIONS = 300
FULL_WIDTHS = [120, 180, 240]
QUICK_WIDTHS = [60, 120]
FULL_PARCELS, QUICK_PARCELS = 2000, 200

# Each wider crop must take at least this fraction of the previous
# crop's routing time (see module docstring: the gate is against
# material inversions, not strict growth).
WALL_FLOOR = 0.9

# Wet interior cells only, two clear of the freeze ring.
WET_DEPTH = 0.0
INTERIOR_MARGIN = 2


def _crop_params(base: pt.modelParams, width: int) -> pt.modelParams:
    """Right-anchored ``width``-column crop of a DeltaRCM field.

    Returns fresh contiguous arrays, so no crop aliases another's
    memory or the loader's.
    """
    start = int(np.asarray(base.depth).shape[1]) - width
    params = pt.modelParams()
    params.stage = np.array(base.stage[:, start:], copy=True)
    params.depth = np.array(base.depth[:, start:], copy=True)
    params.qx = np.array(base.qx[:, start:], copy=True)
    params.qy = np.array(base.qy[:, start:], copy=True)
    params.dx = base.dx
    params.theta = base.theta
    params.model = base.model
    return params


def _wet_seed_cells(params: pt.modelParams) -> list[tuple[int, int]]:
    """Row-major wet interior cells of the crop (deterministic order)."""
    depth = np.asarray(params.depth)
    nrow, ncol = depth.shape
    return [
        (x, y)
        for x in range(INTERIOR_MARGIN, nrow - INTERIOR_MARGIN)
        for y in range(INTERIOR_MARGIN, ncol - INTERIOR_MARGIN)
        if depth[x, y] > WET_DEPTH
    ]


def _assert_no_inversion(routing_seconds: dict[int, float]) -> None:
    """Routing time must not materially decrease as the domain widens."""
    widths = sorted(routing_seconds)
    for lo, hi in zip(widths, widths[1:]):
        if routing_seconds[hi] < WALL_FLOOR * routing_seconds[lo]:
            raise AssertionError(
                f"routing time materially inverted across the sweep:"
                f" {routing_seconds[lo]:.4g}s at {lo} columns vs"
                f" {routing_seconds[hi]:.4g}s at {hi} columns"
                f" (floor {WALL_FLOOR})")


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Sweep domain width at fixed parcels; return one metrics record."""
    np.random.seed(seed)
    widths = QUICK_WIDTHS if quick else FULL_WIDTHS
    parcels = QUICK_PARCELS if quick else FULL_PARCELS
    headline = widths[-1]  # counters + throughput describe this crop

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        base = load_deltarcm_field()
        crops = {w: _crop_params(base, w) for w in widths}
        wet_cells = {w: _wet_seed_cells(crops[w]) for w in widths}

    routing_seconds: dict[int, float] = {}
    headline_walk = None
    for width in widths:
        cells = wet_cells[width]
        pick = np.random.choice(len(cells), size=parcels, replace=False)
        with clock.timed(f"seed_w{width}"), memory.sample():
            particles = pt.Particles(crops[width])
            particles.generate_particles(
                parcels, [cells[i][0] for i in pick],
                [cells[i][1] for i in pick])
        if width == headline:
            # plain 'routing' mirrors the run the record describes
            with clock.timed("routing"), memory.sample():
                with clock.timed(f"route_w{width}"):
                    _route_fixed(particles, SWEEP_ITERATIONS)
            headline_walk = particles.walk_data
        else:
            with clock.timed(f"route_w{width}"), memory.sample():
                _route_fixed(particles, SWEEP_ITERATIONS)
        routing_seconds[width] = clock.snapshot()[f"route_w{width}"]

    _assert_no_inversion(routing_seconds)

    with clock.timed("analysis"):
        n_rows = int(np.asarray(base.depth).shape[0])
        areas = [n_rows * w for w in widths]
        exponent, r2 = loglog_fit(
            areas, [routing_seconds[w] for w in widths])
        processed = analyze_walk(headline_walk, crops[headline])
        processed.update(
            field_info(crops[headline]), iterations=SWEEP_ITERATIONS)

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
            efficiency_extras={
                "scaling_exponent": exponent,
                "scaling_r2": r2,
            },
        )


def _route_fixed(particles: pt.Particles, iterations: int) -> None:
    """Run exactly ``iterations`` routing iterations, no early exit."""
    for _ in range(iterations):
        particles.run_iteration()
