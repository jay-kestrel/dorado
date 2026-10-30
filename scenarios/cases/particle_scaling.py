"""S3 — particle-scaling: routing cost as the seeded parcel count grows.

Routes fixed-iteration workloads at growing parcel counts on the bundled
DeltaRCM field and fits the log-log slope of routing wall time against
Np. Routing work is linear in parcels, so the exponent should sit near
1.0; departures are flagged for review, never gated — the smallest quick
points are milliseconds of work, where jitter can bend a fit without
indicating a real problem.

The record's processed counters and throughput describe the largest
sweep point's final repeat (the demo-scale run, comparable with S1);
the sweep itself lives in the per-point timing phases and the fitted
exponent. Peak memory covers every seeding/routing region of the sweep.
"""

from __future__ import annotations

import logging

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

NAME = "particle-scaling"
DESCRIPTION = (
    "Wall time and peak memory as the parcel count grows at fixed"
    " iterations: fitted log-log scaling exponent of routing cost vs Np."
)

logger = logging.getLogger(__name__)

# Fixed seed patch and iteration budget, so work per sweep point is
# proportional to Np alone. No early exit: the sweep compares equal
# iteration budgets, not time-to-freeze.
SEED_X = list(range(15, 17))
SEED_Y = list(range(137, 140))
SWEEP_ITERATIONS = 20
FULL_PARTICLES = [500, 2000, 8000, 32000]
QUICK_PARTICLES = [10, 40, 160, 640]

# Quick points are tens of milliseconds, where GC and allocator jitter
# rival the signal: a median over repeats makes the monotonicity check
# trustworthy. Full points self-average, so one repeat is enough.
QUICK_REPEATS, FULL_REPEATS = 5, 1

# Advisory-only sanity band for the fitted exponent (flag, never fail).
EXPONENT_CENTER, EXPONENT_TOLERANCE = 1.0, 0.3


def _route_fixed(particles: pt.Particles, iterations: int) -> None:
    """Run exactly ``iterations`` routing iterations, no early exit."""
    for _ in range(iterations):
        particles.run_iteration()


def _assert_monotonic(medians: dict[int, float]) -> None:
    """Median routing time must not decrease as Np grows.

    Equal iteration budgets mean more parcels is strictly more work, so
    a decrease means the sweep measured noise or a bookkeeping bug.
    """
    counts = sorted(medians)
    for lo, hi in zip(counts, counts[1:]):
        if medians[hi] < medians[lo]:
            raise AssertionError(
                f"routing time not monotone in Np: {medians[lo]:.4g}s at"
                f" Np={lo} above {medians[hi]:.4g}s at Np={hi}")
    if medians[counts[-1]] <= medians[counts[0]]:
        raise AssertionError(
            f"routing time did not increase across the sweep:"
            f" {medians[counts[0]]:.4g}s at Np={counts[0]} vs"
            f" {medians[counts[-1]]:.4g}s at Np={counts[-1]}")


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Sweep Np at fixed iterations; return one metrics record."""
    np.random.seed(seed)
    sweep = QUICK_PARTICLES if quick else FULL_PARTICLES
    repeats = QUICK_REPEATS if quick else FULL_REPEATS
    headline = sweep[-1]  # counters + throughput describe this point

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        params = load_deltarcm_field()

    routing_times: dict[int, list[float]] = {n: [] for n in sweep}
    headline_walk = None
    for n_parcels in sweep:
        for repeat in range(repeats):
            phase = f"route_np{n_parcels}_r{repeat}"
            with clock.timed(f"seed_np{n_parcels}"), memory.sample():
                particles = pt.Particles(params)
                particles.generate_particles(n_parcels, SEED_X, SEED_Y)
            if n_parcels == headline and repeat == repeats - 1:
                # plain 'routing' mirrors the run the record describes
                with clock.timed("routing"), memory.sample():
                    with clock.timed(phase):
                        _route_fixed(particles, SWEEP_ITERATIONS)
                headline_walk = particles.walk_data
            else:
                with clock.timed(phase), memory.sample():
                    _route_fixed(particles, SWEEP_ITERATIONS)
            routing_times[n_parcels].append(clock.snapshot()[phase])

    medians = {n: float(np.median(ts)) for n, ts in routing_times.items()}
    _assert_monotonic(medians)

    with clock.timed("analysis"):
        exponent, r2 = loglog_fit(sweep, [medians[n] for n in sweep])
        if exponent is not None and abs(
                exponent - EXPONENT_CENTER) > EXPONENT_TOLERANCE:
            logger.warning(
                "particle-scaling: fitted exponent %.3f outside the"
                " sanity band %.1f +/-%.1f (flag only, never gating)",
                exponent, EXPONENT_CENTER, EXPONENT_TOLERANCE)
        processed = analyze_walk(headline_walk, params)
        processed.update(field_info(params), iterations=SWEEP_ITERATIONS)

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
