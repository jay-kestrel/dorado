"""Harness-only batch baseline. Not a supported dorado backend."""

from collections.abc import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from dorado.particle_track import Particles

StepResult = tuple[list[list[int]], list[float], dict[str, list]]
Stepper = Callable[[Particles, list[list[int]], list[float], dict[str, list]], StepResult]


def boundary_mask(cell_type: NDArray) -> NDArray[np.bool_]:
    """Match check_for_boundary's Python slices, including empty slices at zero."""
    padded = np.pad(cell_type == -1, 1, constant_values=False)
    mask = sliding_window_view(padded, (3, 3)).any(axis=(-1, -2))
    mask[0, :] = False
    mask[:, 0] = False
    return mask


def choose_cells(weights: NDArray, draws: NDArray) -> NDArray[np.int64]:
    cutoffs = np.cumsum(np.where(np.isnan(weights), 0.0, weights), axis=1)
    # searchsorted(side='left'): equality selects the first cutoff, even at 0.
    return np.sum(cutoffs < (draws * cutoffs[:, -1])[:, None], axis=1)


def travel_increments(particles: Particles, old: NDArray, new: NDArray,
                      cells: NDArray, draws: NDArray) -> NDArray:
    distances = particles.distances.ravel()[cells] * particles.dx
    projection = np.maximum(0.0, np.cos(
        particles.angles.ravel()[cells] - particles.velocity_angle[old[:, 0], old[:, 1]]))
    diffusion = (0.5 - draws) * particles.diff_coeff if particles.diff_coeff > 0 else 0.0
    return (0.5 * distances * projection
            * (1 / particles.velocity[old[:, 0], old[:, 1]]
               + 1 / particles.velocity[new[:, 0], new[:, 1]]) * (1 + diffusion))


def optional_values(particles: Particles, positions: NDArray,
                    requested: dict[str, list]) -> dict[str, list]:
    rows, cols = positions.T
    result = {}
    roi = getattr(particles, "roi_grid", None)
    if roi is not None:
        result["roi_flag"] = (roi[rows, cols] == 1).astype(int).tolist()
    for name in requested:
        if name not in result and name in particles.available_particle_vars:
            result[name] = particles.available_particle_vars[name][rows, cols].tolist()
    return result


def make_stepper(staged: Particles) -> Stepper:
    """Precompute static boundary membership outside the timed loop.

    Only the stochastic steady-iteration workload is supported. In particular,
    steep_descent mutates shared weight rows sequentially; batching that behavior
    would change its semantics when particles share a cell. The park-while-dry
    stranding machine is batched here, so its probability-rule re-entrainment
    trials are drawn ahead of the weight picks: draw order matches the Python
    stepper exactly under 'immediate' and 'rest' (which consume no extra
    draws) and only diverges under 'probability'.
    """
    if staged.steepest_descent:
        raise ValueError("NumPy benchmark does not support steepest_descent")
    frozen_cells = boundary_mask(staged.cell_type)

    def step(particles: Particles, current_inds: list[list[int]],
             travel_times: list[float], start_optional: dict[str, list]) -> StepResult:
        old = np.asarray(current_inds, dtype=np.int64).reshape(-1, 2)
        # park-while-dry stranding machine, mirroring the Python stepper:
        # parked particles stay in place without drawing a weight pick and
        # book stranding_dt of travel time; release is only ever evaluated
        # from a wet cell
        park = np.zeros(len(old), dtype=bool)
        stranded_time = np.zeros(len(old), dtype=float)
        if particles.stranding_handling == 'park':
            rows, cols = old[:, 0], old[:, 1]
            wet = ((particles.depth[rows, cols] > particles.dry_depth)
                   & (particles.cell_type[rows, cols] != 2))
            was = np.asarray([0 if v is None else v for v in
                              start_optional.get('stranded',
                                                 [None]*len(old))], dtype=int)
            was_time = np.asarray([0.0 if v is None else v for v in
                                   start_optional.get('stranded_time',
                                                      [None]*len(old))],
                                  dtype=float)
            park[:] = ~wet                 # dry or land: park
            free = wet & (was == 0)        # wet, never stranded: move
            park[free] = False
            held = wet & (was != 0)        # stranded, re-wetted: rule
            if particles.stranding_release == 'immediate':
                park[held] = False
            elif particles.stranding_release == 'rest':
                park[held] = was_time[held] < particles.rest_time
            else:  # 'probability'
                u = np.random.random(int(np.count_nonzero(held)))
                park[held] = ~(u < particles.entrain_prob)
            stranded_time = np.where(park,
                                     was_time + particles.stranding_dt,
                                     was_time)
        cells = np.full(len(old), 4, dtype=np.int64)
        live = np.any(old != 0, axis=1) & ~park
        cells[live] = choose_cells(particles.weight[old[live, 0], old[live, 1]],
                                  np.random.random(np.count_nonzero(live)))
        new = old + np.column_stack((particles.jwalk.ravel()[cells],
                                     particles.iwalk.ravel()[cells]))
        new[particles.distances.ravel()[cells] * particles.dx == 0] = 0
        frozen = frozen_cells[old[:, 0], old[:, 1]]
        new[frozen] = old[frozen]
        moving = np.any(old != new, axis=1)
        draws = (np.random.random(np.count_nonzero(moving)) if particles.diff_coeff > 0
                 else np.zeros(np.count_nonzero(moving)))
        times = np.asarray(travel_times, dtype=float).copy()
        times[moving] += travel_increments(particles, old[moving], new[moving], cells[moving], draws)
        times[park] += particles.stranding_dt
        result = optional_values(particles, new, start_optional)
        if particles.stranding_handling == 'park':
            result['stranded'] = park.astype(int).tolist()
            result['stranded_time'] = stranded_time.tolist()
        return new.tolist(), times.tolist(), result

    return step
