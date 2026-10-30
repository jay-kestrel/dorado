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
    would change its semantics when particles share a cell.
    """
    if staged.steepest_descent:
        raise ValueError("NumPy benchmark does not support steepest_descent")
    frozen_cells = boundary_mask(staged.cell_type)

    def step(particles: Particles, current_inds: list[list[int]],
             travel_times: list[float], start_optional: dict[str, list]) -> StepResult:
        old = np.asarray(current_inds, dtype=np.int64).reshape(-1, 2)
        cells = np.full(len(old), 4, dtype=np.int64)
        live = np.any(old != 0, axis=1)
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
        return new.tolist(), times.tolist(), optional_values(particles, new, start_optional)

    return step
