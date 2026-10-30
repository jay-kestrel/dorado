"""S5 — unsteady-flow: parcels carried through every model timestep.

Mirrors the ``routines.unsteady_plots`` loop without the plotting: the
flow field (depth, stage, qx, qy) is reloaded from the bundled unsteady
example CSVs for every timestep, then re-seeded and iterated. The first
step seeds the parcel patch; every later step re-seeds with zero new
particles plus the accumulated walk data, so the same parcels experience
every timestep. One ``run_iteration`` call per timestep, targeting the
cumulative model time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dorado.particle_track as pt
from scenarios.core import (
    REPO_ROOT,
    MemoryTracker,
    PhaseClock,
    analyze_walk,
    build_record,
    field_info,
)

NAME = "unsteady-flow"
DESCRIPTION = (
    "Unsteady routing over the bundled ANUGA example output: the flow"
    " field is reloaded every timestep while the same parcels carry"
    " forward (the unsteady_plots pattern, without the plotting)."
)

# examples/unsteady_data bundles depth/stage/qx/qy CSVs for timesteps
# 10..35 (26 of each). Quick mode runs the first few timesteps so CI stays
# in seconds; full mode walks every available one.
UNSTEADY_DATA_DIR = REPO_ROOT / "examples" / "unsteady_data"
QUICK_TIMESTEPS = 4

# The unsteady example's setup: dx = 5 m, 75 s model timestep, a seed
# patch at rows 5-15 x columns 48-52 (x indexes rows, y columns), and 50
# parcels.
DX = 5.0
TIMESTEP_SECONDS = 75.0
SEED_X = list(range(5, 16))
SEED_Y = list(range(48, 53))
PARCELS = 50

VARIABLES = ("depth", "stage", "qx", "qy")


class MissingUnsteadyDataError(RuntimeError):
    """The bundled unsteady example data is missing or incomplete."""


def _timestep_files(
        data_dir: Path, num_steps: int | None,
) -> tuple[dict[str, list[Path]], int]:
    """Sorted per-variable CSV lists and the resolved timestep count.

    Mirrors the file discovery in ``routines.unsteady_plots`` (list the
    directory, filter by variable prefix, sort). ``num_steps=None``
    resolves to every available timestep. Raises
    :class:`MissingUnsteadyDataError` when the directory is missing,
    empty, or any variable lacks enough files for the requested
    timesteps.
    """
    if not data_dir.is_dir():
        raise MissingUnsteadyDataError(
            f"unsteady example data directory not found: {data_dir}"
            " (expected the bundled examples/unsteady_data"
            " depth/stage/qx/qy CSVs)")
    lists = {
        variable: sorted(data_dir.glob(f"{variable}*.csv"))
        for variable in VARIABLES
    }
    found = ", ".join(
        f"{name}={len(files)}" for name, files in sorted(lists.items()))
    available = min(len(files) for files in lists.values())
    resolved = available if num_steps is None else num_steps
    if resolved <= 0:
        raise MissingUnsteadyDataError(
            f"no complete set of unsteady output CSVs under {data_dir}"
            f" (found {found}; expected the bundled"
            " examples/unsteady_data CSVs)")
    if resolved > available:
        raise MissingUnsteadyDataError(
            f"{data_dir} lacks the CSVs for {resolved} unsteady timesteps:"
            f" found {found}, and every variable needs at least {resolved}"
            " files (expected the bundled examples/unsteady_data CSVs)")
    return lists, resolved


def _load_timestep(
        lists: dict[str, list[Path]], step: int) -> pt.modelParams:
    """modelParams for one timestep's depth/stage/qx/qy CSVs."""
    params = pt.modelParams()
    params.dx = DX
    params.depth = np.loadtxt(lists["depth"][step], delimiter=",")
    params.stage = np.loadtxt(lists["stage"][step], delimiter=",")
    params.qx = np.loadtxt(lists["qx"][step], delimiter=",")
    params.qy = np.loadtxt(lists["qy"][step], delimiter=",")
    return params


def run(quick: bool = False, seed: int = 0) -> dict[str, object]:
    """Carry parcels through the unsteady timesteps; return one record."""
    np.random.seed(seed)

    clock, memory = PhaseClock(), MemoryTracker()
    with clock.timed("load_field"):
        lists, timesteps = _timestep_files(
            UNSTEADY_DATA_DIR, QUICK_TIMESTEPS if quick else None)

    # One pass per timestep: reload the field, (re)seed, iterate. The
    # per-phase timers accumulate across the loop, so each phase reports
    # its total over the run.
    timesteps_run = 0
    # Both are set on the first pass; _timestep_files guarantees at least
    # one timestep.
    walk_data = None
    params = None
    for step in range(timesteps):
        with clock.timed("load_field"):
            params = _load_timestep(lists, step)
        with clock.timed("seeding"), memory.sample():
            particles = pt.Particles(params)
            if walk_data is None:
                particles.generate_particles(PARCELS, SEED_X, SEED_Y)
            else:
                # Continue the same parcels: zero new particles, previous
                # walk data restored; run_iteration sizes the walk from it
                # (the unsteady_plots continuation pattern).
                particles.generate_particles(
                    0, [], [], previous_walk_data=walk_data)
        with clock.timed("routing"), memory.sample():
            walk_data = particles.run_iteration(
                target_time=TIMESTEP_SECONDS * (step + 1))
        timesteps_run += 1

    with clock.timed("analysis"):
        processed = analyze_walk(walk_data, params)
        processed.update(
            field_info(params),
            iterations=timesteps_run,
            grids_loaded=timesteps_run,
        )
        _assert_counts_consistent(processed, timesteps)

    with clock.timed("export"):
        return build_record(
            scenario=NAME,
            quick=quick,
            seed=seed,
            clock=clock,
            memory=memory,
            processed=processed,
        )


def _assert_counts_consistent(
        processed: dict[str, object], timesteps: int) -> None:
    """Built-in assertion: the load counter and walk continuation held.

    A mismatch means the record no longer describes the run: the grid
    count drifted from the timesteps requested, or the walk-data
    continuation dropped or duplicated a particle history.
    """
    if processed["grids_loaded"] != timesteps:
        raise AssertionError(
            f"grids_loaded={processed['grids_loaded']} but the loop"
            f" processed {timesteps} timesteps; the load counter drifted")
    if processed["particles_seeded"] != PARCELS:
        raise AssertionError(
            f"particles_seeded={processed['particles_seeded']} after"
            f" carrying {PARCELS} parcels through {timesteps} timesteps;"
            " the walk-data continuation dropped or duplicated particles")
