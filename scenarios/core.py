"""Shared infrastructure for the dorado demo scenario suite.

Every scenario emits one metrics record with the same schema: ``meta``
provenance, ``speed`` (wall time, per-phase timing, throughput), ``efficiency``
(peak memory, scaling fits), and ``processed`` (counters derived from
``walk_data``). This module owns timing, memory sampling, walk-data analysis,
field loading, and record assembly so scenario cases stay small and their
numbers comparable.

The suite is a pure consumer of dorado's public API: nothing here modifies
dorado state, packaging, or dependencies.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

import dorado.example_data
import dorado.particle_track as pt

# scenarios/ sits directly under the repo root; the suite runs by path, so
# provenance resolves against this checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_DATA_DIR = Path(dorado.example_data.__file__).resolve().parent

# Sections and fields of the unified metrics record. build_record fills every
# field (null where a scenario does not measure it) and rejects unknown keys,
# so schema drift between cases fails loudly instead of silently.
RECORD_SCHEMA: dict[str, tuple[str, ...]] = {
    "meta": (
        "scenario",
        "quick",
        "seed",
        "git_rev",
        "git_dirty",
        "python",
        "numpy",
        "platform",
        "timestamp",
    ),
    "speed": (
        "wall_seconds",
        "phases_seconds",
        "particles_per_second",
        "steps_per_second",
    ),
    "efficiency": (
        "peak_rss_bytes",
        "peak_tracemalloc_bytes",
        "scaling_exponent",
        "scaling_r2",
        "parallel_speedup",
        "parallel_efficiency",
    ),
    "processed": (
        "field_shape",
        "field_dtype",
        "field_bytes",
        "iterations",
        "particles_seeded",
        "particles_routed_to_boundary",
        "particles_active_at_end",
        "steps_appended",
        "exits_per_boundary_cell",
        "unique_cells_visited",
        "travel_time_seconds",
        "per_inlet_seeded",
        "grids_loaded",
        "walk_data_sha256",
        "timing_cv",
    ),
}


class PhaseClock:
    """Times named pipeline phases and overall wall time.

    Wrap each pipeline stage in ``with clock.timed("name")``; repeated or
    nested use of the same name accumulates. ``snapshot`` includes phases that
    are still open (their elapsed-so-far), which is what a record built inside
    the final ``export`` phase should report.
    """

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._closed: dict[str, float] = {}
        self._open: dict[str, float] = {}

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        self._open[name] = start
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._closed[name] = self._closed.get(name, 0.0) + elapsed
            self._open.pop(name, None)

    @property
    def wall_seconds(self) -> float:
        """Seconds elapsed since the clock was created."""
        return time.perf_counter() - self._start

    def snapshot(self) -> dict[str, float]:
        """Per-phase durations, open phases included up to now."""
        now = time.perf_counter()
        phases = dict(self._closed)
        for name, start in self._open.items():
            phases[name] = phases.get(name, 0.0) + (now - start)
        return phases


class MemoryTracker:
    """Peak-memory sampling bracketing the allocation-heavy phases.

    ``sample()`` brackets a region (seeding, routing): tracemalloc is started
    on first use and each region's peak is tracked; ``snapshot`` reads peak
    RSS from ``ru_maxrss``, the kernel's process-wide high-water mark.

    ``ru_maxrss`` counts kilobytes on Linux and bytes on macOS; platforms
    without ``resource`` (Windows) report None rather than a wrong number.
    tracemalloc adds per-allocation overhead to bracketed phases by design —
    that visibility is the cost of measuring the allocation peak.
    """

    def __init__(self) -> None:
        self._peak_traced = 0
        self._owns_tracemalloc = False

    @contextmanager
    def sample(self) -> Iterator[None]:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._owns_tracemalloc = True
        tracemalloc.reset_peak()
        try:
            yield
        finally:
            _, peak = tracemalloc.get_traced_memory()
            self._peak_traced = max(self._peak_traced, peak)

    def snapshot(self) -> dict[str, int | None]:
        """Peak RSS and peak tracemalloc observed so far; stops tracing."""
        peak_rss = _peak_rss_bytes()
        if self._owns_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._owns_tracemalloc = False
        return {
            "peak_rss_bytes": peak_rss,
            "peak_tracemalloc_bytes": int(self._peak_traced),
        }


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        # Windows has no resource module
        return None
    # ru_maxrss is kilobytes on Linux, bytes on macOS
    scale = 1024 if sys.platform.startswith("linux") else 1
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def _position_histories(walk_data: dict) -> tuple[list, list]:
    """Validate and return the per-particle (xinds, yinds) history lists."""
    if not walk_data:
        raise ValueError("walk_data is empty; run the routing loop first")
    xinds = walk_data.get("xinds")
    yinds = walk_data.get("yinds")
    if xinds is None or yinds is None or len(xinds) != len(yinds) or not xinds:
        raise ValueError(
            "walk_data lacks matching, non-empty xinds/yinds histories")
    return xinds, yinds


def _domain_shape(params: pt.modelParams) -> tuple[int, int]:
    depth = getattr(params, "depth", None)
    if depth is None:
        raise ValueError(
            "params has no depth array; the scenario field loaders always"
            " set one")
    nrow, ncol = np.asarray(depth).shape
    return int(nrow), int(ncol)


def _at_boundary(x: int, y: int, nrow: int, ncol: int) -> bool:
    """True for cells in the edge-adjacent ring where routing freezes.

    dorado's cell_type marks the outermost ring of the domain as edge (-1)
    and check_for_boundary freezes any particle whose neighborhood touches
    one, so the closest a routed particle can get to the domain edge is one
    cell in. A particle whose final position is in that ring has reached the
    boundary and will never move again.
    """
    return x <= 1 or x >= nrow - 2 or y <= 1 or y >= ncol - 2


def _final_positions(xinds: list, yinds: list) -> Iterator[tuple[int, int]]:
    for x_hist, y_hist in zip(xinds, yinds):
        yield int(x_hist[-1]), int(y_hist[-1])


def analyze_walk(walk_data: dict, params: pt.modelParams) -> dict[str, object]:
    """Count processed work from one completed walk.

    Pure function of ``walk_data`` and the domain shape. Every
    ``processed``-schema counter in the suite derives from this one analyzer
    so numbers stay consistent across scenarios. x indexes rows and y indexes
    columns, matching dorado's seeding convention. "Routed to boundary"
    counts particles frozen in the edge-adjacent ring (see _at_boundary);
    "active at end" counts everything still in the domain interior, including
    particles parked on dry or land cells.
    """
    xinds, yinds = _position_histories(walk_data)
    nrow, ncol = _domain_shape(params)

    seeded = len(xinds)
    exits: dict[str, int] = {}
    routed = 0
    for x, y in _final_positions(xinds, yinds):
        if _at_boundary(x, y, nrow, ncol):
            routed += 1
            key = f"{x},{y}"
            exits[key] = exits.get(key, 0) + 1

    return {
        "particles_seeded": seeded,
        "particles_routed_to_boundary": routed,
        "particles_active_at_end": seeded - routed,
        "steps_appended": sum(len(hist) for hist in xinds),
        "exits_per_boundary_cell": exits,
        "unique_cells_visited": _unique_cells(xinds, yinds, ncol),
        "travel_time_seconds": _travel_time_percentiles(
            walk_data.get("travel_times")),
        "walk_data_sha256": walk_data_sha256(walk_data),
    }


def all_on_boundary(walk_data: dict, params: pt.modelParams) -> bool:
    """True when every seeded particle is frozen in the boundary ring.

    Lets scenarios stop iterating once nothing in the domain can move again.
    Vacuously true for zero particles, since there is nothing left to route.
    """
    xinds, yinds = _position_histories(walk_data)
    nrow, ncol = _domain_shape(params)
    return all(
        _at_boundary(x, y, nrow, ncol)
        for x, y in _final_positions(xinds, yinds))


def _unique_cells(xinds: list, yinds: list, ncol: int) -> int:
    """Count distinct (x, y) cells touched anywhere in the walk."""
    flat_x = np.fromiter(
        (x for hist in xinds for x in hist), dtype=np.int64)
    flat_y = np.fromiter(
        (y for hist in yinds for y in hist), dtype=np.int64)
    cells = flat_y * np.int64(ncol) + flat_x
    return int(np.unique(cells).size)


def _travel_time_percentiles(histories) -> dict[str, float] | None:
    """min/median/p90/max of each particle's final finite travel time.

    Frozen or parked particles can carry non-finite tail entries; the last
    finite value is the particle's arrival time. None when no particle ever
    recorded a travel time.
    """
    finals = []
    for hist in histories or []:
        arr = np.asarray(hist, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finals.append(float(finite[-1]))
    if not finals:
        return None
    lo, med, p90, hi = np.percentile(finals, [0, 50, 90, 100])
    return {
        "min": float(lo),
        "median": float(med),
        "p90": float(p90),
        "max": float(hi),
    }


def walk_data_sha256(walk_data: dict) -> str:
    """Digest of walk data, dtype-normalized so equal walks hash equal.

    Per key: the key name, the particle count, then each particle history as
    float64 bytes — integer indices are lossless in float64, and NaN payloads
    are canonical IEEE-754 bytes, so repeated seeded runs hash identically.
    Non-numeric histories (e.g. particle kinds) fall back to a canonical JSON
    encoding.
    """
    digest = hashlib.sha256()
    for key in sorted(walk_data):
        histories = walk_data[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(len(histories)).encode("utf-8"))
        for hist in histories:
            digest.update(_history_bytes(hist))
    return digest.hexdigest()


def _history_bytes(history: list) -> bytes:
    try:
        arr = np.asarray(history, dtype=np.float64)
    except (TypeError, ValueError):
        return json.dumps(history, sort_keys=True, default=repr).encode(
            "utf-8")
    return arr.tobytes()


def merge_walk_data(walks: list[dict]) -> dict:
    """Concatenate per-particle histories from several walks into one.

    Used where per-worker or per-batch walks must be accounted jointly: the
    merged result feeds :func:`analyze_walk`, so merged counts can be checked
    against the sum of the parts (e.g. parallel routing bookkeeping —
    dorado's own ``parallel_routing`` already combines per-core results
    internally and returns one dict). Keys must match across every walk; each
    key's per-particle lists are concatenated in order.
    """
    if not walks:
        raise ValueError("no walk data to merge")
    keys = set(walks[0])
    for i, walk in enumerate(walks[1:], start=1):
        if set(walk) != keys:
            raise ValueError(
                f"walk 0 and walk {i} track different variables:"
                f" {sorted(keys)} vs {sorted(walk)}")
    return {
        key: [hist for walk in walks for hist in walk[key]]
        for key in walks[0]
    }


def loglog_fit(sizes, values) -> tuple[float | None, float | None]:
    """Fit ``values = a * sizes**b`` in log-log space; return (exponent, r2).

    Returns (None, None) when fewer than two positive points remain or the
    sizes are degenerate, so callers store nulls instead of inventing a slope;
    r2 is None when the log-values have zero variance (a perfect horizontal
    line has no meaningful r2 denominator).
    """
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    mask = (
        np.isfinite(sizes_arr)
        & np.isfinite(values_arr)
        & (sizes_arr > 0)
        & (values_arr > 0)
    )
    sizes_arr, values_arr = sizes_arr[mask], values_arr[mask]
    if sizes_arr.size < 2 or np.ptp(sizes_arr) == 0:
        return None, None
    log_sizes, log_values = np.log(sizes_arr), np.log(values_arr)
    slope, intercept = np.polyfit(log_sizes, log_values, 1)
    residuals = log_values - (slope * log_sizes + intercept)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_values - log_values.mean()) ** 2))
    r2 = None if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return float(slope), r2


def field_info(params: pt.modelParams) -> dict[str, object]:
    """Shape/dtype/size of the routed depth field, for the processed section."""
    field = np.asarray(params.depth)
    return {
        "field_shape": [int(dim) for dim in field.shape],
        "field_dtype": str(field.dtype),
        "field_bytes": int(field.nbytes),
    }


def load_deltarcm_field() -> pt.modelParams:
    """modelParams for the bundled DeltaRCM demo field.

    Mirrors ``dorado.example_data.define_params.make_rcm_particles``:
    120x240 depth/stage at dx=50 with zero discharge (weights fall back to
    water depth), theta=1.0. np.load returns fresh arrays, so the returned
    params share no state with other callers.
    """
    params = pt.modelParams()
    with np.load(_EXAMPLE_DATA_DIR / "ex_deltarcm_data.npz") as data:
        params.stage = data["stage"]
        params.depth = data["depth"]
        params.dx = 50.0
        # no discharge data: zeros in place of it; weighting can still work
        # from the water depths
        params.qx = np.zeros(np.shape(params.depth))
        params.qy = np.zeros(np.shape(params.depth))
    params.theta = 1.0
    params.model = "DeltaRCM"
    return params


def load_anuga_field() -> pt.modelParams:
    """modelParams for the bundled ANUGA demo field.

    Mirrors ``define_params.make_anuga_params``: 100x100 depth with real
    discharge (qx/qy) at dx=10, stage copied from depth.
    """
    params = pt.modelParams()
    with np.load(_EXAMPLE_DATA_DIR / "ex_anuga_data.npz") as data:
        params.stage = np.copy(data["depth"])
        params.depth = data["depth"]
        params.qx = data["qx"]
        params.qy = data["qy"]
        params.dx = 10.0
    params.theta = 1.0
    params.model = "Anuga"
    return params


def build_record(
    *,
    scenario: str,
    quick: bool,
    seed: int,
    clock: PhaseClock,
    memory: MemoryTracker,
    processed: dict[str, object],
    efficiency_extras: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the unified metrics record; unknown fields raise.

    Sparse by design: every schema field is present, null when the scenario
    does not measure it. Throughput is derived here — seeded particles and
    appended steps over the routing phase — so every scenario reports it the
    same way.
    """
    phases = clock.snapshot()
    routing_seconds = phases.get("routing")
    record = {
        "meta": {
            "scenario": scenario,
            "quick": bool(quick),
            "seed": seed,
            **_git_provenance(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": f"{platform.system()}-{platform.machine()}",
            "timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        },
        "speed": {
            "wall_seconds": float(clock.wall_seconds),
            "phases_seconds": {
                name: float(seconds) for name, seconds in sorted(phases.items())
            },
            "particles_per_second": _per_second(
                processed.get("particles_seeded"), routing_seconds),
            "steps_per_second": _per_second(
                processed.get("steps_appended"), routing_seconds),
        },
        "efficiency": {
            "peak_rss_bytes": None,
            "peak_tracemalloc_bytes": None,
            "scaling_exponent": None,
            "scaling_r2": None,
            "parallel_speedup": None,
            "parallel_efficiency": None,
            **memory.snapshot(),
            **(dict(efficiency_extras) if efficiency_extras else {}),
        },
        "processed": {
            **{name: None for name in RECORD_SCHEMA["processed"]},
            **processed,
        },
    }
    _check_schema(record)
    return record


def _per_second(count: object, seconds: float | None) -> float | None:
    if count is None or not seconds or seconds <= 0:
        return None
    return float(count) / seconds


def _git_provenance() -> dict[str, object]:
    """Short revision and dirty flag of the checkout the run came from."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        # run outside a checkout (e.g. installed distribution): provenance is
        # unknown, not fabricated
        return {"git_rev": None, "git_dirty": None}
    return {"git_rev": rev, "git_dirty": dirty_output.strip() != ""}


def _check_schema(record: dict[str, object]) -> None:
    for section, fields in RECORD_SCHEMA.items():
        present = set(record[section])
        expected = set(fields)
        if present != expected:
            raise ValueError(
                f"metrics record section '{section}' deviates from the"
                f" schema: missing={sorted(expected - present)}"
                f" unknown={sorted(present - expected)}")
