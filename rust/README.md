# Native particle stepping

`dorado._native` contains an opt-in, low-level batched stepper. This crate does
not yet select a backend in `Particles.run_iteration`; the Python walker and
public API are unchanged. The backend-switch task will connect this interface.

**Load-bearing assumption:** moving the scalar walk into Rust removes the
interpreter cost. The end-to-end benchmark gate must still check that Python
history assembly, field copies, memory traffic, and thread scheduling do not
consume that gain; a kernel-only timing cannot establish the API speedup.

## Semantics source

The oracle is `dorado/lagrangian_walker.py` and the adaptive branch in
`dorado/particle_track.py` at master **1898034**, not the older 9a437c4 blueprint
snapshot. Rebase verification against **df863fc** found no stepping-semantic
changes (only logging and `run_walk` initialization changed in those files).
`tests/test_native_kernel.py` compares the native operations with these retained
Python implementations using the same controlled uniforms.

The port includes corrected zero-distance freezing (the particle stays at its
current coordinates, not at `(0, 0)`), steepest-descent center exclusion and
center-only dead ends, particle kinds/sizes via effective multipliers,
deposition, and park/release stranding. The existing `(0, 0)` sentinel still
suppresses movement. NaNs in weight rows become zero; all-zero stochastic rows
retain NumPy's left-search CDF behavior. Gamma mixing belongs to precomputed
weights, not to the stepper. Coordinate conventions are already resolved by
Python before staging.

Some reference behaviors are intentionally unusual. Park/release executes
before sentinel, deposition, and boundary movement checks. A wet stranded
particle beside `-1` consumes a probabilistic release draw; a dry boundary
particle books `stranding_dt`. Neither consumes routing, deposition, or diffusion
draws. The adaptive `run_iteration` path instead skips initially boundary-adjacent
particles before the park/release loop. These are preserved, not corrected here.

## Stage fields once

```python
from dorado import _native

fields = _native.RoutingFields(
    weight, velocity, velocity_angle, cell_type, depth,
    distances, iwalk, jwalk, angles, dx, diff_coeff,
    steepest_descent=False,
    stranding_handling="park", dry_depth=0.1, stranding_dt=1.0,
    stranding_release="immediate", rest_time=0.0, entrain_prob=1.0,
)
rng = _native.RngStreams(seed=123)
result = _native.step_particles(
    fields, positions, travel_times, particle_state, rng,
    behavior=behavior, n_threads=1,
)
```

Inputs are C-contiguous NumPy arrays with these exact dtypes and shapes:

| Input | dtype / shape | Meaning |
|---|---|---|
| `weight` | float64 `(L, W, 9)` | Static, already mixed/normalized D8 weights |
| `velocity`, `velocity_angle`, `depth` | float64 `(L, W)` | Speed, flow angle, wet/dry depth |
| `cell_type` | int8 `(L, W)` | Codes in `[-1, 2]` |
| `distances`, `angles` | float64 `(9,)` | Flattened D8 constants |
| `iwalk`, `jwalk` | int64 `(9,)` | Flattened offsets; row uses `jwalk`, column uses `iwalk` |
| `positions` | int64 `(N, 2)` | Mutable row/column coordinates |
| `travel_times` | float64 `(N,)` | Mutable accumulated time |
| `particle_state` | float64 `(N, 3)` | Mutable `[deposited, stranded, stranded_time]`; flags must be 0 or 1 |
| `behavior` (optional) | float64 `(N, 3)` | `[time_multiplier, diffusion_multiplier, deposition_probability]`; defaults `[1, 1, 0]` |

`RoutingFields` validates and owns a copy of its arrays. It is immutable and
shared by Rayon workers. Construct it once per static field, not per iteration;
re-stage when flow/depth changes. This costs one owned field snapshot but removes
per-iteration field copies and avoids shared mutable NumPy fields in workers.
The staging contract requires positive finite velocities (the Python constructor
already regularizes zero speeds). It accepts NaN depths and weights, rejects
infinite weights or row-sum overflow, and rejects negative weights.

The mutable buffers must be writable, non-overlapping and correctly typed.
The entire batch is validated before any particle or RNG advances. Invalid
shapes, coordinates, flags, scalar ranges, strides or values raise a Python
exception. `n_threads` is an integer in `[1, 256]`. No raw pointers or unsafe
blocks are used by this crate. The GIL stays held while NumPy buffers are
borrowed; scoped Rayon workers touch disjoint mutable rows only. The caller must
not mutate those buffers concurrently from another native thread.

Particle kinds/sizes stay a Python policy: call
`lagrangian_walker.effective_multipliers(size, kind)` to form each behavior row.
Other optional tracked fields can be sampled by Python at returned coordinates;
there are no Python callbacks inside the loop.

## Assemble history without changing the public format

For an ordinary step, read mutated buffers and `result.recorded` (one boolean per
particle). Append one row only for a moving or parked particle. `result.histories`
is empty per particle in this mode, avoiding unused per-step allocations. A
stationary deposition flip must still update the existing last `deposited` entry,
without appending a position/time row. Other flags on unrecorded steps are not
committed to Python history, matching `run_iteration`.

For adaptive routing pass `target_time`, optional `est_next_dt` (float64 `(N,)`),
and `max_iter` (positive integer; default 10000). Python derives each initial
estimate from its last two history times, or `0.1` for a one-row history. The
kernel preserves that initial value, then uses `max(0.1, new_time-old_time)` after
each recorded substep. It preserves the closest-time comparison, boundary early
skip, stationary break, and iteration-cap ordering.

`result.histories[i]` then contains the newly recorded tuples
`(row, column, time, deposited, stranded, stranded_time)` in chronological order.
Append these to the existing ragged lists. If deposition happens after movement,
the last returned history row carries the flip. If no row was recorded, copy the
final deposition flag to the existing last row. `result.hit_max_iter` identifies
which particles need the Python iteration-limit log message. The initial row is
never returned, so it must not be duplicated.

## Persistent RNG streams

Each `RngStreams` stores a 64-bit seed and lazy per-particle `ChaCha8Rng` streams.
Stream `i` is `ChaCha8Rng::seed_from_u64(seed)` followed by `set_stream(i)`;
`rand::Rng::gen::<f64>()` supplies uniforms in `[0, 1)`. This is the concrete
stream derivation, **not NumPy SeedSequence or legacy MT19937 emulation**. The
crate versions are locked in `Cargo.lock`. `seed=None` uses OS entropy once and
exposes the resolved seed as `rng.seed`, so the run can be replayed.

Keep the same RNG object across iterations and assign stable particle IDs by row
index. Appending new rows extends streams without resetting existing particles.
Do not compact/reorder particle rows or recreate the RNG each iteration. New
`RngStreams(seed)` resets the whole walk. No NumPy global RNG is read or written.
Independent streams provide the same trajectories for one or eight threads;
Rayon pools are cached per requested thread count in the RNG object.

Within a particle, draw consumption follows the Python state machine: optional
probabilistic release; routing unless frozen or a steepest dead end; deposition
only on a valid nonzero movement with positive deposition probability; diffusion
only after a surviving movement with positive diffusion coefficient. Frozen
particles consume no routing/deposition/diffusion randomness. The diagnostic
`rng.uniforms(particle_id, count)` **consumes** draws, and `rng.draw_counts` reports
consumed counts; use a separate identically seeded object for parity tests.

To regenerate native trajectory constants, set a fixed native seed, preserve
initial particle coordinates/row order, run with a fresh RNG, and record the
mutated buffers and returned histories. Do not regenerate the existing Python
fixtures in this kernel-only PR; NumPy's old global stream remains unchanged.
Statistical ensemble equivalence and full API speedup remain integration gates,
not claims established by helper parity tests.

## Verification

```sh
python -m pip install -e '.[dev]'
pytest -q
cargo fmt --manifest-path rust/Cargo.toml --all -- --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path rust/Cargo.toml
```

The native tests cover controlled-uniform helper parity, all D8 directions,
NaN/zero rows, zero-distance and boundary freezing, kinds/sizes and deposition,
park/release ordering, adaptive termination/history persistence, stream growth,
thread-count determinism, field ownership, and atomic input validation. Python
backend selection, fused weight construction, and end-to-end benchmarks remain
separate tasks. This backend-only change has no rendered UI or interactive
workflow; browser screenshots are not applicable evidence.
