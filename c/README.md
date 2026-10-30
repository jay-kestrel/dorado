# dorado C port

A standalone C11 port of dorado's Lagrangian particle-routing kernel, with
**zero external dependencies** (CMake ≥ 3.16 + any C11 compiler). It
reproduces the Python reference's seeded outputs **byte-for-byte** — not
approximately: the end-to-end test compares walk-data JSON bytes against the
Python reference, and the benchmark-scope run (10,000 particles × 100
iterations on the normalized 120×240 fixture) produces byte-identical
`walk_data.json`.

## Scope

The port covers the compute core only:

- **MT19937 RNG** with NumPy-legacy double conversion — `mt19937.[ch]`
- **`.npy` reader/writer** (little-endian `<f8`/`<i8`, C order, npy v1) — `npy.[ch]`
- **params JSON parser** (flat manifest produced by
  `benchmarks/normalize_fixture.py`) and **walk-data/timing JSON writer** — `json.[ch]`
- **Field initialization**: velocity field, cell-type estimation, D8 tables,
  and routing-weight raster construction (`make_weight` parity) — `field.[ch]`
- **Particle walker**: seed picks, weighted random walk / steepest descent,
  park-mode stranding, boundary handling, diffusion travel times, steady-mode
  `run_iteration` with history accumulation — `walker.[ch]`
- **Standalone driver** `dorado_c` — `main.c`

Out of scope by design (spec: "Scope boundary"): plotting, analysis helpers,
netCDF output, Anuga-only extras, and the target-time stepping mode.

## Invocation contract

```sh
dorado_c --input bench/fixture_rcm/ --particles 10000 --iterations 100 \
         --seed 0 --out out/c/
```

Reads the normalized fixture directory (`*.npy` arrays + `params.json`,
produced once by `benchmarks/normalize_fixture.py`), runs the walk, and writes
`<out>/walk_data.json` and `<out>/timing.json`. The **compute clock excludes
I/O**: it starts at field initialization and stops after the final iteration,
before any output is written. `timing.json` records the timed region, appended
step count, fixture, seed, git revision, compiler, and platform — the
provenance fields the benchmark harnesses require.

## Build & test

```sh
cmake -S c -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure
```

The test suite (7 tests):

1. `test_mt19937` — RNG stream anchors: seed 0 first double
   `0.5488135039273248`, seed 42 `0.3745401188473625`, and the
   maximum-32-bit-seed anchor, plus uniform-`searchsorted` draw semantics.
2. `test_npy_json` — `.npy` round-trip, header parsing, JSON parse/format
   fidelity (int/float rendering, shortest-round-trip doubles).
3. `test_weight` — golden weight anchor `weight[1,1,4] == 1.0`.
4. `test_travel_times` — golden travel time `0.5609806565385976` (rel 1e-6).
5. `test_trajectory` — seeded trajectory `xinds == [16, 17, 18, 19]` with
   travel times to rel 1e-6 (the reference's own example-case gate).
6. `e2e_driver` — the driver binary runs the 120×240 fixture (200 particles,
   5 iterations, seed 0) end-to-end.
7. `e2e_compare` — the driver's walk-data output is **byte-identical** to the
   Python reference output checked in under `tests/golden/` (regeneration
   instructions there).

## Parity quirks (replicated, not fixed)

Per the port spec, the port replicates documented reference behavior —
including behavior that is arguably a defect — so trajectories and parity
gates stay comparable. Fixes belong to the Python repos, not the ports.

- **parity quirk — all-zero cutoff rows resolve to cell 0 (NW).** When a
  weight row sums to zero (particle in a cell with no weight-bearing
  neighbors), the cumulative-sum + `searchsorted` scheme returns index 0.
  The port reproduces this instead of erroring or re-picking.
- **parity quirk — boundary check uses clamped slices.** `check_for_boundary`
  examines the 3×3 neighborhood of the *current* cell with slices clamped at
  0 (`max(0, current-1)`), so particles on the first row/column check a real
  neighborhood rather than wrapping. (The port spec described the older
  negative-slice wrap; current master clamps, and the port follows master.)
  A particle whose neighborhood contains an edge cell (`cell_type == -1`)
  freezes in place for the step, even if it was already going to move.
- **parity quirk — steepest descent does not restore ties and operates on a
  copy.** In current master, `steep_descent` copies the weight row (the
  shared raster is *not* mutated — the port spec described the older
  mutating behavior), excludes the current cell (index 4), picks the maximum
  (randomly among ties via `max` on the copied row), and returns the current
  cell at a dead end. The port mirrors this exactly, including the
  copy-not-mutate behavior.

## Precision contract

All grid arrays are float64 (`<f8`) and all accumulation is plain `double`
arithmetic in the reference's exact operation order — no FMA, no
reassociation, no `-ffast-math` (CMake never sets it). This is what makes
byte-identical output possible.

## Verification summary

- `ctest`: 7/7 (anchors, goldens, byte-identical e2e) — warning-free
  `-Wall -Wextra -O2` build.
- Benchmark-scope check (10,000 particles × 100 iterations, seed 0,
  `bench/fixture_rcm`): `walk_data.json` byte-identical to the Python
  reference run of `benchmarks/run_reference.py`; 979,519 appended steps in
  both.
- Python suite (`pytest tests/`): 224 passed on the same commit — the port
  does not touch any Python source.
