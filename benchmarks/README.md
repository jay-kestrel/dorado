# Routing benchmark

Measure `Particles.run_iteration`, including its Python walk-data bookkeeping,
without plotting, profiling, file loading, weight construction, or particle
seeding in the timed region. No library code is changed by this harness.

```sh
python benchmarks/bench_routing.py --output /tmp/routing.json
python benchmarks/bench_routing.py --quick --output /tmp/routing-quick.json
python benchmarks/bench_routing.py --backend python --output /tmp/python.json
```

Run from a Git checkout with dorado's dependencies installed. The script also
works by absolute path from another directory. `--steps`, `--repeats`, and
`--seed` override the defaults. Unset `DORADO_BACKEND` first; an environment
backend override could otherwise mislabel a result.

## Workloads and timing contract

The full command loads both bundled fields once, then stages one particle object
per field/count/backend combination. Each repeat deep-copies this pristine object
outside timing, including mutable weights and RNG state. One separate warmup
iteration runs on a disposable copy. All repeats use the same seed and starting
positions; repeats measure timing noise, not independent scientific ensembles.

| Field | Grid | dx | Seed rows / columns | Particles | Iterations | Repeats |
| --- | --- | --- | --- | --- | --- | --- |
| DeltaRCM | 120 x 240 | 50 m | 15–16 / 137–139 | 500 / 2,500 / 5,000 | 20 | 3 |
| ANUGA | 100 x 100 | 10 m | 20–29 / 48–52 | 500 / 2,500 / 5,000 | 50 | 3 |

Parameters follow `dorado/example_data/define_params.py`; ANUGA uses that helper's
10 m dx, not `examples/timing_anuga_particles.py`'s 50 m. `--quick` runs only
ANUGA at 500 particles, three iterations, one repeat. CI runs it before the
existing examples and uploads its JSON; it does not assert noisy speed thresholds.

The table reports the median wall time and microseconds per **attempted**
particle-step, including frozen particles. JSON also records every timing sample,
appended-step counts, field hashes, source hashes, Git revision/dirty state,
Python/NumPy/platform versions, and thread environment. Compare appended-step
counts as well as timings so early freezing cannot masquerade as acceleration.
Do not compare elapsed times across different hardware or iteration counts.

## Backends and limits

`python` runs the unchanged reference walker. `numpy` replaces only the stepper
inside a temporary benchmark context, so both paths pay the same
`run_iteration` bookkeeping cost. The NumPy variant gathers `(Np, 9)` weights and
batches direction, boundary, and travel-time operations. Boundary membership is
precomputed outside timing because the field is static. It supports these
stochastic steady workloads only, rejects steepest descent (whose sequential
weight mutations need separate treatment), and is **not a shipped library backend**.

The reference chooses all directions before drawing diffusion noise. The batch
variant retains that order for these workloads, including direction draws for
boundary particles. Tests compare per-step displacement histograms, trajectories,
and travel times on both fields; edge tests cover the sentinel, NaNs, zero rows,
boundaries, and optional sampling. These checks do not replace the future
2,500-particle statistical-equivalence gate for a different native RNG.

The default command also selects `rust` when both `dorado._native` and the public
`Particles(backend=..., seed=..., n_threads=...)` constructor exist. Before that
port lands, JSON and stdout explicitly record a skip; `--backend rust` fails
instead of silently substituting Python. A broken native import or runtime raises
an error. Native state must support deepcopy for repeat isolation. This adapter
is not native-performance evidence until the kernel exists and is tested.

## Recorded pre-port baseline

`results/baseline-py.json` was measured on 2026-10-30 against unchanged master
routing at `9a437c4699f53874d7c7d16d8c52ea96d9622989`, before any port commit.
The dirty flag is true because the new harness was uncommitted; both routing
source hashes are recorded and the library files were unchanged. This is the
output of the Python-only command above, not hand-entered timing data.

| Field | Particles | Median (s) | Microseconds / attempted particle-step |
| --- | ---: | ---: | ---: |
| DeltaRCM | 500 | 0.117927 | 11.793 |
| DeltaRCM | 2,500 | 0.660999 | 13.220 |
| DeltaRCM | 5,000 | 1.279317 | 12.793 |
| ANUGA | 500 | 0.291813 | 11.673 |
| ANUGA | 2,500 | 1.537036 | 12.296 |
| ANUGA | 5,000 | 3.148002 | 12.592 |

All six values are within twice the approximately 12 microseconds/step in the
2026-10-30 research findings (`art_jiwouHPN`, section 8). The full provenance is
in the JSON; these measurements use Python 3.13.14 and NumPy 2.3.5.

`results/python-numpy.json` records the full default command on 2026-10-30
at harness commit `a49ff18`. Python/NumPy appended-step counts match in all six
cases. On ANUGA with 2,500 particles, the medians are 1.553572 s / 0.175990 s
(8.83x for NumPy); Rust is explicitly skipped. The dirty flag here comes from
untracked test-generated `figs/`, not source changes.

The later native gate is at least 10x single-thread speedup on ANUGA with 2,500
particles, using a same-machine reference rerun. If NumPy is within 20% of native,
the approved spec requires revisiting the toolchain decision. Neither native
gate can be evaluated by this pre-port PR.

## Normalized fixtures (bench/)

`python benchmarks/normalize_fixtures.py` converts the bundled npz archives
into the uncompressed `.npy` arrays and flat `params.json` manifests that the
C and Rust ports consume: `bench/fixture_rcm/` (120 x 240 DeltaRCM field) and
`bench/fixture_anuga/` (100 x 100 ANUGA field, NaN depths preserved). Arrays
are cast to the dtype the reference computes in (`<name>.f64.npy`), plus the
init-estimated `cell_type.i64.npy`; manifest scalars follow the parameter set
that reproduces the seeded goldens in `tests/test_examplecases.py` (`dx`
50 m/10 m, `theta` 1.0, `gamma` 0.05, `diff_coeff` 0.2, `dry_depth` 0.1). The
DeltaRCM field has no discharge arrays, so its zero `qx`/`qy` are baked in.
Two runs produce byte-identical files (JSON is written as raw bytes, no
platform newline translation), and a test keeps the committed bytes equal to a
fresh normalization. Default workload in each manifest: seed 0, 10,000
particles, 100 iterations.

## Reference runner

`benchmarks/run_reference.py` runs the unchanged Python routing stack on one
fixture directory and writes `walk_data.json` (the reference output the native
binaries must reproduce from the same bytes) and `timing.json` (seed, workload,
and environment provenance) to `bench/out/python/` by default:

```sh
python benchmarks/run_reference.py --input bench/fixture_rcm/ --particles 10000 \
    --iterations 100 --seed 0 --out bench/out/python/
```

`--particles`, `--iterations`, and `--seed` default to the manifest values;
`--seed-x/--seed-y` default to the `define_params.py` seeding regions per
field. The timed region covers `Particles` init (velocity/cell-type/D8/weight
derivation), `generate_particles`, and the `run_iteration` loop; npy/params
loading and JSON writing are excluded, matching the port binaries' contract.
Single runs of the default workload on 2026-10-30 (sandbox, Python 3.13.14):
DeltaRCM 18.37 s / 979,519 appended steps; ANUGA 17.82 s / 1,000,000 appended
steps. The seeded golden gate (one particle at x=16, y=140, 3 iterations)
reproduces the `tests/test_examplecases.py` travel times to rel=1e-6; the
tiny gap to the exact printed values is the fixture's float64 cast, not drift.

## Harness runner (python vs C vs Rust)

`benchmarks/run_harness.py` is the head-to-head rig: all three implementations
run the same normalized fixture bytes, and the output is a machine-readable
`results.json` per workload.

```sh
python benchmarks/run_harness.py --input bench/fixture_rcm/ --out bench/out/harness/
python benchmarks/run_harness.py --particles 500 --iterations 20 --repeats 3
```

Workload defaults (particles, iterations, seed, seeding region) come from the
fixture manifest, so the harness and `run_reference.py` benchmark the same
workload. Both native binaries are built fresh — CMake Release for `dorado_c`,
cargo release for `dorado_rust` (cargo must be on `PATH`) — unless
`--c-binary`/`--rust-binary` point at prebuilt ones.

**Parity gate.** Repetition 1 of each implementation produces `walk_data.json`;
the C and Rust bytes must equal the Python reference's bytes exactly. Any
mismatch fails the run (nonzero exit, parity `fail` in `results.json`, the
first differing byte reported) and is never resolved by loosening tolerance —
walk data is byte-exact by design, so a mismatch is a port bug. Repetition-1
walk files are kept under `<out>/<impl>/rep1/` as parity evidence; later reps
keep only their `timing.json`.

**Timing contract.** One untimed warmup per implementation (same code paths,
bounded at 200 particles × 5 iterations) precedes the timed repetitions.
Compute seconds come from each implementation's own clock: the port binaries
report `timing.json` seconds with fixture load and JSON write excluded; the
Python reference is timed in-process by `run_reference.run_reference` over the
same region. Wall time around each whole port process (startup + I/O included)
is recorded separately as `all_wall_seconds`/`median_wall_seconds` — on the
default workload the walk-data JSON write dominates port wall time, which is
exactly the distinction the two numbers exist to show. dorado has no JIT, so
there is no cold-start line item. Headline numbers are warm medians of
`--repeats` (default 5); every sample is kept in `all_seconds`.

**results.json.** One file per harness run: `workload` (fixture path + hashes,
particles, iterations, seed, seeding box, repetitions, warmup size, binary
paths), `provenance` (git revision + dirty flag, platform/machine,
Python/NumPy versions, C compiler and Rust toolchain as reported by the port
binaries, binary sha256s, `DORADO_BACKEND`), `results` per implementation
(`all_seconds`, `median_seconds`, `all_wall_seconds`,
`median_wall_seconds`, `seconds_per_iteration`, `appended_steps`, `parity`),
and `speedup_vs_python` (compute medians). Fixed-seed repetitions must agree
on appended steps — a disagreement aborts the run rather than producing
misleading timings.

**Environment guards.** `DORADO_BACKEND` overrides the `Particles`
constructor argument (`particle_track._resolve_backend`), so the harness
refuses to run with it set to anything but `python` — otherwise the "Python
reference" could silently be the native kernel. `DORADO_COBOL=1` is refused
for the same reason.

First harness run on 2026-10-30 (sandbox, commit `7477df0` plus an uncommitted
tree, 5 reps, default workload — 10,000 particles × 100 iterations, 979,519
appended steps for every implementation, parity PASS on both ports):

| Implementation | Median compute (s) | s/iteration | Median wall (s) |
| --- | ---: | ---: | ---: |
| python | 21.845 | 0.21845 | 21.845 |
| c | 0.182 | 0.00182 | 1.615 |
| rust | 0.201 | 0.00201 | 1.430 |

Speedups vs python (compute medians): c ≈ 120x, rust ≈ 109x. Python's wall and
compute coincide because the in-process clock excludes only JSON writing,
while the ports' wall includes process startup, fixture load, and a ~39 MB
walk-data write. Full provenance in the run's `results.json`.

## Local checks

```sh
pytest -q
ruff check --isolated --select E4,E7,E9,F,I,UP --target-version py311 benchmarks \
    tests/test_bench_routing.py tests/test_benchmark_fixtures.py \
    tests/test_benchmark_reference.py tests/test_benchmark_harness.py
```

This is backend-only benchmarking and CI work: no rendered UI, visual state, or
interactive workflow changes, so browser/visual evidence is not applicable.
