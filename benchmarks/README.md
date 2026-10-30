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

## Local checks

```sh
pytest -q
ruff check --isolated --select E4,E7,E9,F,I,UP --target-version py311 benchmarks tests/test_bench_routing.py
```

This is backend-only benchmarking and CI work: no rendered UI, visual state, or
interactive workflow changes, so browser/visual evidence is not applicable.
