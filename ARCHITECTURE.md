# Architecture

This document describes `dorado` as built on `master` (version 2.7.1). It is the as-built companion to the theory in [`docs/source/background/index.rst`](docs/source/background/index.rst) and the per-module API reference in [`docs/source/apiref/index.rst`](docs/source/apiref/index.rst). `PLAN`-style design intent lives in the upstream project's publications; everything below describes what the code on this branch actually does.

## Overview

dorado routes passive tracer particles — optionally carrying per-particle kinds/sizes with modified transport behavior — across the flow field of any 2D shallow-water hydrodynamic model. It is a Python package (Python 3.11+) built on NumPy, with matplotlib for the plotting surfaces and SciPy for the spatial analysis statistics. An optional Rust/PyO3 routing-kernel extension (`dorado._native`) ships in the maturin-built wheels; source builds are pure-Python and `import dorado` works either way, with the backend switch itself not yet wired (the extension is scaffolding for spec D3).

The code is layered so that the stochastic routing kernel knows nothing about plotting, and the high-level conveniences never re-implement routing:

```
┌───────────────────────────────────────────────────────────────────┐
│ routines.py            high-level orchestration & plotting        │
│ spatial.py             spatial exposure-time analysis (post-hoc)  │
├───────────────────────────────────────────────────────────────────┤
│ parallel_routing.py    multiprocessing fan-out of the walk loop   │
├───────────────────────────────────────────────────────────────────┤
│ particle_track.py      modelParams + Particles: input validation, │
│                        defaults, seeding, walk_data bookkeeping   │
├───────────────────────────────────────────────────────────────────┤
│ lagrangian_walker.py   the routing kernel: neighbor weights, move │
│                        selection, boundaries, travel times        │
├───────────────────────────────────────────────────────────────────┤
│ logging_config.py      package logger, DORADO_LOG_LEVEL, verbose  │
│                        deprecation bridge                         │
├───────────────────────────────────────────────────────────────────┤
│ cli.py                 pydorado console entry point (info,        │
│                        validate, demo)                            │
└───────────────────────────────────────────────────────────────────┘
```

### Module map

| Module | Responsibility |
| --- | --- |
| `dorado/particle_track.py` | `modelParams` (the user-populated grid/flow attribute holder) and `Particles` (validation, defaulting, particle generation, per-iteration routing, `walk_data` bookkeeping). Also module-level helpers: coordinate transforms (`coord2ind`, `ind2coord`, `unstruct2grid`), exposure-time and nourishment calculations, seeding input validation (`gen_input_check`), and `flux_proportional_seeding`. |
| `dorado/lagrangian_walker.py` | The routing kernel: neighbor organization (`big_sliding_window`, `tile_local_array`, `tile_domain_array`, `clear_borders`), weight construction (`make_weight`, `get_weight`), move selection (`random_pick`, `steep_descent`), index/distance bookkeeping (`calculate_new_ind`, `step_update`), travel times (`calc_travel_times`), boundary handling (`check_for_boundary`), and the stepping loops (`particle_stepper`, `run_walk`). |
| `dorado/routines.py` | High-level orchestration and plotting: `steady_plots`, `unsteady_plots`, `time_plots`, `get_state`, `get_time_state`, `plot_exposure_time`, `plot_exposure_time_thresholds`, `animate_plots`, `draw_travel_path`, `plot_state`, `snake_plots`, and the nourishment display helpers. |
| `dorado/parallel_routing.py` | `parallel_routing(particles, num_iter, ...)`: splits routing across a `multiprocessing.Pool` via the `run_iter` wrapper and the `parallel_obj` attribute carrier, then merges per-worker dictionaries with `combine_result`. |
| `dorado/spatial.py` | Spatial exposure-time analysis over `walk_data`: `systemwide` and `localized` chunked exposure maps, `compute_thresholds` (KDE-based), `percentile_from_cdf`, `median_absolute_deviation_filter`, and `plot_spatial`. |
| `dorado/logging_config.py` | The `dorado` package logger, `DORADO_LOG_LEVEL` environment handling (`configure_from_env`, run at import), `setup_logging`, and `handle_verbose_deprecation`. |
| `dorado/cli.py` | The `pydorado` console entry point (registered in `pyproject.toml` as `[project.scripts]`): `info` (dependency/example-data inventory without importing the routing stack), `validate` (map an `.npz` onto `modelParams` and run it through `Particles`), and `demo` (short seeded routing run). Stdlib argparse only; heavy imports are deferred into the subcommand handlers. |
| `dorado/example_data/define_params.py` | Builds ready-to-run `modelParams`/`Particles` from the bundled ANUGA and DeltaRCM example fields (`ex_anuga_data.npz`, `ex_deltarcm_data.npz`). |

## Simulation lifecycle

1. **Populate `modelParams`.** The user assigns domain attributes: `dx` plus water `depth` (or `stage` and `topography`, from which depth is derived), and flow fields as either discharge (`qx`/`qy`) or velocity (`u`/`v`). Tuning parameters (`theta`, `gamma`, `diff_coeff`, `dry_depth`, `cell_type`, `steepest_descent`) are optional.

2. **Validate and default in `Particles(params)`.** Construction copies and normalizes the inputs: `NaN`s in depth/discharge are zeroed, missing arrays are derived (`depth = stage - topography`, clamped at 0; `stage = topography + depth`), a velocity-magnitude field is computed and floored at `1e-8` (with sign-preserving handling of tiny components so velocity angles keep pointing along the flow), and `velocity_angle = arctan2(-u, v)` is precomputed. Required-but-missing inputs raise `ValueError`. Optional parameters receive defaults — `theta = 1.0`, `gamma = 0.05` (out-of-range values raise `ValueError`; `gamma` must satisfy 0 ≤ γ ≤ 1), `diff_coeff = 0.2` (disabled, i.e. `0.0`, for steepest descent), `dry_depth = 0.1` m, `cell_type` estimated from the depth field with `-1` borders, `steepest_descent = False` — and each default is logged at INFO. Routing weights for the whole domain are built once here (`lw.make_weight(self)`). If `roi_grid` is provided (shape-validated against the depth array), an `roi_flag` tracked variable is registered.

3. **Seed particles.** `Particles.generate_particles(Np_tracer, seed_xloc, seed_yloc, ...)` validates its inputs (`gen_input_check`) and the seed locations against the domain bounds (out-of-bounds seeds raise rather than wrap around via negative indexing), then places particles either `'random'`ly across the seed lists or `'exact'`ly at given `(x, y)` points. Successive calls append to the same `walk_data`; `previous_walk_data` folds in independent walk data (e.g. loaded from JSON).

4. **Route.** `Particles.run_iteration(target_time=None, max_iter=10_000)` advances all particles one iteration (or loops per-particle toward `target_time`, capped by `max_iter`), delegating the movement to `lw.particle_stepper` and appending results to `walk_data`.

5. **Consume.** `routines` wraps steps 3–4 with plotting and saving; `parallel_routing` fans out `num_iter` iterations across CPU cores; `spatial` analyzes the resulting `walk_data`; `exposure_time`, `nourishment_area`, and `nourishment_time` answer targeted questions about it.

### The `walk_data` contract

`Particles.walk_data` is a dictionary mapping field names to per-particle, per-iteration histories:

- Keys: `'xinds'`, `'yinds'`, `'travel_times'` — the base fields — plus any tracked optional variables (from `modelParams.particle_variables`, e.g. sampled `'depth'`/`'stage'`/`'qx'`/`'qy'`/`'u'`/`'v'` values or the `'roi_flag'` of a provided `roi_grid`), plus the persistent per-particle attribute keys `'particle_size'`, `'particle_type'`, and the derived `'deposited'` flag when particles were generated with a size and/or kind (batches generated without an attribute carry `None` entries so histories stay aligned across merged batches), plus `'stranded'` and `'stranded_time'` when `stranding_handling = 'park'`.
- Indexing: `walk_data[field][particle][iteration]`, e.g. `walk_data['travel_times'][5][10]` is the travel time of the 5th particle at the 10th iteration.
- Positions are stored as grid indices (`xinds`/`yinds`); the `coord2ind`/`ind2coord` helpers convert to and from real coordinates.
- A frozen particle (boundary handling, below) appends nothing for that iteration — its history simply does not grow.
- `run_iteration` returns the same dictionary it stores; `clear_walk_data()` resets it to `None`.

## The routing kernel

`lagrangian_walker.py` implements the weighted random walk:

- **Neighborhood organization.** `big_sliding_window` produces an `(L, W, 9)` view of every cell's 3×3 neighborhood ordered `[NW, N, NE, W, center, E, SW, S, SE]` to match `np.ravel` order; `tile_local_array`/`tile_domain_array` replicate local and domain-wide arrays across the domain and `clear_borders` trims the tiled artifacts, together enabling vectorized weight assembly.
- **Weights.** `make_weight` builds the domain-wide weight field by blending the water-surface-gradient and inertial (discharge) routing components with `gamma`, sharpening by local depth via `theta`, masking dry, land, and border cells, and falling back to the deepest neighbor where everything else is masked. The formulas are the ones derived on the background theory page. `get_weight` extracts a particle's 9-cell weight vector from the prebuilt field.
- **Move selection.** `random_pick` draws the next cell by inverse-CDF over the neighbor weights (`np.random.uniform` against `np.cumsum`). `steep_descent` instead picks the maximum-weight neighbor — excluding the current cell, breaking ties randomly, and staying put at a dead end. Which of the two applies is decided by `modelParams.steepest_descent`. NaN weights are treated as zero in both, and both work on copies so the shared weight field is never mutated.
- **Boundaries.** `check_for_boundary` cancels the proposed move of any particle whose current location borders an edge cell (`cell_type == -1`), freezing it for the iteration; lower slice bounds are clamped at 0 so edge particles check a real neighborhood instead of wrapping around. Independently, `particle_stepper` freezes any particle whose proposed step would leave the grid entirely — when the border cells are water there is no `-1` edge for the boundary check to react to, and an unchecked step would walk off-grid and silently wrap through downstream index lookups.
- **Stranding (park-and-release).** Under the default `stranding_handling = 'park'`, a particle whose current cell is dry (`depth ≤ dry_depth`) or permanent land (`cell_type == 2`) parks: it does not consult the routing weights, stays in place, and books `stranding_dt` of travel time with its `'stranded'` walk_data flag set and its `'stranded_time'` total advanced. Release is only ever evaluated on a wet cell and governed by `stranding_release`: `'immediate'` releases on the first wet step, `'rest'` holds until the accumulated stranded time reaches `rest_time`, `'probability'` runs a per-step Bernoulli re-entrainment trial with probability `entrain_prob`. Setting `stranding_handling = 'none'` restores the exact legacy behavior with no stranded variables tracked (walks carried over from `'none'` are zero-padded). Under `'park'`, the deepest-cell fallback is dry-aware: permanent land is not a fallback destination while any non-land cell exists in the window. The stranding theory is on the background page's [stranding section](docs/source/background/index.rst).
- **Travel times.** `calc_travel_times` computes `0.5 * projected_dist * (1/v_old + 1/v_new) * (1 + diff)`, where `projected_dist` is the step distance projected onto the mean-flow orientation (clamped at zero — backward or orthogonal steps are instantaneous) and `diff` is a uniform draw in `[-diff_coeff/2, +diff_coeff/2]` re-introducing the diffusivity lost to the gridded scheme. A particle that does not move accrues no time.
- **Per-particle behavior modifiers.** Particles can carry a numeric `particle_size` and/or a named `particle_type` (a *kind*, a key of `KIND_PRESETS` in `lagrangian_walker.py` — shipped presets: `sand`, `mud`, `plankton`). Each step, the kind's `travel_time_multiplier` scales the computed travel time, its `diffusion_multiplier` scales the diffusion term, and its `deposition_probability` is the per-step chance the particle settles permanently (its `'deposited'` walk_data flag flips to 1 and it stops moving). A size without a kind scales travel time by the size and diffusion by `1/size`; size and kind multipliers multiply when both are given. All particles in one `Particles` object share the routing weight field — heterogeneity comes from these per-particle modifiers, not per-kind weight fields. Without `particle_size`/`particle_type` in any `generate_particles` call, behavior is exactly the passive-tracer random walk; see the [particle kinds and sizes guide](docs/source/userguide/particle_sizes_types.rst).
- **Stepping loops.** `particle_stepper` runs one iteration for every particle and returns new indices, travel times, and updated tracked variables; `run_walk` loops it for a fixed number of iterations and is what the parallel wrapper calls.

## Randomness and reproducibility

All stochasticity — random seed placement, neighbor selection, and travel-time diffusion — draws from NumPy's global random state (`np.random`). `dorado` never seeds or restores it itself; for reproducible runs, seed the global RNG (e.g. `np.random.seed(...)`) before generating and routing particles, as the test suite and benchmark harness do. Because the global state is shared, concurrently executing walks (e.g. `parallel_routing` workers) are not independent random streams unless the caller arranges per-process seeding.

## Parallel routing

`parallel_routing` maps `run_iter` over a `multiprocessing.Pool`: each worker receives a `parallel_obj` carrying the parameters, seed lists, and iteration count, generates its own particles, and runs `run_walk` for its share of the iterations. `combine_result` merges the per-worker `walk_data` dictionaries into one. Since workers each seed and walk their own particle set, the result is the union of independent batches — see the randomness note above for the reproducibility implications.

## Logging

`dorado.logging_config` owns logging:

- The package logger is `logging.getLogger("dorado")`, re-exported as `dorado.logger`; submodule loggers (`dorado.lagrangian_walker`, `dorado.parallel_routing`, `dorado.spatial`) derive from it through the standard hierarchy.
- At import time, `configure_from_env()` reads `DORADO_LOG_LEVEL`. Unset, the logger is untouched and Python's `lastResort` handler surfaces WARNING-and-above records to stderr. Set to a valid level name, the level is applied and a stderr `StreamHandler` (format `%(name)s - %(levelname)s - %(message)s`) is attached. An invalid value logs a warning and falls back to WARNING; the import never fails.
- `setup_logging(level, handler=None)` is the programmatic API, accepting a custom handler (e.g. a file handler).
- The deprecated `verbose` parameter of `modelParams` is bridged by `handle_verbose_deprecation`, a non-persistent shim: `verbose=True` (or `None`) raises the logger to INFO for the legacy run, `verbose=False` leaves the configured level in place, and a `DeprecationWarning` is emitted; prefer `setup_logging`.

What is logged at INFO by default: defaulting notices during `Particles` construction (theta, gamma, diffusion coefficient, dry depth, cell types, routing mode), ROI registration, and per-walk run start/complete summaries from `lagrangian_walker`.

## Benchmarks

`benchmarks/bench_routing.py` times `Particles.run_iteration` — including its walk-data bookkeeping, excluding plotting, data loading, weight construction, and seeding — over the two bundled example fields at several particle counts, normalizing by attempted particle-steps and recording full provenance (field hashes, source revision, versions). `--quick` mode (ANUGA, 500 particles, 3 iterations, 1 repeat) runs in CI's `examples` job with the JSON result uploaded as an artifact. The timing contract, workloads, and recorded baselines live in [`benchmarks/README.md`](benchmarks/README.md). The harness stages particles through the public API without modifying library code, so alternative routing backends can be measured against the Python kernel by extending the harness alone.
