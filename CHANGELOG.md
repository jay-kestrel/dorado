# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file begins at version 2.7.1, the package version carried by this repository's `pyproject.toml` at the start of the fork's changelog. Earlier release history belongs to the upstream [`passaH2O/dorado`](https://github.com/passaH2O/dorado) project.

## [2.7.1] - 2026-10-30

### Added

- Routing benchmark harness under `benchmarks/`: reproducible workloads over the bundled ANUGA and DeltaRCM fields, a documented timing contract, and a `--quick` mode that runs in CI with its JSON result uploaded as an artifact ([#1](https://github.com/jay-kestrel/dorado/pull/1)).
- `DORADO_LOG_LEVEL` environment variable configuring the `dorado` logger at import time, plus the `dorado.setup_logging` API accepting a level name and an optional custom handler ([#6](https://github.com/jay-kestrel/dorado/pull/6)).
- Run-level logging in the core simulation modules: defaulting notices while a `Particles` object is constructed and per-walk run start/complete summaries from `lagrangian_walker` ([#8](https://github.com/jay-kestrel/dorado/pull/8)).
- The `pydorado` console command (`pydorado = "dorado.cli:main"`) with three standard-library-only subcommands: `info` (dependency and example-data inventory), `validate` (map an `.npz` onto `modelParams` and validate through `Particles`), and `demo` (short seeded routing run) ([#10](https://github.com/jay-kestrel/dorado/pull/10)).
- Per-particle kinds and sizes with size-dependent transport behavior: `generate_particles` accepts `particle_size` and `particle_type` arguments; the shipped `KIND_PRESETS` kinds (`sand`, `mud`, `plankton`) scale travel time, diffusion, and per-step deposition; the attributes persist in `walk_data` under `particle_size`, `particle_type`, and the `deposited` flag. Particles without these attributes remain identical passive tracers.
- Stranded-particle handling with park-and-release semantics: `stranding_handling = 'park'` (the new default) stops a particle on a dry or permanent-land cell — booking `stranding_dt` of model time per parked step and tracking `stranded`/`stranded_time` in `walk_data` — and releases it by `stranding_release` (`'immediate'`, `'rest'` with `rest_time`, or `'probability'` with `entrain_prob`) once its cell is wet again; `stranding_handling = 'none'` restores the exact legacy behavior. The deepest-cell weight fallback is dry-aware under `'park'`, and the stranding theory is documented on the background page ([#16](https://github.com/jay-kestrel/dorado/pull/16)).
- Rust crate scaffolding for the native routing backend: maturin build backend, an optional `dorado._native` PyO3 extension that ships in maturin-built wheels (source builds stay pure-Python and `import dorado` works either way), and a `wheels.yml` workflow building release wheels across platforms ([#2](https://github.com/jay-kestrel/dorado/pull/2)). The routing backend switch itself is not yet wired.

### Changed

- The deprecated `verbose` parameter shim is non-persistent: `verbose=True` (or `None`) raises the logger to INFO for the legacy run, `verbose=False` no longer forces ERROR and leaves the configured level in place ([#3](https://github.com/jay-kestrel/dorado/pull/3)).

### Fixed

- Core particle-routing defects, with regression tests ([#5](https://github.com/jay-kestrel/dorado/pull/5)).
- Packaging: the bundled example data ships in built distributions via corrected package-data keys, and the in-repo conda recipe builds again ([#3](https://github.com/jay-kestrel/dorado/pull/3)).
- Particle stepping freezes any particle whose proposed move would leave the domain instead of silently wrapping through index lookups, and out-of-range `gamma` values (outside `[0, 1]`) now raise `ValueError` rather than being clamped ([#11](https://github.com/jay-kestrel/dorado/pull/11)).
