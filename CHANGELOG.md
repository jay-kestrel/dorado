# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file begins at version 2.7.1, the package version carried by this repository's `pyproject.toml` at the start of the fork's changelog. Earlier release history belongs to the upstream [`passaH2O/dorado`](https://github.com/passaH2O/dorado) project.

## [2.7.1] - 2026-10-30

### Added

- Routing benchmark harness under `benchmarks/`: reproducible workloads over the bundled ANUGA and DeltaRCM fields, a documented timing contract, and a `--quick` mode that runs in CI with its JSON result uploaded as an artifact ([#1](https://github.com/jay-kestrel/dorado/pull/1)).
- `DORADO_LOG_LEVEL` environment variable configuring the `dorado` logger at import time, plus the `dorado.setup_logging` API accepting a level name and an optional custom handler ([#6](https://github.com/jay-kestrel/dorado/pull/6)).
- Run-level logging in the core simulation modules: defaulting notices while a `Particles` object is constructed and per-walk run start/complete summaries from `lagrangian_walker` ([#8](https://github.com/jay-kestrel/dorado/pull/8)).

### Fixed

- Core particle-routing defects, with regression tests ([#5](https://github.com/jay-kestrel/dorado/pull/5)).
