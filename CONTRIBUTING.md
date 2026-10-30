# Contributing to dorado

Thanks for contributing! This guide covers the development setup and conventions for this repository (`jay-kestrel/dorado`). For the contribution philosophy — how to open issues, propose features, and structure pull requests — see the [contributing page in the documentation](docs/source/misc/contributing.rst).

## About this fork

This repository is a fork of [`passaH2O/dorado`](https://github.com/passaH2O/dorado), the upstream project where dorado was developed and published. Open issues and pull requests **here**, against this fork's `master` branch — not upstream. The fork's continuous integration builds and tests every pull request on Linux, macOS, and Windows across the supported Python versions, and the upstream project's rendered documentation remains available at [passah2o.github.io/dorado](https://passah2o.github.io/dorado).

## Development setup

dorado supports Python 3.11+. From a clone of this repository:

```sh
pip install -e .[dev]
```

This installs the package in editable mode with the development dependencies: `pytest`, `pytest-cov`, `coveralls`, and the documentation toolchain (`sphinx`, `m2r2`, `ipython`, `ipykernel`).

## Running the tests

The test suite lives in `tests/`:

```sh
pytest
```

Coverage (as CI runs it):

```sh
pytest --cov=dorado/ --cov-report=xml
```

No `conftest.py` juggling is needed — `pyproject.toml` sets pytest's `pythonpath` so the source tree is importable from the repository root.

## Building the documentation

The Sphinx sources live in `docs/source/`:

```sh
cd docs
make html
```

The rendered site lands in `docs/build/html/`. A plain HTML build does not require a LaTeX toolchain, even though the pages include matplotlib-generated figures (the `plot` directive in the quickstart executes `docs/source/pyplots/*.py` at build time).

Two things to keep in mind while editing docs:

- Every new page must be reachable from a `toctree`, or Sphinx warns about an orphan document.
- The docs CI job treats warnings as pass-through, so broken references will not fail CI by themselves — run the build locally and check its output.

## Benchmarks

The routing kernel is benchmarked with the harness in `benchmarks/`. The quick mode is what CI runs on every pull request:

```sh
python benchmarks/bench_routing.py --quick --output /tmp/benchmark-quick.json
```

Workloads, the timing contract, and the full run are documented in [benchmarks/README.md](benchmarks/README.md).

## Commit and merge conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `build:`, `test:`, `refactor:`, optionally scoped — e.g. `docs(apiref): ...`, `fix(routing): ...`.
- Pull requests are squash-merged into `master` with a linear history.
- Please make sure the test suite and the docs build pass locally before opening a pull request; CI will run the full matrix (Linux/macOS/Windows × Python 3.11/3.12), the examples, and the docs build on your PR.

## License

By contributing, you agree that your contributions will be licensed under the repository's MIT license (see [LICENSE](LICENSE)).
