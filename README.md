# dorado - Lagrangian particle routing

![build](https://github.com/jay-kestrel/dorado/actions/workflows/build.yml/badge.svg)
[![codecov](https://codecov.io/gh/jay-kestrel/dorado/branch/master/graph/badge.svg)](https://codecov.io/gh/jay-kestrel/dorado)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pydorado)
[![PyPI version](https://badge.fury.io/py/pydorado.svg)](https://badge.fury.io/py/pydorado)
[![status](https://joss.theoj.org/papers/f1f61292f998588ae06bb1cd14dd4063/status.svg)](https://joss.theoj.org/papers/f1f61292f998588ae06bb1cd14dd4063)
<div class="nav3" style="height:705px;">
    <img src="docs/source/examples/images/logo.gif" alt="Particle routing on Lidar-derived bathymetry" width="65%"></a>
</div>

dorado is a Python package for simulating passive Lagrangian particle transport over flow-fields from any 2D shallow-water hydrodynamic model using a weighted random walk methodology.

**About this repository.** This is the `jay-kestrel/dorado` fork of [`passaH2O/dorado`](https://github.com/passaH2O/dorado). The packages published to [PyPI](https://pypi.org/project/pydorado/) and [conda-forge](https://anaconda.org/conda-forge/pydorado) track the upstream project's releases; building from this repository gives you the current state of this fork. The build and codecov badges above report this fork's builds and coverage.

User guides and detailed examples live in the Sphinx documentation under [`docs/source/`](docs/source/index.rst), which is built on every pull request. The upstream project also [renders this documentation online](https://passah2o.github.io/dorado/index.html).

### Example Uses:

### Particles on an Unsteady ANUGA Flow Field of the Wax Lake Delta
<div class="nav3" style="height:705px;">
    <img src="docs/source/examples/images/waxlake.gif" alt="Example" width="75%"></a>
</div>

### Particles on a DeltaRCM Simulated Delta
<div class="nav3" style="height:705px;">
    <img src="docs/source/examples/images/example02/steady_deltarcm.gif" alt="Example" width="75%"></a>
</div>

## Installation:

dorado supports Python 3.11+.

For a lightweight distribution including just the core functionality, use `pip` to install via PyPI:

    pip install pydorado

Installation using `conda` via `conda-forge` is also supported:

    conda install -c conda-forge pydorado

To install from this repository — including the examples, tests, and benchmark harness — clone it and install locally:

    git clone https://github.com/jay-kestrel/dorado
    cd dorado
    pip install .

For an editable installation, where modifications to the code are picked up when you import and run scripts, use:

    pip install -e .

To include the development dependencies (pytest, pytest-cov, and the documentation toolchain) as well, install with `pip install -e .[dev]` instead.

To test your installation, first install `pytest` via `pip install pytest`. Then from the cloned directory the command `pytest` can be run to ensure that your installed distribution passes all of the unit tests.

For additional installation options and instructions, refer to the [installation documentation](docs/source/install/index.rst).

## Logging

`dorado` logs through the standard `logging` module on the `dorado` logger; out of the box, warnings and above are surfaced to stderr. Two ways to turn the level up:

- **Environment variable:** set `DORADO_LOG_LEVEL` to one of `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` before importing `dorado`:

      DORADO_LOG_LEVEL=INFO python my_script.py

- **Programmatically:** call `dorado.setup_logging("INFO")`, optionally passing a custom `logging.Handler` (for example `logging.FileHandler("dorado.log")`).

An invalid `DORADO_LOG_LEVEL` value does not fail the import — `dorado` logs a warning naming the bad value and falls back to `WARNING`. The legacy `verbose` parameter is deprecated and maps onto the logger; prefer the options above. See the [logging guide](docs/source/userguide/index.rst) and the [`dorado.logging_config` API reference](docs/source/apiref/loggingconfig.rst) for details.

## Command-line interface

Installing the package exposes a `pydorado` console command with three subcommands built on the standard library only:

- `pydorado info` — package, Python, and dependency versions plus the bundled example-data files; it never imports the routing code.
- `pydorado validate` — map the arrays in an `.npz` file onto `dorado.particle_track.modelParams` following the example-data conventions and run them through `Particles` as the validator.
- `pydorado demo` — a short seeded routing run on a bundled example field.

## Benchmarks

The routing kernel is timed by the harness in [`benchmarks/`](benchmarks/README.md), whose `--quick` mode runs in CI on every pull request. See the harness README for the workloads, the timing contract, and how to run the full benchmark.

## Contributing

We welcome contributions to the dorado project. Please open an issue or a pull request in this repository, and see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and conventions. The upstream project's contributing guide remains available [here](https://passah2o.github.io/dorado/misc/contributing.html).

## Citing

If you use this package and wish to cite it, please use the [Journal of Open Source Software article](https://joss.theoj.org/papers/10.21105/joss.02585#).

## Funding Acknowledgments

This work was supported in part by NSF EAR-1719670, the NSF GRFP under grant DGE-1610403 and the NASA Earth Venture Suborbital (EVS) award 17-EVS3-17_1-0009 in support of the DELTA-X project.
