# dorado - Lagrangian particle routing
![build](https://github.com/passaH2O/dorado/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/passaH2O/dorado/branch/master/graph/badge.svg?token=A4MWN4K1XJ)](https://codecov.io/gh/passaH2O/dorado)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pydorado)
[![PyPI version](https://badge.fury.io/py/pydorado.svg)](https://badge.fury.io/py/pydorado)
[![status](https://joss.theoj.org/papers/f1f61292f998588ae06bb1cd14dd4063/status.svg)](https://joss.theoj.org/papers/f1f61292f998588ae06bb1cd14dd4063)
<div class="nav3" style="height:705px;">
    <img src="docs/source/examples/images/logo.gif" alt="Particle routing on Lidar-derived bathymetry" width="65%"></a>
</div>

dorado is a Python package for simulating passive Lagrangian particle transport over flow-fields from any 2D shallow-water hydrodynamic model using a weighted random walk methodology.

For user guides and detailed examples, refer to the [documentation](https://passah2o.github.io/dorado/index.html).

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
dorado supports Python 3.11+. For the full distribution including examples, clone this repository using `git clone` and run `python setup.py install` from the cloned directory. To test this "full" installation, you must first install `pytest` via `pip install pytest`. Then from the cloned directory the command `pytest` can be run to ensure that your installed distribution passes all of the unit tests.

For a lightweight distribution including just the core functionality, use `pip` to install via PyPI:

    pip install pydorado

Installation using `conda` via `conda-forge` is also supported:

    conda install -c conda-forge pydorado

For additional installation options and instructions, refer to the [documentation](https://passah2o.github.io/dorado/install/index.html).

### Configuring log output
dorado logs through the standard `logging` module under the `dorado` logger. The `DORADO_LOG_LEVEL` environment variable controls log visibility at import time:

- **Unset** (default): the package attaches no handler and leaves the logger at Python's default level, so WARNING-and-above records (warnings and errors) still reach stderr while informational narration stays silent.
- **Set** to `DEBUG`, `INFO`, `WARNING`, or `ERROR`: the `dorado` logger is configured at that level with a handler writing to stderr. For example:

        DORADO_LOG_LEVEL=INFO python my_script.py

An invalid value does not fail the import — dorado logs a warning naming the bad value and falls back to `WARNING`.

Logging can also be configured programmatically at any time:

    import dorado
    dorado.setup_logging("INFO")

The deprecated `verbose` parameter on dorado's routines is still accepted, but new code should configure logging with `DORADO_LOG_LEVEL` or `setup_logging()` instead.

## Contributing
We welcome contributions to the dorado project. Please open an issue or a pull request if there is functionality you would like to see or propose. Refer to our [contributing guide](https://passah2o.github.io/dorado/misc/contributing.html) for more information.

## Citing
If you use this package and wish to cite it, please use the [Journal of Open Source Software article](https://joss.theoj.org/papers/10.21105/joss.02585#).

## Funding Acknowledgments
This work was supported in part by NSF EAR-1719670, the NSF GRFP under grant DGE-1610403 and the NASA Earth Venture Suborbital (EVS) award 17-EVS3-17_1-0009 in support of the DELTA-X project.
