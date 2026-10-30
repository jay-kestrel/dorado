"""Packaging tests: the built wheel must ship the documented example data.

The documented quickstart is ``from dorado.example_data import define_params``
followed by loading the bundled ``.npz`` archives, so the wheel has to contain
the ``dorado.example_data`` subpackage and its data files.
"""
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

# source tree root, if the tests are running from a checkout
REPO_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "pyproject.toml").is_file():
        REPO_ROOT = _parent
        break

EXPECTED_EXAMPLE_DATA = [
    "dorado/example_data/__init__.py",
    "dorado/example_data/define_params.py",
    "dorado/example_data/ex_anuga_data.npz",
    "dorado/example_data/ex_deltarcm_data.npz",
]


def test_example_data_import_smoke():
    # the documented quickstart import, with the loader it documents
    from dorado.example_data import define_params

    assert callable(define_params.make_rcm_particles)


@pytest.fixture(scope="module")
def wheel_path(tmp_path_factory):
    """Build the wheel once for the module and return its path."""
    if REPO_ROOT is None:
        pytest.skip("tests are not running from a package source tree")
    out_dir = tmp_path_factory.mktemp("wheel")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "-w", str(out_dir), str(REPO_ROOT)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.fail(f"pip wheel failed:\n{proc.stdout[-1000:]}\n{proc.stderr[-1000:]}")
    return next(out_dir.glob("*.whl"))


@pytest.fixture(scope="module")
def wheel_names(wheel_path):
    with zipfile.ZipFile(wheel_path) as zf:
        return zf.namelist()


def test_wheel_ships_example_data(wheel_names):
    for name in EXPECTED_EXAMPLE_DATA:
        assert name in wheel_names


def test_wheel_ships_dorado_modules(wheel_names):
    for module in ("particle_track", "lagrangian_walker", "routines"):
        assert f"dorado/{module}.py" in wheel_names


def test_quickstart_runs_from_installed_wheel(wheel_path, tmp_path):
    """F7: the documented quickstart must work from the INSTALLED wheel,
    not just the source tree -- CI only ever exercised ``pip install -e .``,
    which cannot see a packaging defect. The install is hermetic
    (--no-deps into an isolated --target; runtime dependencies resolve from
    the running environment) and the quickstart runs outside the checkout
    so ``import dorado`` can only resolve from the installed copy."""
    target = tmp_path / "site"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps",
         "--target", str(target), str(wheel_path)],
        capture_output=True, text=True, check=True)

    # the documented quickstart (docs/source/pyplots/quickstart/demo1.py):
    # example-data import plus a short steady run, headless
    quickstart = textwrap.dedent("""
        import matplotlib
        matplotlib.use('Agg')
        import numpy as np
        from dorado import routines
        from dorado.example_data.define_params import make_rcm_particles

        np.random.seed(1)  # fix random seed so example is always the same
        particles = make_rcm_particles()
        walk = routines.steady_plots(particles, 2, save_output=False)
        assert len(walk['xinds']) == particles.Np_tracer == 50
    """)
    proc = subprocess.run(
        [sys.executable, "-c", quickstart],
        capture_output=True, text=True,
        cwd=tmp_path,  # keep the source tree off sys.path
        env={**os.environ, "PYTHONPATH": str(target)},
    )
    assert proc.returncode == 0, proc.stderr
