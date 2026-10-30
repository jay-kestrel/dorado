"""Packaging tests: the built wheel must ship the documented example data.

The documented quickstart is ``from dorado.example_data import define_params``
followed by loading the bundled ``.npz`` archives, so the wheel has to contain
the ``dorado.example_data`` subpackage and its data files.
"""
import subprocess
import sys
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
def wheel_names(tmp_path_factory):
    """Build the wheel once for the module and return its file list."""
    if REPO_ROOT is None:
        pytest.skip("tests are not running from a package source tree")
    out_dir = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-build-isolation", "-w", str(out_dir), str(REPO_ROOT)],
        check=True, capture_output=True,
    )
    wheel = next(out_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


def test_wheel_ships_example_data(wheel_names):
    for name in EXPECTED_EXAMPLE_DATA:
        assert name in wheel_names


def test_wheel_ships_dorado_modules(wheel_names):
    for module in ("particle_track", "lagrangian_walker", "routines"):
        assert f"dorado/{module}.py" in wheel_names
