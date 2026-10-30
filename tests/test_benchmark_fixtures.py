"""Fixture normalizer: byte stability, manifest completeness, init equivalence."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks import normalize_fixtures as nf

ROOT = Path(__file__).resolve().parents[1]
ARRAY_KEYS = {"depth", "stage", "qx", "qy", "cell_type"}


def normalize_tree(tmp: Path, root_name: str) -> Path:
    out_root = tmp / root_name
    assert nf.main(["--out-root", str(out_root)]) == 0
    return out_root


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("name", ["rcm", "anuga"])
def test_normalize_is_byte_stable(tmp_path, name):
    out_root = tmp_path / "first"
    assert nf.main(["--fixture", name, "--out-root", str(out_root)]) == 0
    assert nf.main(["--fixture", name, "--out-root", str(tmp_path / "second")]) == 0
    first = tree_bytes(out_root / f"fixture_{name}")
    second = tree_bytes(tmp_path / "second" / f"fixture_{name}")
    assert first and set(first) == set(second)
    assert first == second


@pytest.mark.parametrize("name", ["rcm", "anuga"])
def test_committed_fixtures_are_current(tmp_path, name):
    out_root = normalize_tree(tmp_path, "fresh")
    committed = ROOT / "bench" / f"fixture_{name}"
    assert tree_bytes(committed) == tree_bytes(out_root / f"fixture_{name}")


@pytest.mark.parametrize("name,shape,dx,model", [
    ("rcm", (120, 240), 50.0, "DeltaRCM"),
    ("anuga", (100, 100), 10.0, "Anuga"),
])
def test_manifest_lists_every_array(tmp_path, name, shape, dx, model):
    fixture = normalize_tree(tmp_path, "manifest") / f"fixture_{name}"
    manifest = json.loads((fixture / "params.json").read_text())
    assert set(manifest) == set(nf.SCALARS) | {"arrays"}
    assert set(manifest["arrays"]) == ARRAY_KEYS
    assert sorted(manifest["arrays"].values()) == \
        sorted(path.name for path in fixture.glob("*") if path.name != "params.json")
    for key, fname in manifest["arrays"].items():
        arr = np.load(fixture / fname)
        assert arr.shape == shape
        assert arr.dtype == (np.int64 if key == "cell_type" else np.float64)
    assert manifest["dx"] == dx
    assert manifest["model"] == model
    assert manifest["steepest_descent"] is False
    assert manifest["theta"] == 1.0
    assert manifest["gamma"] == 0.05
    assert manifest["diff_coeff"] == 0.2
    assert manifest["dry_depth"] == 0.1
    assert manifest["seed"] == 0
    assert manifest["Np_tracer"] == 10000
    assert manifest["iterations"] == 100


@pytest.mark.parametrize("name,source", [
    ("rcm", "ex_deltarcm_data.npz"), ("anuga", "ex_anuga_data.npz"),
])
def test_array_values_match_source(name, source):
    """Fixture arrays are the npz values cast to the reference dtype, unmodified."""
    fixture = ROOT / "bench" / f"fixture_{name}"
    with np.load(ROOT / "dorado" / "example_data" / source) as data:
        depth = np.load(fixture / "depth.f64.npy")
        assert np.array_equal(depth, data["depth"].astype(np.float64), equal_nan=True)
        stage = np.load(fixture / "stage.f64.npy")
        if name == "rcm":
            assert np.array_equal(stage, data["stage"].astype(np.float64))
            for key in ("qx", "qy"):
                assert not np.load(fixture / f"{key}.f64.npy").any()
        else:
            assert np.array_equal(stage, depth, equal_nan=True)
            for key in ("qx", "qy"):
                arr = np.load(fixture / f"{key}.f64.npy")
                assert np.array_equal(arr, data[key].astype(np.float64), equal_nan=True)


@pytest.mark.parametrize("name,source,dx,model", [
    ("rcm", "ex_deltarcm_data.npz", 50.0, "DeltaRCM"),
    ("anuga", "ex_anuga_data.npz", 10.0, "Anuga"),
])
def test_baked_cell_type_matches_particles_init(name, source, dx, model):
    """The fixture cell_type is exactly what Particles init would estimate."""
    from dorado.particle_track import Particles, modelParams

    with np.load(ROOT / "dorado" / "example_data" / source) as data:
        params = modelParams()
        params.depth = data["depth"].astype(np.float64)
        params.stage = data["stage"].astype(np.float64) if name == "rcm" \
            else params.depth.copy()
        params.qx = np.zeros_like(params.depth) if name == "rcm" \
            else data["qx"].astype(np.float64)
        params.qy = np.zeros_like(params.depth) if name == "rcm" \
            else data["qy"].astype(np.float64)
        params.dx = dx
        params.model = model
        particles = Particles(params)
    baked = np.load(ROOT / "bench" / f"fixture_{name}" / "cell_type.i64.npy")
    assert np.array_equal(np.asarray(particles.cell_type), baked)
