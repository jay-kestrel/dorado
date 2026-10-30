"""Normalize bundled npz fixtures to uncompressed .npy arrays plus params.json.

The C and Rust ports consume these normalized bytes instead of parsing npz
archives (spec art_hRL3eAyC, "Interface contracts", locked decision 5): every
array is saved uncompressed as ``<name>.f64.npy`` or ``<name>.i64.npy`` in the
dtype the reference computes in, and every scalar parameter rides in one flat
``params.json``. Running the normalizer twice produces byte-identical files.

The scalar values per field follow the parameter set that reproduces the
seeded goldens in ``tests/test_examplecases.py`` (they are the merge criteria),
matching ``dorado/example_data/define_params.py`` plus the ``Particles``
defaults it leaves unset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Flat params.json schema shared with the native ports: scalar keys plus the
# "arrays" mapping of logical name to file name inside the fixture directory.
SCALARS = ("dx", "theta", "gamma", "diff_coeff", "dry_depth", "steepest_descent",
           "model", "seed", "Np_tracer", "iterations")

# DeltaRCM input has no discharge arrays, so the reference (and
# define_params.py before it) routes on zeros; those zeros are baked into the
# fixture so every implementation reads the same bytes.
FIELDS: dict[str, dict] = {
    "rcm": {
        "source": "ex_deltarcm_data.npz",
        "dx": 50.0, "theta": 1.0, "gamma": 0.05, "diff_coeff": 0.2,
        "dry_depth": 0.1, "steepest_descent": False, "model": "DeltaRCM",
        "seed": 0, "Np_tracer": 10000, "iterations": 100,
    },
    "anuga": {
        "source": "ex_anuga_data.npz",
        "dx": 10.0, "theta": 1.0, "gamma": 0.05, "diff_coeff": 0.2,
        "dry_depth": 0.1, "steepest_descent": False, "model": "Anuga",
        "seed": 0, "Np_tracer": 10000, "iterations": 100,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_depth(depth: np.ndarray) -> np.ndarray:
    """Replicate the Particles init input cleanup: NaN depths become zero."""
    cleaned = depth.astype(np.float64, copy=True)
    cleaned[np.isnan(cleaned)] = 0
    return cleaned


def estimate_cell_type(cleaned_depth: np.ndarray, dry_depth: float) -> np.ndarray:
    """Mirror the Particles init estimate: 2 where dry, -1 on the border."""
    cell_type = np.zeros_like(cleaned_depth, dtype="int")
    cell_type[cleaned_depth < dry_depth] = 2
    padded = np.pad(cell_type[1:-1, 1:-1], 1, "constant", constant_values=-1)
    return padded.astype(np.int64)


def build_fixture(name: str, source_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Return the normalized arrays and flat manifest for one field."""
    spec = FIELDS[name]
    with np.load(source_dir / spec["source"]) as data:
        depth = data["depth"].astype(np.float64)
        if name == "rcm":
            stage = data["stage"].astype(np.float64)
            qx = np.zeros_like(depth)
            qy = np.zeros_like(depth)
        else:
            # anuga has no stage array; the reference routes on a stage copy
            # of depth (dorado/example_data/define_params.py).
            stage = depth.copy()
            qx = data["qx"].astype(np.float64)
            qy = data["qy"].astype(np.float64)

    arrays = {
        "depth": depth,
        "stage": stage,
        "qx": qx,
        "qy": qy,
        "cell_type": estimate_cell_type(clean_depth(depth), spec["dry_depth"]),
    }
    manifest = {key: spec[key] for key in SCALARS}
    manifest["arrays"] = {
        key: f"{key}.{'i64' if arr.dtype == np.int64 else 'f64'}.npy"
        for key, arr in arrays.items()
    }
    return arrays, manifest


def write_fixture(out_dir: Path, arrays: dict[str, np.ndarray], manifest: dict) -> list[Path]:
    """Write one fixture directory; np.save output is uncompressed .npy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, arr in arrays.items():
        target = out_dir / manifest["arrays"][name]
        np.save(target, arr)
        written.append(target)
    # write_bytes, not write_text: no platform newline translation, so the
    # committed bytes match a fresh normalization on every CI operating system.
    params_path = out_dir / "params.json"
    params_path.write_bytes((json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode())
    written.append(params_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("rcm", "anuga", "all"), default="all")
    parser.add_argument("--out-root", type=Path, default=ROOT / "bench",
                        help="directory that receives fixture_<name>/ subdirectories")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "dorado" / "example_data")
    args = parser.parse_args(argv)

    names = ["rcm", "anuga"] if args.fixture == "all" else [args.fixture]
    for name in names:
        arrays, manifest = build_fixture(name, args.source_dir)
        out_dir = args.out_root / f"fixture_{name}"
        written = write_fixture(out_dir, arrays, manifest)
        source = args.source_dir / FIELDS[name]["source"]
        print(f"{name}: source {source.name} sha256={sha256(source)}")
        for path in sorted(written):
            print(f"  wrote {path.relative_to(args.out_root)} sha256={sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
