"""Command-line interface for pydorado.

Three subcommands built on stdlib argparse only:

- ``pydorado info`` — package/Python/dependency versions and the bundled
  example-data files with existence flags. Never imports the routing code.
- ``pydorado validate`` — map the arrays in an ``.npz`` file onto
  :obj:`dorado.particle_track.modelParams` following the conventions in
  ``dorado/example_data/define_params.py``, then run
  :obj:`dorado.particle_track.Particles` as the validator.
- ``pydorado demo`` — short seeded routing run on a bundled example field.

numpy and the dorado routing modules are imported inside the subcommand
handlers, so this module adds no heavy imports of its own and ``info``
stays independent of the routing stack.

``validate`` and ``demo`` always load fresh copies of the arrays:
``Particles()`` mutates the arrays it receives in place (NaN depths are
replaced with 0, dry-cell stage values are set to NaN), so a params object
must never alias arrays the caller still cares about.
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

DIST_NAME = "pydorado"
DEPENDENCIES = ("numpy", "scipy", "matplotlib", "tqdm")
EXAMPLE_DATA_FILES = {
    "rcm": "ex_deltarcm_data.npz",
    "anuga": "ex_anuga_data.npz",
}
# Seed cells follow dorado/example_data/define_params.py for the RCM field
# and benchmarks/bench_routing.py for the ANUGA field.
SEED_LOCATIONS = {
    "rcm": (list(range(15, 17)), list(range(137, 140))),
    "anuga": (list(range(20, 30)), list(range(48, 53))),
}
FIELD_MODELS = {"rcm": "DeltaRCM", "anuga": "Anuga"}


def _installed_version(distribution: str) -> str | None:
    """Version string for a distribution, or None if it is not installed."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _example_data_dir() -> Path | None:
    """Locate the bundled example_data directory without executing modules."""
    spec = find_spec("dorado")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "example_data"


def cmd_info(args: argparse.Namespace) -> int:
    """Report versions and bundled example-data availability."""
    del args  # info takes no options; uniform handler signature
    installed = _installed_version(DIST_NAME)
    if installed is None:
        print(f"{DIST_NAME}: distribution metadata not found (not installed?)")
    else:
        print(f"{DIST_NAME} {installed}")
    print(f"Python {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    deps = ", ".join(
        f"{name} {_installed_version(name) or 'not installed'}"
        for name in DEPENDENCIES)
    print(f"dependencies: {deps}")
    data_dir = _example_data_dir()
    if data_dir is None:
        print("example data: dorado package location not found")
    else:
        print(f"example data (bundled in {data_dir}):")
        for filename in EXAMPLE_DATA_FILES.values():
            path = data_dir / filename
            state = "present" if path.is_file() else "missing"
            print(f"  {path} [{state}]")
    return 0


def _required_arrays(model: str) -> tuple[str, ...]:
    """npz keys each supported model must provide."""
    if model == "DeltaRCM":
        return ("depth", "stage")
    return ("depth", "qx", "qy")


def _params_from_npz(path: Path, model: str, dx: float | None = None,
                     theta: float | None = None, gamma: float | None = None,
                     dry_depth: float | None = None):
    """Map npz arrays onto modelParams per example_data/define_params.py.

    DeltaRCM: depth and stage come from the file and zeros stand in for
    qx/qy (the weighting scheme still uses depth). Anuga/other: depth, qx,
    and qy come from the file and stage is depth.copy(). dx defaults to 50
    for DeltaRCM and 10 otherwise; theta defaults to 1.0; gamma and
    dry_depth fall through to the Particles() defaults when not given.

    Arrays are freshly loaded because ``Particles()`` mutates its inputs
    in place. Returns (params, [(array name, shape), ...]).
    """
    import numpy as np

    from dorado.particle_track import modelParams

    params = modelParams()
    with np.load(path) as data:
        available = set(data.files)
        missing = [name for name in _required_arrays(model)
                   if name not in available]
        if missing:
            raise ValueError(
                f"{path} lacks required array(s) for --model {model}: "
                + ", ".join(missing))
        params.depth = data["depth"]
        if model == "DeltaRCM":
            params.stage = data["stage"]
            # define_params.py convention: no discharge data in DeltaRCM
            # exports, zeros let weighting fall back to depth
            params.qx = np.zeros_like(params.depth)
            params.qy = np.zeros_like(params.depth)
        else:
            params.qx = data["qx"]
            params.qy = data["qy"]
            params.stage = params.depth.copy()
    params.dx = float(dx) if dx is not None else (50.0 if model == "DeltaRCM" else 10.0)
    params.theta = float(theta) if theta is not None else 1.0
    params.model = model
    if gamma is not None:
        params.gamma = float(gamma)
    if dry_depth is not None:
        params.dry_depth = float(dry_depth)
    shapes = [(name, tuple(getattr(params, name).shape))
              for name in ("depth", "stage", "qx", "qy")]
    return params, shapes


def _list_arrays(path: Path) -> int:
    """Print the arrays in an npz file with shapes and dtypes."""
    import numpy as np

    try:
        with np.load(path) as data:
            entries = [(name, data[name].shape, str(data[name].dtype))
                       for name in data.files]
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    print(f"arrays in {path}:")
    for name, shape, dtype in entries:
        print(f"  {name}: shape {shape}, dtype {dtype}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate an npz input by constructing Particles() from it."""
    if args.list_arrays:
        return _list_arrays(args.data)
    try:
        params, shapes = _params_from_npz(
            args.data, args.model, dx=args.dx, theta=args.theta,
            gamma=args.gamma, dry_depth=args.dry_depth)
        from dorado.particle_track import Particles
        # Particles.__init__ checks the mapped params and raises ValueError
        # on invalid inputs; the instance is discarded, nothing is routed.
        Particles(params)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {args.model} parameters from {args.data} "
          "accepted by Particles()")
    for name, shape in shapes:
        print(f"  {name}: shape {shape}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Short seeded routing run on a bundled example field."""
    import numpy as np

    from dorado.particle_track import Particles

    path = None
    data_dir = _example_data_dir()
    if data_dir is not None:
        path = data_dir / EXAMPLE_DATA_FILES[args.field]
    if path is None or not path.is_file():
        print(f"error: bundled example data not found "
              f"(expected {EXAMPLE_DATA_FILES[args.field]} in {data_dir})",
              file=sys.stderr)
        return 1
    seed_xloc, seed_yloc = SEED_LOCATIONS[args.field]
    try:
        params, _ = _params_from_npz(path, FIELD_MODELS[args.field])
        particles = Particles(params)
        np.random.seed(args.seed)  # repo convention: legacy global RNG
        particles.generate_particles(args.particles, seed_xloc, seed_yloc)
        for _ in range(args.steps):
            particles.run_iteration()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    walk = particles.walk_data
    final_times = [times[-1] for times in walk["travel_times"]]
    print(f"demo: field={args.field} particles={args.particles} "
          f"steps={args.steps} seed={args.seed}")
    print("final cell indices (x, y) per particle:")
    for xinds, yinds in zip(walk["xinds"], walk["yinds"]):
        print(f"  ({xinds[-1]}, {yinds[-1]})")
    times = ", ".join(f"{stat}={value:.6g}" for stat, value in (
        ("min", np.min(final_times)), ("mean", np.mean(final_times)),
        ("median", np.median(final_times)), ("max", np.max(final_times))))
    print(f"final travel times (seconds): {times}")
    return 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pydorado",
        description="Utilities for the pydorado particle-routing package.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser(
        "info",
        help="print package, Python, dependency, and example-data info")
    info_parser.set_defaults(func=cmd_info)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate an .npz input against Particles() without routing",
        description="Map npz arrays onto modelParams per "
                    "dorado/example_data/define_params.py and construct "
                    "Particles() as the validator. Arrays are freshly "
                    "loaded because Particles() mutates its inputs in "
                    "place. Nothing is written or routed.")
    validate_parser.add_argument(
        "--data", type=Path, required=True, help="path to the .npz file")
    validate_parser.add_argument(
        "--model", required=True, choices=("DeltaRCM", "Anuga", "other"),
        help="hydrodynamic model the arrays come from; controls how npz "
             "keys map onto modelParams ('other' maps like Anuga, whose "
             "non-DeltaRCM branch Particles() also uses)")
    validate_parser.add_argument(
        "--list-arrays", action="store_true",
        help="list the arrays in --data (name, shape, dtype) and exit")
    validate_parser.add_argument(
        "--dx", type=float,
        help="cell size; default 50 for DeltaRCM, 10 otherwise")
    validate_parser.add_argument(
        "--theta", type=float, help="depth weighting; default 1.0")
    validate_parser.add_argument(
        "--gamma", type=float, help="weight ratio; default 0.05 "
                                    "via Particles()")
    validate_parser.add_argument(
        "--dry-depth", dest="dry_depth", type=float,
        help="minimum wet depth; default 0.1 via Particles()")
    validate_parser.set_defaults(func=cmd_validate)

    demo_parser = subparsers.add_parser(
        "demo", help="short seeded routing run on a bundled example field")
    demo_parser.add_argument(
        "--field", choices=("rcm", "anuga"), default="rcm",
        help="bundled example field to route on (default: rcm)")
    demo_parser.add_argument("--particles", type=positive_int, default=50)
    demo_parser.add_argument("--steps", type=positive_int, default=3)
    demo_parser.add_argument("--seed", type=int, default=0)
    demo_parser.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    if args.command == "demo" and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
