"""CLI runner for the dorado demo scenario suite; run from any directory.

Runs the selected scenario cases and exports one metrics.json per scenario
plus a summary.md comparison table::

    python scenarios/runner.py --list
    python scenarios/runner.py --all --quick
    python scenarios/runner.py --scenario multi-inlet --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Prefer this checkout, not an unrelated installed dorado distribution.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported after the sys.path bootstrap so the script runs from a source
# checkout without installing the package.
import scenarios  # noqa: E402
from scenarios import ScenarioError  # noqa: E402

DEFAULT_RESULTS_ROOT = ROOT / "scenarios" / "results"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run dorado demo scenarios and export metrics records (one"
            " metrics.json per scenario plus a summary.md table)."
        ),
        epilog="--all supersedes --scenario when both are given.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list scenarios with one-line descriptions and exit",
    )
    parser.add_argument(
        "--scenario", action="append", metavar="NAME",
        help="run one scenario by name (repeatable; see --list)",
    )
    parser.add_argument("--all", action="store_true", help="run every scenario")
    parser.add_argument(
        "--quick", action="store_true",
        help="quick mode: tiny counts sized for CI (default: full demo sizes)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="random seed passed to every scenario (default: 0)",
    )
    parser.add_argument(
        "--output", type=Path, metavar="DIR",
        help="results directory (default: scenarios/results/<timestamp>)",
    )
    return parser


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _selection(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[str]:
    """Scenario names to run, deduplicated; usage errors exit via parser."""
    if args.all:
        return scenarios.scenario_names()
    if not args.scenario:
        parser.error(
            "nothing to run: pass --list, --scenario NAME (repeatable),"
            " or --all")
    known = scenarios.scenario_names()
    unknown = [name for name in args.scenario if name not in known]
    if unknown:
        parser.error(
            f"unknown scenario(s): {', '.join(unknown)}; known:"
            f" {', '.join(known)}")
    # dedupe repeated --scenario flags, preserving first-seen order
    return list(dict.fromkeys(args.scenario))


def _run_all(
    names: list[str], *, quick: bool, seed: int, out_dir: Path,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Run each scenario, writing its metrics.json; report per-run outcome.

    One broken case must not eat the batch: its error is printed, recorded
    in the failures map (summarized and reflected in the exit code), and the
    remaining scenarios still run.
    """
    records: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            record = scenarios.run_scenario(name, quick=quick, seed=seed)
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            failures[name] = str(exc)
            print(f"{name}: FAILED ({exc})", file=sys.stderr)
            continue
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records[name] = record
        speed, processed = record["speed"], record["processed"]
        print(
            f"{name}: {speed['wall_seconds']:.2f}s wall,"
            f" {processed['particles_seeded']} seeded,"
            f" {processed['particles_routed_to_boundary']} routed,"
            f" {processed['particles_active_at_end']} active"
            f" -> {path}")
    return records, failures


def _summary_markdown(
    records: dict[str, dict], failures: dict[str, str],
    *, quick: bool, seed: int,
) -> str:
    """Comparison table for one run batch; full field detail lives in JSON."""
    mode = "quick" if quick else "full"
    lines = [
        "# Scenario run summary",
        "",
        f"- Mode: {mode} (seed {seed})",
        "- Generated: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "- One metrics.json per scenario in this directory.",
        "",
        "| scenario | wall s | parcels/s | peak RSS MiB | seeded | routed"
        " | active |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in records.items():
        speed, processed = record["speed"], record["processed"]
        lines.append(
            f"| {name} | {_fixed(speed['wall_seconds'], 2)}"
            f" | {_fixed(speed['particles_per_second'], 1)}"
            f" | {_mib(record['efficiency']['peak_rss_bytes'])}"
            f" | {processed['particles_seeded']}"
            f" | {processed['particles_routed_to_boundary']}"
            f" | {processed['particles_active_at_end']} |"
        )
    if failures:
        lines += ["", "## Failed scenarios", ""]
        lines += [f"- `{name}`: {message}" for name, message in failures.items()]
    return "\n".join(lines) + "\n"


def _fixed(value: object, places: int) -> str:
    return "—" if value is None else f"{float(value):.{places}f}"


def _mib(num_bytes: object) -> str:
    return "—" if num_bytes is None else f"{int(num_bytes) / (1024 * 1024):.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.list:
            pairs = scenarios.list_scenarios()
            width = max(len(name) for name, _ in pairs)
            for name, description in pairs:
                print(f"{name:<{width}}  {description}")
            return 0

        names = _selection(parser, args)
        out_dir = args.output if args.output is not None else (
            DEFAULT_RESULTS_ROOT / _timestamp())
        out_dir.mkdir(parents=True, exist_ok=True)

        records, failures = _run_all(
            names, quick=args.quick, seed=args.seed, out_dir=out_dir)
        summary_path = out_dir / "summary.md"
        summary_path.write_text(
            _summary_markdown(
                records, failures, quick=args.quick, seed=args.seed),
            encoding="utf-8")
        print(f"summary -> {summary_path}")
        return 1 if failures else 0
    except ScenarioError as exc:
        # case modules that miss the scenario contract: loud, clean, non-zero
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
