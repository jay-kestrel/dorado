"""Dorado demo scenario suite.

Auto-discovers runnable scenario cases under :mod:`scenarios.cases`. Every
case module defines ``NAME`` (registry key), ``DESCRIPTION`` (one-liner), and
``run(quick, seed) -> dict`` returning one metrics record;
:mod:`scenarios.core` owns the record schema.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
import pkgutil

_CASES_PACKAGE = __name__ + ".cases"

# Populated lazily by discover_scenarios; a module-level cache is safe here
# because the case set is static for the life of the process.
_REGISTRY: dict[str, ModuleType] | None = None

__all__ = [
    "ScenarioError",
    "discover_scenarios",
    "list_scenarios",
    "run_scenario",
    "scenario_names",
]


class ScenarioError(RuntimeError):
    """A case module violates the scenario contract."""


def discover_scenarios() -> dict[str, ModuleType]:
    """Import every case module and key them by their ``NAME``.

    Case modules that miss the contract (NAME, DESCRIPTION, and a callable
    ``run(quick, seed)``) or that collide on a name raise
    :class:`ScenarioError` naming the module — discovery failures should be
    loud at listing time, not mysterious at run time.
    """
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    cases_package = import_module(_CASES_PACKAGE)
    found: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(cases_package.__path__):
        if info.name.startswith("_"):
            continue
        module = import_module(f"{_CASES_PACKAGE}.{info.name}")
        name = _contract_checked(module)
        if name in found:
            raise ScenarioError(
                f"duplicate scenario name {name!r} in {module.__name__}")
        found[name] = module
    _REGISTRY = found
    return found


def _contract_checked(module: ModuleType) -> str:
    missing = [
        attr for attr in ("NAME", "DESCRIPTION", "run")
        if not hasattr(module, attr)
    ]
    if missing:
        raise ScenarioError(
            f"{module.__name__} does not define {', '.join(missing)};"
            " every case needs NAME, DESCRIPTION, and run(quick, seed)")
    if not callable(module.run):
        raise ScenarioError(f"{module.__name__}.run must be callable")
    if not isinstance(module.NAME, str) or not module.NAME:
        raise ScenarioError(
            f"{module.__name__}.NAME must be a non-empty string")
    return module.NAME


def scenario_names() -> list[str]:
    """All registered scenario names, sorted."""
    return sorted(discover_scenarios())


def list_scenarios() -> list[tuple[str, str]]:
    """(name, one-line description) pairs, ordered by name."""
    return [
        (name, module.DESCRIPTION)
        for name, module in sorted(discover_scenarios().items())
    ]


def run_scenario(name: str, quick: bool = False, seed: int = 0) -> dict:
    """Run one scenario by registry name and return its metrics record."""
    scenarios = discover_scenarios()
    if name not in scenarios:
        raise KeyError(
            f"unknown scenario {name!r}; known: {sorted(scenarios)}")
    return scenarios[name].run(quick=quick, seed=seed)
