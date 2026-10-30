"""Coverage configuration: the routing modules the adversarial review found
crashes and duplication in (routines.py, parallel_routing.py -- review F13)
must be measured by coverage, not omitted, so the reports actually see the
paths that break."""
import configparser
from pathlib import Path

import pytest

# source tree root, if the tests are running from a checkout
REPO_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "pyproject.toml").is_file():
        REPO_ROOT = _parent
        break

# modules that must never return to the coverage omit list
MEASURED_MODULES = ("dorado/routines.py", "dorado/parallel_routing.py")


def test_coverage_does_not_omit_routing_modules():
    if REPO_ROOT is None:
        from dorado import __file__ as pkg_init
        config_path = Path(pkg_init).parent.parent / ".coveragerc"
    else:
        config_path = REPO_ROOT / ".coveragerc"

    parser = configparser.ConfigParser()
    assert parser.read(config_path), f".coveragerc not found at {config_path}"

    omit = parser.get("run", "omit", fallback="")
    omitted = {line.strip() for line in omit.splitlines() if line.strip()}
    for module in MEASURED_MODULES:
        assert module not in omitted, (
            f"{module} must be measured by coverage, not omitted "
            f"(review F13); omit list: {sorted(omitted)}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
