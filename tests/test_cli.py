"""Tests for the pydorado CLI (dorado/cli.py)."""

from pathlib import Path

import numpy as np
import pytest

from dorado import cli

RCM_DATA = Path(__file__).parent / "data" / "ex_deltarcm_data.npz"
ANUGA_DATA = Path(__file__).parent / "data" / "ex_anuga_data.npz"


def test_info_reports_python_platform_and_dependencies(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Python" in out
    assert "platform:" in out
    for dep in cli.DEPENDENCIES:
        assert dep in out


def test_info_never_uses_unknown_version_fallback(capsys):
    assert cli.main(["info"]) == 0
    assert "unknown" not in capsys.readouterr().out


def test_info_lists_bundled_example_data_with_flags(capsys):
    assert cli.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "ex_deltarcm_data.npz" in out
    assert "ex_anuga_data.npz" in out
    assert "present" in out


def test_validate_accepts_bundled_deltarcm_data(capsys):
    code = cli.main(["validate", "--data", str(RCM_DATA), "--model", "DeltaRCM"])
    assert code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "depth: shape (120, 240)" in out
    assert "stage: shape (120, 240)" in out


def test_validate_accepts_bundled_anuga_data(capsys):
    code = cli.main(["validate", "--data", str(ANUGA_DATA), "--model", "Anuga"])
    assert code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "qx: shape (100, 100)" in out


def test_validate_accepts_numeric_overrides(capsys):
    code = cli.main([
        "validate", "--data", str(RCM_DATA), "--model", "DeltaRCM",
        "--dx", "25", "--theta", "0.5", "--gamma", "0.0",
        "--dry-depth", "0.2",
    ])
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_validate_list_arrays_reports_shapes_and_dtypes(capsys):
    code = cli.main([
        "validate", "--data", str(RCM_DATA),
        "--model", "DeltaRCM", "--list-arrays",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "arrays in" in out
    assert "depth: shape (120, 240), dtype" in out
    assert "stage: shape (120, 240), dtype" in out


def test_validate_reports_missing_required_array(capsys):
    # the DeltaRCM export has no qx/qy, so the Anuga mapping cannot proceed
    code = cli.main(["validate", "--data", str(RCM_DATA), "--model", "Anuga"])
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "qx" in err and "qy" in err


def test_validate_reports_shape_mismatch(tmp_path, capsys):
    npz = tmp_path / "mismatch.npz"
    np.savez(npz, depth=np.ones((12, 12)), qx=np.ones((8, 8)),
             qy=np.ones((8, 8)))
    code = cli.main(["validate", "--data", str(npz), "--model", "Anuga"])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_validate_missing_file_exits_one(tmp_path, capsys):
    code = cli.main([
        "validate", "--data", str(tmp_path / "nope.npz"),
        "--model", "DeltaRCM",
    ])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_demo_is_deterministic_and_reports_travel_times(capsys):
    argv = ["demo", "--field", "rcm", "--particles", "50",
            "--steps", "3", "--seed", "0"]
    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert cli.main(argv) == 0
    second = capsys.readouterr().out
    assert first == second
    assert "field=rcm" in first
    assert first.count("\n  (") == 50
    assert "final travel times (seconds):" in first
    assert "mean=" in first and "max=" in first


def test_demo_anuga_field_runs(capsys):
    code = cli.main(["demo", "--field", "anuga", "--particles", "5",
                     "--steps", "2", "--seed", "0"])
    assert code == 0
    out = capsys.readouterr().out
    assert "field=anuga" in out
    assert out.count("\n  (") == 5


@pytest.mark.parametrize("argv", [
    ["validate", "--model", "DeltaRCM"],                     # missing --data
    ["validate", "--data", str(RCM_DATA), "--model", "WRF"],  # bad choice
    ["demo", "--particles", "0"],                             # not positive
    ["demo", "--steps", "-1"],                                # not positive
    ["demo", "--seed", "-1"],                                 # out of range
])
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
