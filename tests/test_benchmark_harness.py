"""Harness runner: parity gate, timing aggregation, results.json contract."""

import json
from pathlib import Path

import pytest

from benchmarks import run_harness as harness
from benchmarks import run_reference as ref

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RCM = ROOT / "bench" / "fixture_rcm"

FAKE_PORT = """\
#!/usr/bin/env python3
import json, os, shutil, sys, time
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
time.sleep(0.02)
walk = os.environ["FAKE_WALK"]
if os.environ.get("FAKE_DOCTORED"):
    data = json.loads(Path(walk).read_text())
    # relative bump: an absolute 1e-12 on a ~1e9 travel time is below one
    # double-precision ULP and would not change the bytes at all
    data["travel_times"][0][1] *= 1 + 1e-13
    (out / "walk_data.json").write_bytes(json.dumps(data, allow_nan=False).encode())
else:
    shutil.copy(walk, out / "walk_data.json")
timing = {"schema_version": 1, "implementation": "fake", "seconds": 0.01,
          "appended_steps": 40, "compiler": "fake cc", "toolchain": "fake rustc",
          "platform": "test"}
(out / "timing.json").write_text(json.dumps(timing))
"""


def reference_bytes(particles: int = 20, iterations: int = 2) -> bytes:
    """In-process reference output on the rcm fixture (the golden bytes)."""
    params, _ = ref.load_inputs(FIXTURE_RCM)
    seed_x, seed_y = ref.SEED_BOXES["rcm"]
    walk_data, _ = ref.run_reference(params, 0, particles, iterations,
                                     seed_x, seed_y)
    return ref.walk_data_bytes(walk_data)


def fake_port(tmp_path: Path, name: str) -> Path:
    script = tmp_path / name
    script.write_text(FAKE_PORT)
    script.chmod(0o755)
    return script


def reps_of(*seconds: float, steps: int = 100) -> list[dict]:
    return [{"seconds": s, "wall_seconds": s + 0.5, "appended_steps": steps}
            for s in seconds]


# ---------------------------------------------------------------- parity gate

def test_parity_passes_identical_bytes():
    verdict = harness.parity_check(b'{"xinds": [1, 2]}', b'{"xinds": [1, 2]}')
    assert verdict["status"] == "pass"
    assert "byte-identical" in verdict["detail"]


def test_parity_rejects_doctored_output(tmp_path):
    """A ~1e-13 relative difference must fail: no tolerance to loosen."""
    golden = reference_bytes()
    data = json.loads(golden)
    # relative bump: an absolute 1e-12 on a ~1e9 travel time is below one ULP
    data["travel_times"][0][1] *= 1 + 1e-13
    doctored = json.dumps(data, allow_nan=False).encode()
    assert doctored != golden
    verdict = harness.parity_check(golden, doctored)
    assert verdict["status"] == "fail"


def test_parity_detail_reports_first_difference():
    verdict = harness.parity_check(b"abcdefgh", b"abcdefXh")
    assert verdict["status"] == "fail"
    assert "byte 6" in verdict["detail"]
    assert "port bug" in verdict["detail"]


def test_parity_detects_length_mismatch():
    verdict = harness.parity_check(b"abcdefgh", b"abcdefg")
    assert verdict["status"] == "fail"
    assert "8 bytes" in verdict["detail"] and "7 bytes" in verdict["detail"]


# --------------------------------------------------------- timing aggregation

def test_summarize_reps_median_iteration_and_wall():
    summary = harness.summarize_reps(
        reps_of(1.0, 2.0, 3.0, 4.0, 5.0), iterations=10)
    assert summary["median_seconds"] == 3.0
    assert summary["seconds_per_iteration"] == pytest.approx(0.3)
    assert summary["all_seconds"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert summary["median_wall_seconds"] == pytest.approx(3.5)
    assert summary["appended_steps"] == 100


def test_summarize_reps_single_repeat():
    summary = harness.summarize_reps(reps_of(2.5), iterations=5)
    assert summary["median_seconds"] == 2.5
    assert summary["seconds_per_iteration"] == pytest.approx(0.5)


def test_summarize_reps_requires_deterministic_steps():
    reps = reps_of(1.0, 2.0)
    reps[1]["appended_steps"] = 99
    with pytest.raises(SystemExit, match="disagree on appended steps"):
        harness.summarize_reps(reps, iterations=10)


# ------------------------------------------------------------ results payload

def test_build_results_schema_complete():
    workload = {"fixture_dir": "bench/fixture_rcm", "model": "DeltaRCM",
                "particles": 10000, "iterations": 100, "seed": 0,
                "seed_x": [15, 16], "seed_y": [137, 138, 139],
                "repetitions": 5, "warmup": {"particles": 200, "iterations": 5},
                "c_binary": "build-c/dorado_c", "rust_binary": "rust/target/release/dorado_rust"}
    provenance = {"revision": "abc123", "working_tree_dirty": False,
                  "platform": "Linux", "machine": "x86_64",
                  "python": "3.13.14", "numpy": "2.3.5", "backend_env": None,
                  "fixture_sha256": {"params.json": "deadbeef"},
                  "c_binary_sha256": "c0ffee", "rust_binary_sha256": "cafe"}
    summaries = {
        "python": harness.summarize_reps(reps_of(2.0), iterations=100),
        "c": harness.summarize_reps(reps_of(0.5), iterations=100),
        "rust": harness.summarize_reps(reps_of(0.25), iterations=100),
    }
    parity = {"python": {"status": "reference"},
              "c": {"status": "pass", "detail": "ok"},
              "rust": {"status": "pass", "detail": "ok"}}
    results = harness.build_results(workload, provenance, summaries, parity)

    assert results["schema_version"] == 1
    assert results["workload"] == workload
    for name in ("python", "c", "rust"):
        entry = results["results"][name]
        assert {"all_seconds", "median_seconds", "all_wall_seconds",
                "median_wall_seconds", "seconds_per_iteration",
                "appended_steps", "parity"} <= set(entry)
        assert entry["parity"] is parity[name]
    assert results["speedup_vs_python"]["c"] == pytest.approx(4.0)
    assert results["speedup_vs_python"]["rust"] == pytest.approx(8.0)


# ------------------------------------------------------------- port execution

def test_run_port_rep_parses_timing_and_measures_wall(tmp_path, monkeypatch):
    walk = tmp_path / "walk.json"
    walk.write_bytes(b"{}")
    monkeypatch.setenv("FAKE_WALK", str(walk))
    script = fake_port(tmp_path, "fake_port.py")
    rep = harness.run_port_rep(script, FIXTURE_RCM, 20, 2, 0,
                               tmp_path / "out")
    assert rep["seconds"] == 0.01
    assert rep["wall_seconds"] >= 0.02
    assert rep["appended_steps"] == 40
    assert rep["timing_meta"]["implementation"] == "fake"


def test_run_port_rep_raises_on_failure(tmp_path):
    script = tmp_path / "failing_port.py"
    script.write_text("#!/usr/bin/env python3\nimport sys\n"
                      "sys.stderr.write('boom\\n')\nsys.exit(3)\n")
    script.chmod(0o755)
    with pytest.raises(RuntimeError, match=r"rc=3") as excinfo:
        harness.run_port_rep(script, FIXTURE_RCM, 20, 2, 0, tmp_path / "out")
    assert "boom" in str(excinfo.value)


# ------------------------------------------------------------ env protections

def test_check_environment_blocks_backend_override(monkeypatch):
    monkeypatch.setenv("DORADO_BACKEND", "rust")
    with pytest.raises(SystemExit, match="DORADO_BACKEND"):
        harness.check_environment()


def test_check_environment_blocks_cobol(monkeypatch):
    monkeypatch.setenv("DORADO_COBOL", "1")
    with pytest.raises(SystemExit, match="DORADO_COBOL"):
        harness.check_environment()


def test_check_environment_allows_python_and_unset(monkeypatch):
    monkeypatch.setenv("DORADO_BACKEND", "python")
    monkeypatch.delenv("DORADO_COBOL", raising=False)
    harness.check_environment()


# ---------------------------------------------------- python rep determinism

def test_python_reps_are_byte_identical_on_shared_params():
    """Same params object, two reps: identical walk bytes and step counts."""
    params, _ = ref.load_inputs(FIXTURE_RCM)
    seed_x, seed_y = ref.SEED_BOXES["rcm"]
    first = harness.run_python_rep(params, 0, 20, 2, seed_x, seed_y)
    second = harness.run_python_rep(params, 0, 20, 2, seed_x, seed_y)
    assert ref.walk_data_bytes(first["walk_data"]) \
        == ref.walk_data_bytes(second["walk_data"])
    assert first["appended_steps"] == second["appended_steps"]


# ---------------------------------------------- end-to-end with fake binaries

def run_harness_with_fakes(tmp_path: Path, monkeypatch,
                           doctored: str = "", repeats: int = 2) -> tuple[int, Path]:
    walk_path = tmp_path / "golden_walk.json"
    walk_path.write_bytes(reference_bytes())
    script = fake_port(tmp_path, "fake_port.py")
    out = tmp_path / "harness-out"
    monkeypatch.setenv("FAKE_WALK", str(walk_path))
    if doctored:
        monkeypatch.setenv("FAKE_DOCTORED", doctored)
    code = harness.main([
        "--input", str(FIXTURE_RCM), "--particles", "20",
        "--iterations", "2", "--repeats", str(repeats),
        "--c-binary", str(script), "--rust-binary", str(script),
        "--out", str(out)])
    return code, out


def test_harness_end_to_end_with_fake_ports(tmp_path, monkeypatch):
    code, out = run_harness_with_fakes(tmp_path, monkeypatch)
    assert code == 0
    results = json.loads((out / "results.json").read_text())
    assert results["workload"]["particles"] == 20
    assert results["workload"]["repetitions"] == 2
    assert results["results"]["python"]["parity"]["status"] == "reference"
    for name in ("c", "rust"):
        assert results["results"][name]["parity"]["status"] == "pass"
        assert results["results"][name]["median_seconds"] == 0.01
        assert len(results["results"][name]["all_seconds"]) == 2
        assert results["speedup_vs_python"][name] > 0
    # rep-1 walk files kept as parity evidence; later reps' walk files pruned.
    for name in ("python", "c", "rust"):
        assert (out / name / "rep1" / "walk_data.json").exists()
        assert (out / name / "rep1" / "timing.json").exists()
        assert not (out / name / "rep2" / "walk_data.json").exists()
        assert (out / name / "rep2" / "timing.json").exists()


def test_harness_fails_on_parity_mismatch(tmp_path, monkeypatch):
    code, out = run_harness_with_fakes(tmp_path, monkeypatch, doctored="1")
    assert code == 1
    results = json.loads((out / "results.json").read_text())
    for name in ("c", "rust"):
        assert results["results"][name]["parity"]["status"] == "fail"
        assert "port bug" in results["results"][name]["parity"]["detail"]
