"""ctypes adapter for the COBOL particle-stepper kernels.

The COBOL kernels are pure numeric functions of the buffers they
receive: a batched D8 direction pick (phase A) and a batched
travel-time computation (phase B, the moved branch of
``calc_travel_times``). Random numbers are never drawn in COBOL: the
Python layer generates the uniform stream from the same ``np.random``
state the reference uses (one raw double per reference draw) and hands
it to the kernels, so a COBOL-routed walk consumes the reference's
MT19937 stream in exactly the reference's order.

Routing is explicit: setting ``DORADO_COBOL=1`` without a built shared
object raises ``FileNotFoundError`` with build instructions rather than
silently falling back to the Python path (port contract C2/C5).
"""
from __future__ import division, print_function, absolute_import

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

_ENV_VAR = "DORADO_COBOL"
_LIB_NAME = "dorado_cobol.so"

_f64p = ctypes.POINTER(ctypes.c_double)
_i32p = ctypes.POINTER(ctypes.c_int32)
_lib_cache = None


def cobol_active():
    """True when the COBOL backend is selected via the environment."""
    return os.environ.get(_ENV_VAR) == "1"


def library_path():
    """Where the built kernel shared object is expected."""
    return Path(__file__).parent / "lib" / _LIB_NAME


def gnucobol_version():
    """GnuCOBOL version string for provenance, or None when absent."""
    try:
        out = subprocess.check_output(["cobc", "--version"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    # First line is "cobc (GnuCOBOL) 3.2.0"; later lines carry the
    # compiler-toolchain versions, so parse the GnuCOBOL line only.
    for line in out.splitlines():
        if "(GnuCOBOL)" in line:
            fields = line.split()
            return fields[-1]
    return out.strip().splitlines()[0] if out.strip() else None


def load_library():
    """Load (once) and return the COBOL kernel shared object.

    Raises FileNotFoundError naming the build step when the backend was
    requested but not built -- never fall back silently.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    path = Path(__file__).parent / "lib" / _LIB_NAME
    if not path.exists():
        raise FileNotFoundError(
            "%s=1 but the COBOL backend is not built -- run "
            "`make -C cobol` in the repository root (requires GnuCOBOL "
            "3.x, e.g. apt gnucobol3); expected %s" % (_ENV_VAR, path))
    # RTLD_GLOBAL is load-bearing: the kernels CALL "cobol_fadd" and
    # friends, which libcob resolves through dlsym(RTLD_DEFAULT); a
    # default RTLD_LOCAL load hides the helper symbols and the runtime
    # aborts with "module 'cobol_fadd' not found".
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    lib.dorado_pick_cobol.argtypes = [_i32p, _i32p, _i32p, _f64p, _f64p,
                                      _i32p]
    lib.dorado_pick_cobol.restype = ctypes.c_int
    lib.dorado_travel_cobol.argtypes = [_i32p] + [_f64p]*8 + \
        [_i32p, _f64p, _f64p]
    lib.dorado_travel_cobol.restype = ctypes.c_int
    _lib_cache = lib
    return lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def pick_cells(rows, uniforms):
    """Batched phase A: pick a D8 slot per particle weight row.

    ``rows`` is a (k, nw) float64 array of weight rows (the
    ``Particles.weight[x, y, :]`` windows), ``uniforms`` is a (k,)
    float64 array of pre-drawn U[0,1) values. Returns an int32 array of
    picked slot indices (0-based), bit-identical to running
    ``lagrangian_walker.random_pick`` per row with the same uniforms
    (NaN-to-zero row cleanup, cumulative sum, u*total,
    searchsorted-left scan).
    """
    if rows.shape[0] != uniforms.shape[0]:
        raise ValueError("pick_cells: rows (%d) and uniforms (%d) "
                         "disagree" % (rows.shape[0], uniforms.shape[0]))
    if rows.shape[1] > 64:
        raise ValueError("pick_cells: at most 64 weights per row "
                         "supported (D8 window is 9), got %d"
                         % rows.shape[1])
    lib = load_library()
    k = np.array([rows.shape[0]], dtype=np.int32)
    nw = np.array([rows.shape[1]], dtype=np.int32)
    n = np.array([rows.size], dtype=np.int32)
    rows = np.ascontiguousarray(rows, dtype=np.float64)
    uniforms = np.ascontiguousarray(uniforms, dtype=np.float64)
    out = np.empty(rows.shape[0], dtype=np.int32)
    rc = lib.dorado_pick_cobol(_ptr(k, ctypes.c_int32), _ptr(nw, ctypes.c_int32),
                               _ptr(n, ctypes.c_int32), _ptr(rows, ctypes.c_double),
                               _ptr(uniforms, ctypes.c_double), _ptr(out, ctypes.c_int32))
    if rc != 0:
        raise RuntimeError("COBOL pick kernel failed with rc=%d" % rc)
    return out


def travel_times(dist, old_vel, new_vel, ang_new, ang_old, diff_u,
                 diff_mult, vel_mult, moved, diff_coeff):
    """Batched phase B: travel times for the walk core.

    All array inputs are 1-D float64 arrays of equal length k (one
    entry per particle); ``moved`` is an int32 0/1 flag array;
    ``diff_u`` holds one pre-drawn U[0,1) per moved particle in index
    order (ignored when ``diff_coeff <= 0``). Returns a Python list
    matching the reference's ``calc_travel_times`` outputs bit for bit:
    the Python int ``0`` for unmoved particles (the reference leaves an
    int there) and floats elsewhere.
    """
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    old_vel = np.ascontiguousarray(old_vel, dtype=np.float64)
    new_vel = np.ascontiguousarray(new_vel, dtype=np.float64)
    ang_new = np.ascontiguousarray(ang_new, dtype=np.float64)
    ang_old = np.ascontiguousarray(ang_old, dtype=np.float64)
    diff_u = np.ascontiguousarray(diff_u, dtype=np.float64)
    diff_mult = np.ascontiguousarray(diff_mult, dtype=np.float64)
    vel_mult = np.ascontiguousarray(vel_mult, dtype=np.float64)
    moved = np.ascontiguousarray(moved, dtype=np.int32)
    k = dist.shape[0]
    for name, arr in (("old_vel", old_vel), ("new_vel", new_vel),
                      ("ang_new", ang_new), ("ang_old", ang_old),
                      ("diff_mult", diff_mult), ("vel_mult", vel_mult),
                      ("moved", moved)):
        if arr.shape[0] != k:
            raise ValueError("travel_times: %s has length %d, expected "
                             "%d" % (name, arr.shape[0], k))
    lib = load_library()
    kn = np.array([k], dtype=np.int32)
    dc = np.array([diff_coeff], dtype=np.float64)
    out = np.empty(k, dtype=np.float64)
    rc = lib.dorado_travel_cobol(
        _ptr(kn, ctypes.c_int32), _ptr(dist, ctypes.c_double),
        _ptr(old_vel, ctypes.c_double), _ptr(new_vel, ctypes.c_double),
        _ptr(ang_new, ctypes.c_double), _ptr(ang_old, ctypes.c_double),
        _ptr(diff_u, ctypes.c_double), _ptr(diff_mult, ctypes.c_double),
        _ptr(vel_mult, ctypes.c_double), _ptr(moved, ctypes.c_int32),
        _ptr(dc, ctypes.c_double), _ptr(out, ctypes.c_double))
    if rc != 0:
        raise RuntimeError("COBOL travel kernel failed with rc=%d" % rc)
    # The reference leaves the Python int 0 for unmoved particles;
    # byte-identical walk data needs that type distinction preserved.
    return [0 if moved[i] == 0 else float(out[i]) for i in range(k)]
