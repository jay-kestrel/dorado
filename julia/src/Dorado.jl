"""
    Dorado

Julia port of dorado's Lagrangian particle-routing kernels
(`dorado/lagrangian_walker.py` and the routing parts of
`dorado/particle_track.py`).

Scope (first port, spec `art_nlQ72wru`): `make_weight`, `particle_stepper`
and its callees (`random_pick`, `steep_descent`, `calc_travel_times`,
`check_for_boundary`, stranding decisions), `run_iteration(target_time=nothing)`
bookkeeping, and `generate_particles`. Out of scope: plotting/glue
(`routines.py`), SciPy analytics, `parallel_routing`, file I/O, and the
target-time routing mode.

Equivalence doctrine: the Python original routes with NumPy's legacy global
MT19937 stream; this port owns an explicit package-managed RNG
(`StableRNGs.LehmerRNG`, seeded from the fixture manifest) and mirrors the
*draw structure* — which draws exist, from which distributions, in what
order — so results are distributionally equivalent, not stream-identical.
The deterministic (steepest-descent, diffusion-free, exact-seeding) path is
free of random draws and is held bit-exact against Python goldens.

Indexing convention: positions are 0-based `(x, y)` cell indices exactly as
in Python (`x` = row, `y` = column); grid access is `grid[x + 1, y + 1]`.
The 9-slot D8 neighborhood keeps the Python slot order
`[NW, N, NE, W, 0, E, SW, S, SE]` addressed 0..8 (`A[x, y, k + 1]`).
"""
module Dorado

using Random
using StableRNGs
using Printf
using Statistics
using JSON
using SHA

# Per-particle optional-variable value: grid samples are Float64, indices and
# flags are Int, per-particle attributes are Float64/String, missing is
# `nothing`. Mirrors the heterogeneous Python lists in walk_data.
const OptVar = Union{Nothing,Int64,Float64,String}

include("npy.jl")
include("state.jl")
include("weights.jl")
include("walker.jl")
include("particles.jl")

export OptVar,
    Particles,
    npread,
    KIND_PRESETS,
    effective_multipliers,
    random_pick_seed,
    random_pick,
    steep_descent,
    get_weight,
    calculate_new_ind,
    step_update,
    calc_travel_times,
    check_for_boundary,
    stranding_park_decisions,
    particle_stepper,
    make_weight!,
    generate_particles,
    run_iteration

end # module
