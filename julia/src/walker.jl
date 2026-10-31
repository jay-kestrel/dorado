# Particle stepping: the inverse-CDF pickers, the weighted random walk step,
# travel-time computation, boundary rules, stranding state machine, and the
# per-particle step loop. Ported bug-for-bug from dorado/lagrangian_walker.py
# at master df863fc.
#
# RNG doctrine: the Python original draws from NumPy's legacy global MT19937.
# This port draws from `p.rng` (an explicit package-managed generator), with
# the same draw structure: one uniform per cell choice, one per moving
# particle for diffusion, plus stranding/probability and deposition draws.
# Distributional equivalence is the contract, not stream identity.

"""
Per-particle-kind multipliers from ``Particle.categorize_particles``:
`(travel_time_multiplier, diffusion_multiplier, deposition_probability)`.
"""
const KIND_PRESETS = Dict(
    "sand" => (travel_time_multiplier = 1.2, diffusion_multiplier = 0.5,
               deposition_probability = 0.02),
    "mud" => (travel_time_multiplier = 1.0, diffusion_multiplier = 1.0,
              deposition_probability = 0.005),
    "plankton" => (travel_time_multiplier = 1.0, diffusion_multiplier = 1.2,
                   deposition_probability = 0.0),
)

"""
    effective_multipliers(particle_size, particle_type)
        -> (velocity_mult, diff_mult, deposition_probability)

Resolve the per-particle travel-time/diffusion multipliers exactly like the
Python original: the kind preset supplies the base multipliers (defaults 1.0,
1.0, 0.0 when the particle has no kind), then `particle_size` scales the
travel-time multiplier up by `size` and the diffusion multiplier down by
`1/size`.
"""
function effective_multipliers(particle_size::Union{Nothing,Real},
                               particle_type::Union{Nothing,String})
    if particle_type !== nothing
        preset = KIND_PRESETS[particle_type]
        velocity_mult = Float64(preset.travel_time_multiplier)
        diff_mult = Float64(preset.diffusion_multiplier)
        deposition_probability = Float64(preset.deposition_probability)
    else
        velocity_mult = 1.0
        diff_mult = 1.0
        deposition_probability = 0.0
    end
    if particle_size !== nothing
        size = Float64(particle_size)
        velocity_mult = velocity_mult * size
        diff_mult = diff_mult * (1.0 / size)
    end
    return velocity_mult, diff_mult, deposition_probability
end

"""
    random_pick_seed(rng, choices, [probs]) -> choice

Inverse-CDF pick for seeding: one uniform decides among `choices`
(uniform weights when `probs` is omitted). Mirrors the Python draw structure;
the returned value is the chosen element of `choices`. A uniform that rounds
up to the cumulative total indexes one past the end (BoundsError — the
Python original's IndexError, a reference defect replicated, not fixed).
"""
function random_pick_seed(rng::AbstractRNG,
                          choices::AbstractVector{<:Integer},
                          probs::Union{Nothing,AbstractVector{<:Real}}=nothing;
                          u::Union{Nothing,Float64}=nothing)
    n = length(choices)
    (n > 0) || error("choices must be non-empty")
    if probs === nothing
        cutoffs = collect(Float64, 1.0:n)
    else
        cutoffs = cumsum(Float64.(probs))
    end
    total = cutoffs[end]
    u_ = u === nothing ? rand(rng) * total : u
    idx = searchsortedfirst(cutoffs, u_)  # 1-based first cutoff >= u
    return choices[idx]                   # no clamp: BoundsError past the end
end

"""
    random_pick(rng, probs; u=nothing) -> slot (0..8, 9 on the rounding defect)

Inverse-CDF pick over the 9-slot D8 weight row: NaN weights count as zero,
and a single uniform `u = raw * total` decides the slot. An all-zero row
yields slot 0, which the caller's boundary check later cancels. A uniform
that rounds up to the total yields slot 9 (the Python original's searchsorted
overflow — a reference defect replicated, not fixed; it surfaces downstream
as a BoundsError in `calculate_new_ind`). `u` injects a recorded uniform
(golden replay); `nothing` draws from `rng`.
"""
function random_pick(rng::AbstractRNG, probs::AbstractVector{Float64};
                     u::Union{Nothing,Float64}=nothing)
    p = copy(probs)
    @. p = ifelse(isnan(p), 0.0, p)
    cutoffs = cumsum(p)
    u_ = u === nothing ? rand(rng) * cutoffs[end] : u
    idx = searchsortedfirst(cutoffs, u_) - 1   # back to 0-based
    return Int(idx)
end

"""
    steep_descent(rng, probs; u=nothing) -> slot (0..8)

Deterministic steepest-descent pick: the center slot (4) is excluded, and a
dead end (no positive neighbor left) returns 4 (stay put) without drawing.
Otherwise every non-maximal slot is zeroed and the surviving max-valued
slots (equal weights, so the cumulative spans are equal) are chosen by one
uniform draw. `u` injects a recorded uniform (golden replay).
"""
function steep_descent(rng::AbstractRNG, probs::AbstractVector{Float64};
                       u::Union{Nothing,Float64}=nothing)
    p = copy(probs)
    @. p = ifelse(isnan(p), 0.0, p)
    p[5] = 0.0
    if !any(>(0.0), p)
        return 4
    end
    max_val = maximum(p)
    @. p = ifelse(p < max_val, 0.0, p)
    cutoffs = cumsum(p)
    u_ = u === nothing ? rand(rng) * cutoffs[end] : u
    idx = searchsortedfirst(cutoffs, u_) - 1
    return Int(idx)
end

"""
    get_weight(p, ind) -> cell (0..8)

Per-particle choice of the next D8 slot from `p.weight[ind...]`: weighted
random pick, or steepest descent when `p.steepest_descent` is set.
"""
function get_weight(p::Particles, ind::Tuple{Int,Int})
    x, y = ind
    row = view(p.weight, x + 1, y + 1, :)
    if p.steepest_descent
        return steep_descent(p.rng, row)
    else
        return random_pick(p.rng, row)
    end
end

"""
    calculate_new_ind(ind, new_cell, iwalk, jwalk) -> (x, y)

New cell index after a D8 step: `(x + jwalk[c], y + iwalk[c])` — the crossed
jwalk/iwalk naming is faithful to the Python original.
"""
function calculate_new_ind(ind::Tuple{Int,Int}, new_cell::Int,
                           iwalk::AbstractMatrix, jwalk::AbstractMatrix)
    return (ind[1] + flatidx(jwalk, new_cell), ind[2] + flatidx(iwalk, new_cell))
end

"""Step length for the chosen D8 slot: `distances.flat[c] * dx`."""
step_update(new_cell::Int, distances::AbstractMatrix{Float64}, dx::Real) =
    flatidx(distances, new_cell) * Float64(dx)

"""
    calc_travel_times(p, new_cell, ind, new_ind, dist, velocity_mult=1.0,
                      diff_mult=1.0) -> Float64

Travel time for one step:
`t = 0.5 * d_proj * (1/v_old + 1/v_new) * (1 + diff)` with
`d_proj = dist * max(0, cos(angle[c] - velocity_angle[ind]))` and
`diff = (0.5 - rand()) * diff_coeff * diff_mult` (drawn only when
`diff_coeff > 0`). A particle that did not move gets `t = 0`.
"""
function calc_travel_times(p::Particles, new_cell::Int, ind::Tuple{Int,Int},
                           new_ind::Tuple{Int,Int}, dist::Float64,
                           velocity_mult::Float64=1.0, diff_mult::Float64=1.0)
    if ind != new_ind
        old_vel = p.velocity[ind[1] + 1, ind[2] + 1]
        new_vel = p.velocity[new_ind[1] + 1, new_ind[2] + 1]
        if p.diff_coeff > 0
            diff = (0.5 - rand(p.rng)) * p.diff_coeff * diff_mult
        else
            diff = 0.0
        end
        projected_dist = dist * max(0.0, cos(flatidx(p.angles, new_cell) -
                                             p.velocity_angle[ind[1] + 1, ind[2] + 1]))
        trav_time = 0.5 * projected_dist * (1 / old_vel + 1 / new_vel) * (1 + diff)
        trav_time = trav_time * velocity_mult
    else
        trav_time = 0.0
    end
    return trav_time
end

"""
    check_for_boundary(new_inds, current_inds, cell_type) -> new_inds

Cancel the proposed move of any particle whose CURRENT cell borders an edge
cell (`cell_type == -1`); neighborhood slices are clamped at the array edge.
"""
function check_for_boundary(new_inds::Vector{Tuple{Int,Int}},
                            current_inds::Vector{Tuple{Int,Int}},
                            cell_type::AbstractMatrix{Int64})
    L, W = size(cell_type)
    for i in eachindex(new_inds)
        x, y = current_inds[i]
        x0 = max(0, x - 1) + 1
        x1 = min(x + 1, L - 1) + 1
        y0 = max(0, y - 1) + 1
        y1 = min(y + 1, W - 1) + 1
        if any(==(-1), @view cell_type[x0:x1, y0:y1])
            new_inds[i] = current_inds[i]
        end
    end
    return new_inds
end

"""
    stranding_park_decisions(p, inds_tuple, start_optional)
        -> (park_mask, stranded_times)

Park-while-dry stranding state machine (`stranding_handling = "park"`): a
particle on a dry or land cell parks unless its stranding release condition
is met (`immediate`, `rest` with `rest_time`, or `probability` with an
entrainment draw). `stranded_time` accumulates `stranding_dt` while parked.
"""
function stranding_park_decisions(p::Particles,
                                  inds_tuple::Vector{Tuple{Int,Int}},
                                  start_optional::Dict{String,Vector{OptVar}})
    n = length(inds_tuple)
    prev_stranded = get(start_optional, "stranded", fill(nothing, n))
    prev_times = get(start_optional, "stranded_time", fill(nothing, n))
    park_mask = fill(false, n)
    stranded_times = Vector{Float64}(undef, n)
    for i in 1:n
        x, y = inds_tuple[i]
        stranded_time = prev_times[i] === nothing ? 0.0 : Float64(prev_times[i])
        stranded = prev_stranded[i] === nothing ? 0 : Int(prev_stranded[i])
        wet = (p.depth[x + 1, y + 1] > p.dry_depth) &&
              (p.cell_type[x + 1, y + 1] != 2)
        park = true
        if wet
            if stranded == 0
                park = false
            elseif p.stranding_release == "immediate"
                park = false
            elseif p.stranding_release == "rest"
                park = stranded_time < p.rest_time
            else  # "probability"
                park = !(rand(p.rng) < p.entrain_prob)
            end
        end
        park_mask[i] = park
        stranded_times[i] = park ? stranded_time + p.stranding_dt : stranded_time
    end
    return park_mask, stranded_times
end

"""
    particle_stepper(p, current_inds, travel_times, start_optional)
        -> (new_inds, travel_times, updated_optional)

One iteration of the walker over all particles — the Julia twin of
``dorado.lagrangian_walker.particle_stepper``: per-particle cell choice,
index update, step distance, zero-distance and sentinel freezing, boundary
cancellation, attribute-driven deposition, travel-time accumulation, and
optional-variable sampling. Particles are independent within one step; the
RNG draw order matches the Python per-particle loop.
"""
function particle_stepper(p::Particles,
                          current_inds::Vector{Tuple{Int,Int}},
                          travel_times::Vector{Float64},
                          start_optional::Dict{String,Vector{OptVar}})
    roi = p.roi_grid
    inds_tuple = current_inds
    n = length(inds_tuple)

    if p.stranding_handling == "park"
        park_mask, stranded_times = stranding_park_decisions(p, inds_tuple, start_optional)
    else
        park_mask = fill(false, n)
        stranded_times = nothing
    end

    new_cells = Vector{Int}(undef, n)
    for i in 1:n
        x = inds_tuple[i]
        new_cells[i] = (x != (0, 0) && !park_mask[i]) ? get_weight(p, x) : 4
    end

    new_inds = Vector{Tuple{Int,Int}}(undef, n)
    for i in 1:n
        new_inds[i] = new_cells[i] != 4 ?
            calculate_new_ind(inds_tuple[i], new_cells[i], p.iwalk, p.jwalk) :
            inds_tuple[i]
    end

    dist = Vector{Float64}(undef, n)
    for i in 1:n
        dist[i] = step_update(new_cells[i], p.distances, p.dx)
    end

    # Zero-distance picks keep their indices (deltas of 0 in both directions).
    for i in 1:n
        if dist[i] == 0
            new_inds[i] = inds_tuple[i]
        end
    end

    new_inds = check_for_boundary(new_inds, inds_tuple, p.cell_type)

    # Particles pushed outside the domain by wrap-around weights stay put.
    L, W = size(p.cell_type)
    for i in 1:n
        nx, ny = new_inds[i]
        if nx < 0 || nx >= L || ny < 0 || ny >= W
            new_inds[i] = inds_tuple[i]
        end
    end

    sizes = get(start_optional, "particle_size", nothing)
    types = get(start_optional, "particle_type", nothing)
    deposited = get(start_optional, "deposited", nothing)

    if sizes !== nothing || types !== nothing
        velocity_mults = Vector{Float64}(undef, n)
        diff_mults = Vector{Float64}(undef, n)
        deposition_probs = Vector{Float64}(undef, n)
        for i in 1:n
            s = sizes === nothing ? nothing : sizes[i]
            t = types === nothing ? nothing : types[i]
            velocity_mults[i], diff_mults[i], deposition_probs[i] =
                effective_multipliers(s, t)
        end
    else
        velocity_mults = fill(1.0, n)
        diff_mults = fill(1.0, n)
        deposition_probs = fill(0.0, n)
    end

    new_deposited = nothing
    if sizes !== nothing || types !== nothing || deposited !== nothing
        new_deposited = deposited === nothing ?
            Vector{OptVar}(fill(Int64(0), n)) : copy(deposited)
        for i in 1:n
            moved = new_inds[i] != inds_tuple[i]
            if new_deposited[i] == 1
                new_inds[i] = inds_tuple[i]
            elseif deposition_probs[i] > 0 && moved && rand(p.rng) < deposition_probs[i]
                new_inds[i] = inds_tuple[i]
                new_deposited[i] = Int64(1)
            end
        end
    end

    temp_travel = Vector{Float64}(undef, n)
    for i in 1:n
        temp_travel[i] = calc_travel_times(p, new_cells[i], inds_tuple[i],
                                           new_inds[i], dist[i],
                                           velocity_mults[i], diff_mults[i])
    end
    if stranded_times !== nothing
        for i in 1:n
            if park_mask[i]
                temp_travel[i] = p.stranding_dt
            end
        end
    end
    travel_times = [travel_times[i] + temp_travel[i] for i in 1:n]

    updated_optional = Dict{String,Vector{OptVar}}()
    if new_deposited !== nothing
        updated_optional["deposited"] = new_deposited
    end
    if roi !== nothing
        updated_optional["roi_flag"] =
            Vector{OptVar}([roi[x + 1, y + 1] == 1 ? Int64(1) : Int64(0)
                            for (x, y) in new_inds])
    end
    for (var, vals) in start_optional
        haskey(updated_optional, var) && continue
        if haskey(p.available_particle_vars, var)
            grid = p.available_particle_vars[var]
            updated_optional[var] =
                Vector{OptVar}([grid[x + 1, y + 1] for (x, y) in new_inds])
        else
            updated_optional[var] = copy(vals)
        end
    end
    if p.stranding_handling == "park"
        updated_optional["stranded"] =
            Vector{OptVar}([parked ? Int64(1) : Int64(0) for parked in park_mask])
        updated_optional["stranded_time"] = Vector{OptVar}(stranded_times)
    end

    return new_inds, travel_times, updated_optional
end
