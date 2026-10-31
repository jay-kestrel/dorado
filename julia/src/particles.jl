# Particle state and drivers: a constructor that
# reproduces Particles.__init__ derivation from a normalized fixture
# directory, generate_particles seeding, and run_iteration bookkeeping.
# Ported bug-for-bug from dorado/particle_track.py at master df863fc.

"""
    Particles(fixture_dir; rng=nothing) -> Particles

Build the particle state from a normalized fixture directory: ``params.json``
plus uncompressed ``.npy`` arrays (the contract shared with
``benchmarks/run_reference.py`` and ``julia/goldens/``). All of the
derivation in ``Particles.__init__`` is replicated: NaN cleanup, the
non-DeltaRCM axis swap, velocity magnitude and sign-preserving floors,
``velocity_angle = atan(-u, v)``, cell-type estimation from depth with a -1
edge ring, defaults and validation for theta/gamma/diff_coeff/dry_depth and
the stranding parameters, the default D8 tables, and the routing weights.

The RNG is package-managed: a `StableRNGs.LehmerRNG` seeded from the
manifest's ``seed`` (or an explicit `rng=` object, which takes precedence),
replacing the Python original's legacy global NumPy stream.
"""
function Particles(fixture_dir::AbstractString; rng::Union{Nothing,AbstractRNG}=nothing)
    manifest = JSON.parsefile(joinpath(fixture_dir, "params.json"))

    arrays = manifest["arrays"]
    depth_in = npread(joinpath(fixture_dir, arrays["depth"]))
    stage_in = npread(joinpath(fixture_dir, arrays["stage"]))
    qx_in = npread(joinpath(fixture_dir, arrays["qx"]))
    qy_in = npread(joinpath(fixture_dir, arrays["qy"]))
    L, W = size(depth_in)

    dx = Float64(manifest["dx"])
    model = get(manifest, "model", nothing)
    model in ("DeltaRCM", "Anuga") ||
        throw(ArgumentError("Invalid model type. Choose from 'DeltaRCM' or 'Anuga'."))
    theta = haskey(manifest, "theta") ? Float64(manifest["theta"]) : 1.0
    (theta >= 0) || throw(ArgumentError("Negative theta is not allowed; theta must be >= 0."))
    gamma = haskey(manifest, "gamma") ? Float64(manifest["gamma"]) : 0.05
    (0 <= gamma <= 1) ||
        throw(ArgumentError("Specified gamma value is outside the range [0, 1]. Gamma must satisfy 0 <= gamma <= 1."))
    steepest_descent = get(manifest, "steepest_descent", false)
    if haskey(manifest, "diff_coeff")
        diff_coeff = Float64(manifest["diff_coeff"])
        if diff_coeff < 0
            diff_coeff = 0.0
        end
    else
        diff_coeff = steepest_descent ? 0.0 : 0.2
    end
    dry_depth = haskey(manifest, "dry_depth") ? Float64(manifest["dry_depth"]) : 0.1

    stranding_handling = get(manifest, "stranding_handling", "park")
    stranding_handling in ("park", "none") ||
        throw(ArgumentError("Invalid stranding_handling; must be \"park\" or \"none\"."))
    stranding_release = get(manifest, "stranding_release", "immediate")
    stranding_release in ("immediate", "rest", "probability") ||
        throw(ArgumentError("Invalid stranding_release; must be \"immediate\", \"rest\", or \"probability\"."))
    rest_time = Float64(get(manifest, "rest_time", 0.0))
    (rest_time >= 0) || throw(ArgumentError("rest_time must be >= 0."))
    entrain_prob = Float64(get(manifest, "entrain_prob", 1.0))
    (0 < entrain_prob <= 1) ||
        throw(ArgumentError("entrain_prob must be in (0, 1]."))
    stranding_dt = Float64(get(manifest, "stranding_dt", 1.0))
    (stranding_dt > 0) || throw(ArgumentError("stranding_dt must be > 0."))

    # Depth: NaN -> 0 (copy semantics, caller arrays untouched).
    depth = copy(depth_in)
    depth[isnan.(depth)] .= 0.0

    # Stage: NaN on dry cells.
    stage = copy(stage_in)
    stage[depth .== 0.0] .= NaN

    # Discharges; the non-DeltaRCM convention swaps the x/y axes.
    if model == "DeltaRCM"
        qx = copy(qx_in)
        qy = copy(qy_in)
    else
        qx = -1.0 .* qy_in
        qy = copy(qx_in)
    end
    qx[isnan.(qx)] .= 0.0
    qy[isnan.(qy)] .= 0.0

    # Depth-averaged velocities with the epsilon floor.
    u = (qx .* depth) ./ (depth .^ 2 .+ 1e-8)
    u[isnan.(u)] .= 0.0
    v = (qy .* depth) ./ (depth .^ 2 .+ 1e-8)
    v[isnan.(v)] .= 0.0

    velocity = sqrt.(u .^ 2 .+ v .^ 2)
    velocity[velocity .< 1e-8] .= 1e-8
    tiny_u = abs.(u) .< 1e-8
    u = ifelse.(tiny_u, (sign.(u) .+ (u .== 0.0)) .* 1e-8, u)
    tiny_v = abs.(v) .< 1e-8
    v = ifelse.(tiny_v, (sign.(v) .+ (v .== 0.0)) .* 1e-8, v)
    velocity_angle = atan.(-1.0 .* u, v)

    # Cell types: from the fixture when given, else estimated from depth with
    # a -1 edge ring.
    if haskey(arrays, "cell_type")
        cell_type = Int64.(npread(joinpath(fixture_dir, arrays["cell_type"])))
    else
        cell_type = zeros(Int64, L, W)
        cell_type[depth .< dry_depth] .= 2
        padded = fill(Int64(-1), L, W)
        padded[2:L-1, 2:W-1] .= cell_type[2:L-1, 2:W-1]
        cell_type = padded
    end

    d8 = d8_constants()
    seed = haskey(manifest, "seed") ? UInt64(Int64(manifest["seed"])) : UInt64(0)
    rng_ = rng === nothing ? StableRNGs.LehmerRNG(seed) : rng

    particle_variables = String.(get(manifest, "particle_variables", String[]))
    available_particle_vars = Dict{String,Matrix{Float64}}(
        "depth" => depth, "stage" => stage, "qx" => qx, "qy" => qy,
        "u" => u, "v" => v,
    )
    tracked_variables = vcat(["xinds", "yinds", "travel_times"], particle_variables)
    if stranding_handling == "park"
        tracked_variables = vcat(tracked_variables, ["stranded", "stranded_time"])
    end

    p = Particles(dx, depth, stage, qx, qy, u, v, velocity, velocity_angle,
        theta, gamma, diff_coeff, dry_depth, Bool(steepest_descent),
        stranding_handling, stranding_release, rest_time, entrain_prob,
        stranding_dt, cell_type, d8.distances, d8.ivec, d8.jvec, d8.iwalk,
        d8.jwalk, d8.angles, nothing, zeros(Float64, L, W, 9), 0, nothing,
        rng_, available_particle_vars, particle_variables, tracked_variables)

    make_weight!(p)
    return p
end

"""
    generate_particles(p, Np_tracer, seed_xloc, seed_yloc; seed_time=0,
                       method="random", particle_size=nothing,
                       particle_type=nothing)

Seed `Np_tracer` particles and initialize `p.walk_data`. `method = "random"`
draws independent coordinates from `seed_xloc`/`seed_yloc` (one uniform per
axis per particle); `method = "exact"` splits the particles evenly across the
seed locations with the remainder going to the first locations. Seed cells
are validated in-domain. Optional per-particle attributes (`particle_size`,
`particle_type` — "sand", "mud", or "plankton") may be given as a scalar or a
length-`Np_tracer` vector; they enable the deposition paths in the stepper.
The Python original's cross-run attribute merge (`previous_walk_data`) is out
of scope for the first port.
"""
function generate_particles(p::Particles, Np_tracer::Integer,
                            seed_xloc::AbstractVector{<:Integer},
                            seed_yloc::AbstractVector{<:Integer};
                            seed_time::Real=0, method::String="random",
                            particle_size=nothing, particle_type=nothing)
    Np = Int(Np_tracer)
    method in ("random", "exact") ||
        throw(ArgumentError("Method input is not a valid method, must be \"random\" or \"exact\"."))
    L, W = size(p.depth)
    if length(seed_xloc) > 0 && length(seed_yloc) > 0
        if minimum(seed_xloc) < 0 || maximum(seed_xloc) > L - 1 ||
           minimum(seed_yloc) < 0 || maximum(seed_yloc) > W - 1
            throw(ArgumentError("Seed location is outside of the domain: x indices must " *
                                "be in [0, $(L - 1)] and y indices must be in [0, $(W - 1)]."))
        end
    end

    new_start_times = fill(Float64(seed_time), Np)
    if method == "random"
        new_start_xindices = [random_pick_seed(p.rng, collect(seed_xloc)) for _ in 1:Np]
        new_start_yindices = [random_pick_seed(p.rng, collect(seed_yloc)) for _ in 1:Np]
    else
        # Exact seeding: the X axis's divmod drives BOTH axes — faithful to
        # the Python original, including that a short seed_yloc raises
        # BoundsError here (IndexError in Python) rather than erroring
        # cleanly, and a longer one is truncated back to Np below.
        q, r = divrem(Np, length(seed_xloc))
        new_start_xindices = vcat(repeat(collect(seed_xloc), q),
                                  [seed_xloc[i] for i in 1:r])
        new_start_yindices = vcat(repeat(collect(seed_yloc), q),
                                  [seed_yloc[i] for i in 1:r])
    end

    new_xinds = Vector{Vector{OptVar}}([[Int64(new_start_xindices[i])] for i in 1:Np])
    new_yinds = Vector{Vector{OptVar}}([[Int64(new_start_yindices[i])] for i in 1:Np])

    # Persistent per-particle attribute entries for the new batch (empty when
    # this call provides no sizes/types, so runs without attributes never
    # gain the new walk_data keys).
    new_attribute_entries = Dict{String,Vector{Vector{OptVar}}}()
    if particle_size !== nothing || particle_type !== nothing
        sizes_full = particle_size === nothing ?
            Vector{OptVar}(fill(nothing, Np)) :
            _attribute_vector(particle_size, Np, "particle_size")
        types_full = particle_type === nothing ?
            Vector{OptVar}(fill(nothing, Np)) :
            _attribute_type_vector(particle_type, Np)
        new_attribute_entries["particle_size"] =
            Vector{Vector{OptVar}}([[s] for s in sizes_full])
        new_attribute_entries["particle_type"] =
            Vector{Vector{OptVar}}([[t] for t in types_full])
        new_attribute_entries["deposited"] =
            Vector{Vector{OptVar}}([[Int64(0)] for _ in 1:Np])
    end

    # Initialize optional variables from available_particle_vars.
    optional_start_vars = Dict{String,Vector{Vector{OptVar}}}()
    for var in p.tracked_variables
        var in ("xinds", "yinds", "travel_times") && continue
        if var == "roi_flag" && p.roi_grid !== nothing
            optional_start_vars[var] = Vector{Vector{OptVar}}(
                [[roi[x + 1, y + 1] == 1 ? Int64(1) : Int64(0)]
                 for (x, y) in zip(new_start_xindices, new_start_yindices)])
        elseif haskey(p.available_particle_vars, var)
            grid = p.available_particle_vars[var]
            optional_start_vars[var] = Vector{Vector{OptVar}}(
                [[OptVar(grid[x + 1, y + 1])]
                 for (x, y) in zip(new_start_xindices, new_start_yindices)])
        elseif var in ("stranded", "stranded_time")
            # Stranding state starts at zero for fresh seeds.
            optional_start_vars[var] =
                Vector{Vector{OptVar}}([[Int64(0)] for _ in 1:Np])
        else
            throw(ArgumentError("Variable $(var) is not a recognized variable to track. " *
                                "Available variables are: " *
                                "$(sort(collect(keys(p.available_particle_vars)))) " *
                                "and 'roi_flag' (if roi_grid is provided)."))
        end
    end

    start_xinds = new_xinds
    start_yinds = new_yinds
    start_times = Vector{Vector{OptVar}}([[t] for t in new_start_times])

    attribute_sources = Vector{Dict{String,Vector{Vector{OptVar}}}}()
    if p.walk_data !== nothing
        # Keep walk data from a previous call and append the new data to it.
        internal_xinds = p.walk_data["xinds"]
        start_xinds = vcat(internal_xinds, start_xinds)
        start_yinds = vcat(p.walk_data["yinds"], start_yinds)
        start_times = vcat(p.walk_data["travel_times"], start_times)

        for (var, new_var_list) in optional_start_vars
            if haskey(p.walk_data, var)
                optional_start_vars[var] = vcat(p.walk_data[var], new_var_list)
            elseif var in ("stranded", "stranded_time")
                # Walk data predates stranding tracking: pad with zeros so
                # each particle's list stays aligned with its recorded steps.
                optional_start_vars[var] = vcat(
                    [[Int64(0) for _ in eachindex(path)] for path in internal_xinds],
                    new_var_list)
            end
        end
        push!(attribute_sources, p.walk_data)
    end

    # Merge the persistent per-particle attributes across sources in the same
    # order the particle histories are concatenated above (self.walk_data +
    # new batch; the Python original's previous_walk_data source is out of
    # scope for the first port), padding particles without attributes with
    # nothing so per-particle list lengths stay aligned with the walk history.
    for var in ("particle_size", "particle_type", "deposited")
        if !haskey(new_attribute_entries, var) &&
           !any(src -> haskey(src, var), attribute_sources)
            # No attributes anywhere for this key: a legacy run that gains
            # none of the new walk_data keys.
            continue
        end
        merged = Vector{Vector{OptVar}}()
        for src in attribute_sources
            merged = vcat(merged,
                          merge_walk_attributes(get(src, var, nothing), src["xinds"]))
        end
        if haskey(new_attribute_entries, var)
            merged = vcat(merged, new_attribute_entries[var])
        else
            # New batch is unattributed but earlier batches carry this
            # attribute: pad the new particles with nothing.
            merged = vcat(merged, Vector{Vector{OptVar}}([[nothing] for _ in 1:Np]))
        end
        optional_start_vars[var] = merged
    end

    p.Np_tracer = length(start_xinds)

    init_walk_data = Dict{String,Vector{Vector{OptVar}}}(
        "xinds" => start_xinds,
        "yinds" => start_yinds,
        "travel_times" => start_times,
    )
    merge!(init_walk_data, optional_start_vars)
    p.walk_data = init_walk_data
    return init_walk_data
end

# Per-particle attribute coercion: scalars broadcast, vectors validate
# length and numeric/string typing, `nothing` passes through (an unattributed
# particle). Mirrors the Python original's light-touch handling.
function _attribute_vector(value, Np::Int, name::String)
    value === nothing && return nothing
    if value isa Real
        return Vector{OptVar}(fill(Float64(value), Np))
    elseif value isa AbstractVector
        length(value) == Np ||
            throw(ArgumentError("$name must be a scalar or a vector of length $Np."))
        out = Vector{OptVar}(undef, Np)
        for (i, v) in enumerate(value)
            (v === nothing || v isa Real) ||
                throw(ArgumentError("$name entries must be numeric."))
            out[i] = v === nothing ? nothing : Float64(v)
        end
        return out
    else
        throw(ArgumentError("$name must be a scalar or a vector."))
    end
end

function _attribute_type_vector(value, Np::Int)
    value === nothing && return nothing
    if value isa String
        value in keys(KIND_PRESETS) ||
            throw(ArgumentError("particle_type must be one of " *
                                "$(sort(collect(keys(KIND_PRESETS))))."))
        return Vector{OptVar}(fill(value, Np))
    elseif value isa AbstractVector
        length(value) == Np ||
            throw(ArgumentError("particle_type must be a string or a vector of length $Np."))
        out = Vector{OptVar}(undef, Np)
        for (i, v) in enumerate(value)
            (v === nothing || v isa String) ||
                throw(ArgumentError("particle_type entries must be strings."))
            out[i] = v
        end
        return out
    else
        throw(ArgumentError("particle_type must be a string or a vector."))
    end
end

"""Concatenate a persistent attribute's histories with a previous walk data,
padding particles that did not carry the attribute with `nothing` entries so
per-particle list lengths stay aligned with the walk history. (The Python
original also appends `new_entries`; here the new batch is concatenated by
the caller.)"""
function merge_walk_attributes(prev_histories::Union{Nothing,Vector{Vector{OptVar}}},
                               prev_xinds::Vector{Vector{OptVar}})
    if prev_histories === nothing
        return Vector{Vector{OptVar}}(
            [[nothing for _ in eachindex(path)] for path in prev_xinds])
    end
    return prev_histories
end

"""
    run_iteration(p) -> walk_data

One full routing iteration for all particles, with the bookkeeping of
``run_iteration(target_time=None)``: walk data is appended only for particles
that actually moved (or are flagged stranded), so frozen particles record
nothing. The target-time routing mode is out of scope for the first port.
"""
function run_iteration(p::Particles)
    p.walk_data === nothing &&
        throw(ArgumentError("Particles have not been initialized. Call the " *
                            "`generate_particles()` function."))
    walk = p.walk_data
    N = p.Np_tracer
    all_xinds = walk["xinds"]
    all_yinds = walk["yinds"]
    all_times = walk["travel_times"]

    start_xindices = [Int(all_xinds[i][end]) for i in 1:N]
    start_yindices = [Int(all_yinds[i][end]) for i in 1:N]

    optional_vars = Dict{String,Vector{Vector{OptVar}}}()
    for var in p.tracked_variables
        var in ("xinds", "yinds", "travel_times") && continue
        optional_vars[var] = haskey(walk, var) ? walk[var] :
                             Vector{Vector{OptVar}}([[nothing] for _ in 1:N])
    end
    for var in ("particle_size", "particle_type", "deposited")
        haskey(walk, var) && (optional_vars[var] = walk[var])
    end

    start_optional = Dict{String,Vector{OptVar}}(
        var => [opt[i][end] for i in 1:N] for (var, opt) in optional_vars)

    start_pairs = [(start_xindices[i], start_yindices[i]) for i in 1:N]
    start_times = [Float64(all_times[i][end]) for i in 1:N]

    new_inds, travel_times, updated_optional =
        particle_stepper(p, start_pairs, start_times, start_optional)

    for i in 1:N
        parked = haskey(updated_optional, "stranded") &&
                 updated_optional["stranded"][i] == 1
        if new_inds[i] != start_pairs[i] || parked
            push!(all_xinds[i], Int64(new_inds[i][1]))
            push!(all_yinds[i], Int64(new_inds[i][2]))
            push!(all_times[i], Float64(travel_times[i]))
            for (var, opt) in optional_vars
                push!(opt[i], updated_optional[var][i])
            end
        elseif haskey(optional_vars, "deposited")
            optional_vars["deposited"][i][end] = updated_optional["deposited"][i]
        end
    end
    return walk
end
