# Particle state: the Particles struct. Positions are 0-based (x, y) pairs,
# grids are indexed [x + 1, y + 1]; the D8 slot axis is 0..8 (slot access
# A[x, y, k + 1]).

"""
    Particles

Mutable particle-routing state \u2014 the Julia twin of
``dorado.particle_track.Particles`` for the fixture-driven constructor.
Positions are 0-based ``(x, y)`` pairs; grids are indexed ``[x + 1, y + 1]``.
"""
mutable struct Particles
    dx::Float64
    depth::Matrix{Float64}
    stage::Matrix{Float64}
    qx::Matrix{Float64}
    qy::Matrix{Float64}
    u::Matrix{Float64}
    v::Matrix{Float64}
    velocity::Matrix{Float64}
    velocity_angle::Matrix{Float64}
    theta::Float64
    gamma::Float64
    diff_coeff::Float64
    dry_depth::Float64
    steepest_descent::Bool
    stranding_handling::String
    stranding_release::String
    rest_time::Float64
    entrain_prob::Float64
    stranding_dt::Float64
    cell_type::Matrix{Int64}
    distances::Matrix{Float64}
    ivec::Matrix{Float64}
    jvec::Matrix{Float64}
    iwalk::Matrix{Int64}
    jwalk::Matrix{Int64}
    angles::Matrix{Float64}
    roi_grid::Union{Nothing,Matrix{Int64}}
    weight::Array{Float64,3}
    Np_tracer::Int
    walk_data::Union{Nothing,Dict{String,Vector{Vector{OptVar}}}}
    rng::AbstractRNG
    available_particle_vars::Dict{String,Matrix{Float64}}
    particle_variables::Vector{String}
    tracked_variables::Vector{String}
end
