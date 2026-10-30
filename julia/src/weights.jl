# D8 neighborhood tables and the routing-weight construction, ported 1:1 from
# dorado/lagrangian_walker.py (big_sliding_window, the tile helpers,
# clear_borders, and make_weight).
#
# Indexing convention: positions (x, y) are 0-based cell indices exactly as in
# Python — x = row, y = column; array access is `grid[x + 1, y + 1]`. The
# 9-slot D8 neighborhood keeps the Python slot order
# [NW, N, NE, W, 0, E, SW, S, SE] addressed 0..8; slot access on a 3-array is
# `A[x, y, k + 1]`, and `A.flat[k]` on a 3x3 table is `flatidx(A, k)`.

"""
    d8_constants() -> NamedTuple

The default D8 tables from ``Particles.__init__``: distances, ivec, jvec,
iwalk, jwalk, angles. (The Python original lets callers override these; the
golden fixtures never do, and the port fixes them at the defaults.)
"""
function d8_constants()
    sqrt2 = sqrt(2.0)
    sqrt05 = sqrt(0.5)
    p = Float64(pi)
    (
        distances = [sqrt2 1.0 sqrt2; 1.0 1.0 1.0; sqrt2 1.0 sqrt2],
        ivec = [-sqrt05 0.0 sqrt05; -1.0 0.0 1.0; -sqrt05 0.0 sqrt05],
        jvec = [-sqrt05 -1.0 -sqrt05; 0.0 0.0 0.0; sqrt05 1.0 sqrt05],
        iwalk = Int64[-1 0 1; -1 0 1; -1 0 1],
        jwalk = Int64[-1 -1 -1; 0 0 0; 1 1 1],
        angles = [3*p/4 p/2 p/4; p 0.0 0.0; 5*p/4 3*p/2 7*p/4],
    )
end

"""`A.flat[k]` for a 3x3 D8 table (0-based, row-major)."""
@inline flatidx(A::AbstractMatrix, k::Integer) = A[k ÷ 3 + 1, k % 3 + 1]

"""
    big_sliding_window(raster) -> Array{Float64,3}

Assemble each cell's D8 neighborhood into an ``(L, W, 9)`` array, slot order
``[NW, N, NE, W, 0, E, SW, S, SE]``. Edge cells wrap around to the opposite
side (faithful to the Python original, including that quirk); border slots
are zeroed later by `clear_borders!`.
"""
function big_sliding_window(raster::AbstractMatrix{<:Real})
    L, W = size(raster)
    (L >= 3 && W >= 3) || error("domain must be at least 3x3")
    big_ravel = zeros(Float64, L, W, 9)
    for k in 1:9
        ox = (k - 1) ÷ 3 - 1
        oy = (k - 1) % 3 - 1
        big_ravel[2:L-1, 2:W-1, k] .= @view raster[2+ox:L-1+ox, 2+oy:W-1+oy]
    end
    return big_ravel
end

"""
    tile_local_array(local_array, L, W) -> Array{Float64,3}

Tile a 3x3 D8 table across the domain: slot ``k`` gets ``local_array.flat[k]``
everywhere.
"""
function tile_local_array(local_array::AbstractMatrix, L::Int, W::Int)
    T = zeros(Float64, L, W, 9)
    for k in 1:9
        T[:, :, k] .= flatidx(local_array, k - 1)
    end
    return T
end

"""Tile a raster along the slot axis: slot ``k`` gets the whole raster."""
tile_domain_array(raster::AbstractMatrix{Float64}) = repeat(raster, 1, 1, 9)

"""
    clear_borders!(tiled) -> tiled

Zero every slot of the edge cells (``x == 0``, ``x == L-1``, ``y == 0``,
``y == W-1``).
"""
function clear_borders!(tiled::AbstractArray{Float64,3})
    tiled[1, :, :] .= 0.0
    tiled[:, 1, :] .= 0.0
    tiled[end, :, :] .= 0.0
    tiled[:, end, :] .= 0.0
    return tiled
end

# NumPy-faithful reduction helpers over the 9-slot axis: NaN counts as zero
# (nansum), and the summation order matches NumPy's pairwise algorithm for
# n = 9 (four two-element accumulators plus a sequential tail), so the sums
# are bit-identical to the Python weights.
@inline _nz(v::Float64) = isnan(v) ? 0.0 : v

function nansum_axis3(a::AbstractArray{Float64,3})
    L, W, _ = size(a)
    out = zeros(Float64, L, W)
    @inbounds for j in 1:W, i in 1:L
        v1 = _nz(a[i, j, 1]); v2 = _nz(a[i, j, 2])
        v3 = _nz(a[i, j, 3]); v4 = _nz(a[i, j, 4])
        v5 = _nz(a[i, j, 5]); v6 = _nz(a[i, j, 6])
        v7 = _nz(a[i, j, 7]); v8 = _nz(a[i, j, 8])
        v9 = _nz(a[i, j, 9])
        out[i, j] = (((v1 + v2) + (v3 + v4)) + ((v5 + v6) + (v7 + v8))) + v9
    end
    return out
end

function max_axis3(a::AbstractArray{Float64,3})
    L, W, _ = size(a)
    out = Matrix{Float64}(undef, L, W)
    @inbounds for j in 1:W, i in 1:L
        m = a[i, j, 1]
        for k in 2:9
            m = max(m, a[i, j, k])
        end
        out[i, j] = m
    end
    return out
end

function any_axis3(mask::AbstractArray{Bool,3})
    L, W, _ = size(mask)
    out = falses(L, W)
    # Genuinely nested loops (NOT the comma-chained form): in Julia, `break`
    # inside a comma-chained `for a, b, c` exits the ENTIRE statement, which
    # silently stops the scan after the first true and misses every later
    # cell. The inner `for k` here scopes the break per-cell, as `any` over
    # axis 3 requires.
    @inbounds for j in 1:W, i in 1:L
        for k in axes(mask, 3)
            if mask[i, j, k]
                out[i, j] = true
                break
            end
        end
    end
    return out
end

"""
    make_weight!(p::Particles) -> p

Construct the routing weights ``(L, W, 9)`` once per domain — the Julia twin
of ``dorado.lagrangian_walker.make_weight``. Surface-gradient and inertial
terms are built, dry/land neighbors zeroed, per-cell normalization applied,
the terms blended by ``gamma``, modulated by ``depth^theta``, and the
dead-end rule (deepest neighbor gets weight 1.0) applied. Behavior is
bug-for-bug, including the wrap-around neighborhood and the
deep-dive quirk of the stranding-aware dead-end rule.
"""
function make_weight!(p::Particles)
    L, W = size(p.stage)

    weight_sfc = tile_domain_array(p.stage) .- big_sliding_window(p.stage)
    weight_sfc ./= tile_local_array(p.distances, L, W)
    @. weight_sfc = ifelse(weight_sfc <= 0, 0.0, weight_sfc)  # NaN stays NaN
    clear_borders!(weight_sfc)

    weight_int = tile_domain_array(p.qx) .* tile_local_array(p.jvec, L, W) .+
                 tile_domain_array(p.qy) .* tile_local_array(p.ivec, L, W)
    weight_int ./= tile_local_array(p.distances, L, W)
    @. weight_int = ifelse(weight_int <= 0, 0.0, weight_int)
    clear_borders!(weight_int)

    depth_ind = big_sliding_window(p.depth)
    ct_ind = big_sliding_window(p.cell_type)

    dry_or_land = (depth_ind .<= p.dry_depth) .| (ct_ind .== 2.0)
    weight_sfc[dry_or_land] .= 0.0
    weight_int[dry_or_land] .= 0.0

    norm_sfc = nansum_axis3(weight_sfc)
    for k in 1:9, j in 1:W, i in 1:L
        if norm_sfc[i, j] > 0
            weight_sfc[i, j, k] /= norm_sfc[i, j]
        end
    end
    norm_int = nansum_axis3(weight_int)
    for k in 1:9, j in 1:W, i in 1:L
        if norm_int[i, j] > 0
            weight_int[i, j, k] /= norm_int[i, j]
        end
    end

    weight = p.gamma .* weight_sfc .+ (1 - p.gamma) .* weight_int
    weight = (depth_ind .^ p.theta) .* weight
    weight[dry_or_land] .= 0.0

    zero_weights = repeat(nansum_axis3(weight) .<= 0, 1, 1, 9)
    deepest_cells = depth_ind .== repeat(max_axis3(depth_ind), 1, 1, 9)
    if p.stranding_handling == "park"
        # Cells flagged as land (cell_type == 2) are excluded from the
        # deepest-neighbor pick per slot; if every neighbor is land the plain
        # deepest rule applies. (Note: the dry-depth half of "dry or land" is
        # NOT excluded here — replicating the Python original, where this is
        # a known defect tracked separately.) All masks are (L, W, 9), so the
        # exclusion lands on the land slots only.
        not_land = ct_ind .!= 2.0
        any_not_land = any_axis3(not_land)
        masked = max_axis3(ifelse.(not_land, depth_ind, -Inf))
        deepest_dry_aware = (depth_ind .== repeat(masked, 1, 1, 9)) .& not_land
        deepest_cells = deepest_dry_aware .|
                        (deepest_cells .& repeat(.!any_not_land, 1, 1, 9))
    end
    choose = zero_weights .& deepest_cells
    weight[choose] .= 1.0

    clear_borders!(weight)
    p.weight = weight
    return p
end
