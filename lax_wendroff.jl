
struct LaxWendroff
    W
    ρ′
    ρ
    interpolated_ρ′
    interpolated_ρ
    F̃
    G̃
    H̃
    W_half
    spatial_avg_temp
    output
end

function LaxWendroff(W :: CuArray{T1}, ρ′ :: CuArray{T2}, ρ :: CuArray{T2}) where {T1, T2}
    return LaxWendroff(W, ρ′, ρ,
                      CuArray{T2}(undef, size(ρ)...),
                      CuArray{T2}(undef, size(ρ)...),
                      CuArray{typeof(zero(T1) ^ 2 / zero(T2))}(undef, size(W)...),
                      CuArray{typeof(zero(T1) ^ 2 / zero(T2))}(undef, size(W)...),
                      CuArray{typeof(zero(T1) ^ 2 / zero(T2))}(undef, size(W)...),
                      CuArray{T1}(undef, size(W)...),
                      CuArray{T1}(undef, size(W)...),
                      CuArray{T1}(undef, size(W)...))
end

function safe_divide!(dest, component, numerator :: CuArray{T1}, denominator :: CuArray{T2}, detector, dims, offsets, grid) where {T1, T2}
    T3 = typeof(zero(T1) ^ 2 / zero(T2))
    run_large_grid(safe_division_kernel, (dest, component, numerator, denominator, detector, zero(T3), zero(T2)), dims, offsets, grid)
    return dest
end


"""
This computes a timestep stored in LaxWendroff.output which is a delta
"""
function lax_wendroff(L :: LaxWendroff, Δx, Δy, Δz, Δt, desired_size, offsets, big_grid, little_grid)
    # clear output
    L.output .*= 0

    # First step spatial functionals
    run_large_grid(x_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(x_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    safe_divide!(L.F̃, 1, L.W, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)

    run_large_grid(y_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(y_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    safe_divide!(L.G̃, 2, L.W, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)

    run_large_grid(z_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(z_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    safe_divide!(L.H̃, 3, L.W, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)
    #println(("lax: ", minimum(L.F̃), minimum(L.G̃), minimum(L.H̃)))
    run_large_grid(oct_average_up_kernel, (L.W, L.W_half), desired_size, offsets, little_grid)

    run_large_grid(x_diff_kernel, (L.F̃, L.W_half, Δt / (2 * Δx)), desired_size, offsets, little_grid)
    run_large_grid(y_diff_kernel, (L.G̃, L.W_half, Δt / (2 * Δx)), desired_size, offsets, little_grid)
    run_large_grid(z_diff_kernel, (L.H̃, L.W_half, Δt / (2 * Δx)), desired_size, offsets, little_grid)
    #println(("lax2: ", minimum(L.W_half)))
    # Compute second step spatial functionals
    run_large_grid(x_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(x_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    run_large_grid(yz_avg_kernel3, (L.W_half, L.spatial_avg_temp), desired_size, offsets, little_grid)
    safe_divide!(L.F̃, 1, L.W_half, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)

    run_large_grid(y_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(y_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    run_large_grid(zx_avg_kernel3, (L.W_half, L.spatial_avg_temp), desired_size, offsets, little_grid)
    safe_divide!(L.G̃, 2, L.W_half, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)

    run_large_grid(z_avg_kernel, (L.ρ′, L.interpolated_ρ′), desired_size, offsets, little_grid)
    run_large_grid(z_avg_kernel, (L.ρ, L.interpolated_ρ), desired_size, offsets, little_grid)
    run_large_grid(xy_avg_kernel3, (L.W_half, L.spatial_avg_temp), desired_size, offsets, little_grid)
    safe_divide!(L.H̃, 3, L.W_half, L.interpolated_ρ′, L.interpolated_ρ, desired_size, offsets, little_grid)

    # assign to output
    run_large_grid(x_diff_kernel, (L.F̃, L.output, 1 * Δt / Δx), desired_size, offsets, little_grid)
    run_large_grid(y_diff_kernel, (L.G̃, L.output, 1 * Δt / Δx), desired_size, offsets, little_grid)
    run_large_grid(z_diff_kernel, (L.H̃, L.output, 1 * Δt / Δx), desired_size, offsets, little_grid)

    return
end
