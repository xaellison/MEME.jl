#using CUDAdrv, CUDAnative, CuArrays
#CuArrays.allowscalar(false)
using CUDA
#CUDA.allowscalar(false)


## Fundamental Maxwell's equations
function E_1_kernel(E, H, ε, σ_E, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    @inbounds c1 = (2 * ε[i, j, k] - σ_E[i, j, k] * Δt) /
                   (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds Δ = (H[i, j, k, 3] - H[i, mod(j - 2, X2) + 1, k, 3]) / Δx_2 -
                  (H[i, j, k, 2] - H[i, j, mod(k - 2, X3) + 1, 2]) / Δx_3
    @inbounds E[i, j, k, 1] = E[i, j, k, 1] * c1 + Δ * c2
    return nothing
end

function E_2_kernel(E, H, ε, σ_E, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    # c1 & c2 describe effects of lossy medium
    @inbounds c1 = (2 * ε[i, j, k] - σ_E[i, j, k] * Δt) /
                   (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds Δ = (H[i, j, k, 1] - H[i, j, mod(k - 2, X3) + 1, 1]) / Δx_3 -
                  (H[i, j, k, 3] - H[mod(i - 2, X1) + 1, j, k, 3]) / Δx_1
    @inbounds E[i, j, k, 2] = E[i, j, k, 2] * c1 + Δ * c2
    return nothing
end

function E_3_kernel(E, H, ε, σ_E, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    @inbounds c1 = (2 * ε[i, j, k] - σ_E[i, j, k] * Δt) /
                   (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * ε[i, j, k] + σ_E[i, j, k] * Δt)
    @inbounds Δ = (H[i, j, k, 2] - H[mod(i - 2, X1) + 1, j, k, 2]) / Δx_1 -
                  (H[i, j, k, 1] - H[i, mod(j - 2, X2) + 1, k, 1]) / Δx_2
    @inbounds E[i, j, k, 3] = E[i, j, k, 3] * c1 + Δ * c2
    return nothing
end

function H_1_kernel(H, E, μ, σ_m, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    # c1 & c2 describe effects of lossy medium
    @inbounds c1 = (2 * μ[i, j, k] - σ_m[i, j, k] * Δt) /
                   (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds Δ = (E[i, j, (k % X3) + 1, 2] - E[i, j, k, 2]) / Δx_3 -
                  (E[i, (j % X2) + 1, k, 3] - E[i, j, k, 3]) / Δx_2
    @inbounds H[i, j, k, 1] = H[i, j, k, 1] * c1  + Δ * c2
    return nothing
end


function H_2_kernel(H, E, μ, σ_m, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    # c1 & c2 describe effects of lossy medium
    @inbounds c1 = (2 * μ[i, j, k] - σ_m[i, j, k] * Δt) /
                   (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds Δ = (E[(i % X1) + 1, j, k, 3] - E[i, j, k, 3]) / Δx_1 -
                  (E[i, j, (k % X3) + 1, 1] - E[i, j, k, 1]) / Δx_3
    @inbounds H[i, j, k, 2] = H[i, j, k, 2] * c1 + Δ * c2
    return nothing
end


function H_3_kernel(H, E, μ, σ_m, Δt, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    # c1 & c2 describe effects of lossy medium
    @inbounds c1 = (2 * μ[i, j, k] - σ_m[i, j, k] * Δt) /
                   (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds c2 = 2 * Δt / (2 * μ[i, j, k] + σ_m[i, j, k] * Δt)
    @inbounds Δ = (E[i, (j % X2) + 1, k, 1] - E[i, j, k, 1]) / Δx_2 -
                  (E[(i % X1) + 1, j, k, 2] - E[i, j, k, 2]) / Δx_1
    @inbounds H[i, j, k, 3] = H[i, j, k, 3] * c1 + Δ * c2
    return nothing
end


## Kernels calculating values using neighboring cells
function divergence_kernel(src, dest, Δx_1, Δx_2, Δx_3, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[(i % X1) + 1, j, k, 1] - src[i, j, k, 1]) / Δx_1 +
                    (src[i, (j % X2) + 1, k, 2] - src[i, j, k, 2]) / Δx_2 +
                    (src[i, j, (k % X3) + 1, 3] - src[i, j, k, 3]) / Δx_3
    return nothing
end

function x_diff_kernel(src, dest, scalar, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] += (src[i, j, k, 1] - src[mod(i - 2, X1) + 1, j, k, 1]) * scalar
    dest[i, j, k, 2] += (src[i, j, k, 2] - src[mod(i - 2, X1) + 1, j, k, 2]) * scalar
    dest[i, j, k, 3] += (src[i, j, k, 3] - src[mod(i - 2, X1) + 1, j, k, 3]) * scalar
    return nothing
end


function y_diff_kernel(src, dest, scalar, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] += (src[i, j, k, 1] - src[i, mod(j - 2, X2) + 1, k, 1]) * scalar
    dest[i, j, k, 2] += (src[i, j, k, 2] - src[i, mod(j - 2, X2) + 1, k, 2]) * scalar
    dest[i, j, k, 3] += (src[i, j, k, 3] - src[i, mod(j - 2, X2) + 1, k, 3]) * scalar
    return nothing
end


function z_diff_kernel(src, dest, scalar, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] += (src[i, j, k, 1] - src[i, j, mod(k - 2, X3) + 1, 1]) * scalar
    dest[i, j, k, 2] += (src[i, j, k, 2] - src[i, j, mod(k - 2, X3) + 1, 2]) * scalar
    dest[i, j, k, 3] += (src[i, j, k, 3] - src[i, j, mod(k - 2, X3) + 1, 3]) * scalar
    return nothing
end


function x_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] + src[mod(i - 2, X1) + 1, j, k]) / 2
    return nothing
end


function y_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] + src[i, mod(j - 2, X2) + 1, k]) / 2
    return nothing
end


function z_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] - src[i, j, mod(k - 2, X3) + 1]) / 2
    return nothing
end

function yz_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] +
                        src[i, mod(j - 2, X2) + 1, k] +
                        src[i, j, mod(k - 2, X3) + 1] +
                        src[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1]
                        ) / 4
    return nothing
end

function zx_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] +
                     src[mod(i - 2 , X1) + 1, j, k] +
                     src[i, j, mod(k - 2 , X3) + 1] +
                     src[mod(i - 2 , X1) + 1, j, mod(k - 2, X3) + 1]
                     ) / 4
    return nothing
end

function xy_avg_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k] = (src[i, j, k] +
                     src[i, mod(j - 2 , X2) + 1, k] +
                     src[mod(i - 2, X1) + 1, j, k] +
                     src[mod(i  - 2, X1) + 1, mod(j - 2, X2) + 1, k]
                     ) / 4
    return nothing
end

#

function yz_avg_kernel3(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] = (src[i, j, k, 1] +
                        src[i, mod(j - 2, X2) + 1, k, 1] +
                        src[i, j, mod(k - 2, X3) + 1, 1] +
                        src[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1, 1]
                        ) / 4
    dest[i, j, k, 2] = (src[i, j, k, 2] +
                        src[i, mod(j - 2, X2) + 1, k, 2] +
                        src[i, j, mod(k - 2, X3) + 1, 2] +
                        src[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1, 2]
                        ) / 4
    dest[i, j, k, 3] = (src[i, j, k, 3] +
                        src[i, mod(j - 2, X2) + 1, k, 3] +
                        src[i, j, mod(k - 2, X3) + 1, 3] +
                        src[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1, 3]
                        ) / 4
    return nothing
end

function zx_avg_kernel3(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] = (src[i, j, k, 1] +
                     src[mod(i - 2, X1) + 1, j, k, 1] +
                     src[i, j, mod(k - 2, X3) + 1, 1] +
                     src[mod(i - 2, X1) + 1, j, mod(k - 2, X3) + 1, 1]
                     ) / 4
    dest[i, j, k, 2] = (src[i, j, k, 2] +
                  src[mod(i - 2, X1) + 1, j, k, 2] +
                  src[i, j, mod(k - 2, X3) + 1, 2] +
                  src[mod(i - 2, X1) + 1, j, mod(k - 2, X3) + 1, 2]
                  ) / 4

    dest[i, j, k, 3] = (src[i, j, k, 3] +
               src[mod(i - 2, X1) + 1, j, k, 3] +
               src[i, j, mod(k - 2, X3) + 1, 3] +
               src[mod(i - 2, X1) + 1, j, mod(k - 2, X3) + 1, 3]
               ) / 4

    return nothing
end

function xy_avg_kernel3(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] = (src[i, j, k, 1] +
                     src[i, mod(j - 2, X2) + 1, k, 1] +
                     src[mod(i - 2, X1) + 1, j, k, 1] +
                     src[mod(i - 2, X1) + 1, mod(j - 2, X2) + 1, k, 1]
                     ) / 4

     dest[i, j, k, 2] = (src[i, j, k, 2] +
                      src[i, mod(j - 2, X2) + 1, k, 2] +
                      src[mod(i - 2, X1) + 1, j, k, 2] +
                      src[mod(i - 2, X1) + 1, mod(j - 2, X2) + 1, k, 2]
                      ) / 4

      dest[i, j, k, 3] = (src[i, j, k, 3] +
                       src[i, mod(j - 2, X2) + 1, k, 3] +
                       src[mod(i - 2, X1) + 1, j, k, 3] +
                       src[mod(i - 2, X1) + 1, mod(j - 2, X2) + 1, k, 3]
                       ) / 4
    return nothing
end

function oct_average_up_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] = (src[i, j, k, 1] +
                     src[(i % X1) + 1, j, k, 1] +
                     src[i, (j % X2) + 1, k, 1] +
                     src[(i % X1) + 1, (j % X2) + 1, k, 1] +
                     src[i, j, (k % X3) + 1, 1] +
                     src[(i % X1) + 1, j, (k % X3) + 1, 1] +
                     src[i, (j % X2) + 1, (k % X3) + 1, 1] +
                     src[(i % X1) + 1, (j % X2) + 1, (k % X3) + 1, 1]
                    ) / 8
    dest[i, j, k, 2] = (src[i, j, k, 2] +
                     src[(i % X1) + 1, j, k, 2] +
                     src[i, (j % X2) + 1, k, 2] +
                     src[(i % X1) + 1, (j % X2) + 1, k, 2] +
                     src[i, j, (k % X3) + 1, 2] +
                     src[(i % X1) + 1, j, (k % X3) + 1, 2] +
                     src[i, (j % X2) + 1, (k % X3) + 1, 2] +
                     src[(i % X1) + 1, (j % X2) + 1, (k % X3) + 1, 2]
                    ) / 8
    dest[i, j, k, 3] = (src[i, j, k, 3] +
                     src[(i % X1) + 1, j, k, 3] +
                     src[i, (j % X2) + 1, k, 3] +
                     src[(i % X1) + 1, (j % X2) + 1, k, 3] +
                     src[i, j, (k % X3) + 1, 3] +
                     src[(i % X1) + 1, j, (k % X3) + 1, 3] +
                     src[i, (j % X2) + 1, (k % X3) + 1, 3] +
                     src[(i % X1) + 1, (j % X2) + 1, (k % X3) + 1, 3]
                    ) / 8
    return nothing
end

function vector_interpolate_up_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    dest[i, j, k, 1] = (src[(i % X1) + 1, j, k, 1] + src[i, j, k, 1]) / 2
    dest[i, j, k, 2] = (src[i, (j % X2) + 1, k, 2] + src[i, j, k, 2]) / 2
    dest[i, j, k, 3] = (src[i, j, (k % X3) + 1, 3] + src[i, j, k, 3]) / 2
    return nothing
end

function vector_interpolate_down_kernel(src, dest, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0

    dest[i, j, k, 1] = (src[mod(i - 2, X1) + 1, j, k, 1] + src[i, j, k, 1]) / 2
    dest[i, j, k, 2] = (src[i, mod(j - 2, X2) + 1, k, 2] + src[i, j, k, 2]) / 2
    dest[i, j, k, 3] = (src[i, j, mod(k - 2, X3) + 1, 3] + src[i, j, k, 3]) / 2
    return nothing
end

## Nonlinear Drude specific

function J1_kernel(J1, J, E, div_E, ρ0, qm, ε0, τ, Δt, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    c1 = (τ - Δt / 2) / (τ + Δt / 2)
    c2 = τ * Δt / (τ + Δt / 2) * ε0
    # we need to interpolate divergence in the direction of the component of J
    cx = c2 * qm * (ρ0[i, j, k] / ε0 + (div_E[i, j, k] + div_E[mod(i - 2, X1) + 1, j, k]) / 2)
    cy = c2 * qm * (ρ0[i, j, k] / ε0 + (div_E[i, j, k] + div_E[i, mod(j - 2, X2) + 1, k]) / 2)
    cz = c2 * qm * (ρ0[i, j, k] / ε0 + (div_E[i, j, k] + div_E[i, j, mod(k - 2, X3) + 1]) / 2)
    J1[i, j, k, 1] = c1 * J[i, j, k, 1] + cx * E[i, j, k, 1]
    J1[i, j, k, 2] = c1 * J[i, j, k, 2] + cy * E[i, j, k, 2]
    J1[i, j, k, 3] = c1 * J[i, j, k, 3] + cz * E[i, j, k, 3]
    return nothing
end

function H_hat_kernel(H_hat, H_prev, H, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    #=WARNING
    Notable break from syntax of the paper. Inan and Liu use opposite
    conventions for half-integer indexing of H vs E. Thus, we subtract our
    H indices for H_hat, rather than add.
    =#
    H_hat[i, j, k, 1] = (H[i, j, k, 1] +
                         H[i, mod(j - 2, X2) + 1, k, 1] +
                         H[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1, 1] +
                         H[i, j, mod(k - 2, X3) + 1, 1] +
                         H_prev[i, j, k, 1] +
                         H_prev[i, mod(j - 2, X2) + 1, k, 1] +
                         H_prev[i, mod(j - 2, X2) + 1, mod(k - 2, X3) + 1, 1] +
                         H_prev[i, j, mod(k - 2, X3) + 1, 1]
                         ) / 8
    H_hat[i, j, k, 2] = (H[i, j, k, 2] +
                         H[mod(i - 2, X1) + 1, j, k, 2] +
                         H[mod(i - 2, X1) + 1, j, mod(k - 2, X3) + 1, 2] +
                         H[i, j, mod(k - 2, X3) + 1, 2] +
                         H_prev[i, j, k, 2] +
                         H_prev[mod(i - 2, X1) + 1, j, k, 2] +
                         H_prev[mod(i - 2, X1) + 1, j, mod(k - 2, X3) + 1, 2] +
                         H_prev[i, j, mod(k - 2, X3) + 1, 2]
                         ) / 8

    H_hat[i, j, k, 3] = (H[i, j, k, 3] +
                         H[mod(i - 2, X1) + 1, j, k, 3] +
                         H[mod(i - 2, X1) + 1, mod(j - 2, X2) + 1, k, 3] +
                         H[i, mod(j - 2, X2) + 1, k, 3] +
                         H_prev[i, j, k, 3] +
                         H_prev[mod(i - 2, X1) + 1, j, k, 3] +
                         H_prev[mod(i - 2, X1) + 1, mod(j - 2, X2) + 1, k, 3] +
                         H_prev[i, mod(j - 2, X2) + 1, k, 3]
                         ) / 8
    return nothing
end

"""
Note args H and J are properly H_hat and J_hat
Stores in J2
"""
function J2_kernel(J, H, J2, a, i0, j0, k0, X1, X2, X3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0

    Hx = H[i, j, k, 1]
    Hy = H[i, j, k, 2]
    Hz = H[i, j, k, 3]

    det = 1 + a ^ 2 * (Hx ^ 2 + Hy ^ 2 + Hz ^ 2)

    M11 = 1 + a ^ 2 * (Hx ^ 2 - Hy ^ 2 - Hz ^ 2)
    M22 = 1 + a ^ 2 * (Hy ^ 2 - Hz ^ 2 - Hx ^ 2)
    M33 = 1 + a ^ 2 * (Hz ^ 2 - Hx ^ 2 - Hy ^ 2)

    M12 = 2 * a * (a * Hx * Hy + Hz)
    M13 = 2 * a * (a * Hz * Hx - Hy)
    M21 = 2 * a * (a * Hx * Hy - Hz)
    M23 = 2 * a * (a * Hy * Hz + Hx)
    M31 = 2 * a * (a * Hz * Hx + Hy)
    M32 = 2 * a * (a * Hy * Hz - Hx)

    J2[i, j, k, 1] = (M11 * J[i, j, k, 1] + M12 * J[i, j, k, 2] + M13 * J[i, j, k, 3])
    J2[i, j, k, 2] = (M21 * J[i, j, k, 1] + M22 * J[i, j, k, 2] + M23 * J[i, j, k, 3])
    J2[i, j, k, 3] = (M31 * J[i, j, k, 1] + M32 * J[i, j, k, 2] + M33 * J[i, j, k, 3])

    return nothing
end

function safe_division_kernel(dest,
                              component,
                              numerator ,
                              denominator,
                              detector,
                              dest_zero :: T3,
                              detector_zero :: T2,
                              i0, j0, k0, X1, X2, X3) where {T2, T3}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + i0
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y + j0
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z + k0
    # WARNING this only makes sense in a specific context
    if detector[i, j, k] == detector_zero
        dest[i, j, k, 1] = dest_zero
        dest[i, j, k, 2] = dest_zero
        dest[i, j, k, 3] = dest_zero
    else
        dest[i, j, k, 1] = numerator[i, j, k, 1] * numerator[i, j, k, component] / denominator[i, j, k]
        dest[i, j, k, 2] = numerator[i, j, k, 2] * numerator[i, j, k, component] / denominator[i, j, k]
        dest[i, j, k, 3] = numerator[i, j, k, 3] * numerator[i, j, k, component] / denominator[i, j, k]
    end
    return nothing
end
