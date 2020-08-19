#Copyright 2020 Alexander Ellison
include("3D_kernels.jl")
include("grid_maker.jl")
include("field_config.jl")
include("lax_wendroff.jl")

using Unitful
using Logging
using ProgressMeter
using Serialization

function run_large_grid(kernel, args, dims, offsets, grid)
    loops, my_blocks, my_threads = grid
    X, Y, Z = dims
    for x_loop in 1:loops[1], y_loop in 1:loops[2], z_loop in 1:loops[3]
        x0, y0, z0 = offsets .+ my_blocks .* my_threads .*
                     # subtracting (1,1,1) is cost of loop ranges as defined
                     ((x_loop, y_loop, z_loop) .- (1, 1, 1))
        grid_args = (x0, y0, z0, X, Y, Z)
        @cuda blocks=my_blocks threads=my_threads kernel(args..., grid_args...)
    end
end

function simulate(physical_values,
                  periodicity,
                  desired_size,
                  PML_thickness,
                  Δx,
                  Δt,
                  stop,
                  metal_boxes,
                  oscillator_boxes,
                  dielectric,
                  oscillator_force,
                  oscillator_force_2,
                  fuzz,
                  E_drive_region,
                  drive_signal,
                  E_samples,
                  J_samples;
                  T=Float32,
                  linear_only=false,
                  remarks=nothing
                )
    q  = physical_values[:q]
    qm = physical_values[:qm]
    vp = physical_values[:vp]
    ε0 = physical_values[:ε0]
    μ0 = physical_values[:μ0]
    ρ0 = physical_values[:ρ0]
    τ  = physical_values[:τ]

    Δy = Δz = Δx

    wp = sqrt(qm * ρ0 / ε0)

    @assert 1 > (τ - Δt / 2) / (τ + Δt / 2) > 0.97
    @info "PML thickness: $(Δx * PML_thickness)"
    @info "Metal edge thickness: $(Δx * fuzz)"

    offsets = map(x -> x ? 0 : 1, periodicity)
    scalar_size = desired_size .+ offsets .* 2
    vector_size = (scalar_size..., 3)

    type_of_J = typeof(T(0)u"A/m^2")
    type_of_H = typeof(T(0)u"A/m")

    # Electromagnetic fields
    E = CuArray(zeros(typeof(T(0)u"V/m"), vector_size))
    H = CuArray(zeros(type_of_H, vector_size))
    H_prev = copy(H)

    # mechanical dielectrics
    P = CuArray(zeros(typeof(1u"C/m^2"), vector_size))
    P2 = CuArray(zeros(eltype(P), vector_size))
    V = CuArray(zeros(typeof(1u"C*m^-2*s^-1"), vector_size))
    Ω = oscillators(T, scalar_size, oscillator_boxes)

    # Current
    J = CuArray(zeros(type_of_J, vector_size))
    # carrier density
    ρ = metal_box_carriers(T, scalar_size, metal_boxes, ρ0, fuzz)
    ρ′ = copy(ρ)

    # simple constitutive values/fields
    ε   = isnothing(dielectric) ? CuArray(ones(T, scalar_size)) * ε0 : dielectric(scalar_size)
    μ   = CuArray(ones(T, scalar_size)) * μ0
    σ_e, σ_m = map(CuArray, PML(sqrt(ε0 / μ0), Δx, scalar_size, periodicity, PML_thickness))

    # work space fields
    div_E = CuArray(zeros(typeof(T(0)u"V/m^2"), scalar_size))
    J1    = CuArray(zeros(type_of_J, vector_size))
    J2    = CuArray(zeros(type_of_J, vector_size))
    J3    = CuArray(zeros(type_of_J, vector_size))
    J_hat = CuArray(zeros(type_of_J, vector_size))
    H_hat = CuArray(zeros(type_of_H, vector_size))

    export_data = Dict(:meta => Dict(:carriers => Array(ρ),
                                     :pml => (Array(σ_e), Array(σ_m)),
                                     :remarks => remarks,
                                     :linear_only => linear_only,
                                    ),
                       :E => Dict(region[1] => Dict() for region in E_samples),
                       :J => Dict(region[1] => Dict() for region in J_samples),
                       :dielectric => Array(ε)
                      )

    # NB if debugging julia with flag -g2 seems like these need to be halved
    grid_1024 = my_grid(desired_size; thread_max=1024)
    grid_512 = my_grid(desired_size; thread_max=512)

    Lax = LaxWendroff(J2, ρ′, ρ)

    @info "Running for $stop steps"
    t0 = time()
    toprint = false
    function execute_time_step(step)
        current_value = drive_signal(step - 1, Δt)
        target_value  = drive_signal(step, Δt)
        V += (((E .* Ω) .* qm .+ P .* (oscillator_force / ρ0 / (q/qm) ))) * Δt * ρ0 #.+ P .* P .* oscillator_force_2 / (ρ0 * q / qm) * Δt
        P += V * Δt
        E[E_drive_region...] .+= target_value - current_value
        run_large_grid(E_1_kernel, (E, H, ε, σ_e, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)
        run_large_grid(E_2_kernel, (E, H, ε, σ_e, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)
        run_large_grid(E_3_kernel, (E, H, ε, σ_e, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)

        H_prev .= H
        E .= E - Δt / ε0 * (J + V)
        run_large_grid(H_1_kernel, (H, E, μ, σ_m, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)
        run_large_grid(H_2_kernel, (H, E, μ, σ_m, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)
        run_large_grid(H_3_kernel, (H, E, μ, σ_m, Δt, Δx, Δy, Δz), desired_size, offsets, grid_1024)
        # Linear part of Drude model
        if length(metal_boxes) > 0
            run_large_grid(divergence_kernel, (E, div_E, Δx, Δy, Δz), desired_size, offsets, grid_1024)
            run_large_grid(J1_kernel, (J1, J, E, div_E, ρ, qm, ε0, τ, Δt), desired_size, offsets, grid_512)

            if linear_only
                J, J1 = J1, J
            else

                # non-linear part of Drude model

                # J_hat will be spatially colocated with H
                run_large_grid(vector_interpolate_up_kernel, (J1, J_hat), desired_size, offsets, grid_1024)
                # H_hat will be both spatially and temporally colocated with J_hat
                run_large_grid(H_hat_kernel, (H_hat, H_prev, H), desired_size, offsets, grid_512)
                run_large_grid(J2_kernel, (J_hat, H_hat, J2, qm * μ0 * Δt / 2), desired_size, offsets, grid_512)
                #run_large_grid(vector_interpolate_down_kernel, (J2, J), desired_size, offsets, grid_1024)

                ρ′ .= ε0 * div_E + ρ

                lax_wendroff(Lax, Δx, Δy, Δz, Δt, desired_size, offsets, grid_1024, grid_512)
                #println((sum(map(x->x^2,Lax.output)),sum(map(x->x^2,Lax.W)), J2 === Lax.W))
                #sleep(0.02)
                J1 .-= Lax.output
                # interpolate J back to E
                run_large_grid(vector_interpolate_down_kernel, (J1, J), desired_size, offsets, grid_1024)

            end
        end
    end


    @showprogress 1 for step in 1:stop
        execute_time_step(step)

        for (f, field, samples) in ((:E, E, E_samples), (:J, J, J_samples))
            for (region, period) in samples
                if 1 >= step % period >= 0
                    synchronize()
                    @assert !any(isnan, Array(field))
                    export_data[f][region][step * Δt] = Array(field[region...])
                end
            end
        end
    end

    @info "~fin"
    return export_data
end
