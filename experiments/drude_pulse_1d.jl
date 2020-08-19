include("../3D.jl")

function main()
      T = Float32

      physical_constants = Dict(:qm => T(-1.75882f11)u"C/kg", # electron charge mass ratio
                                :vp => T(299792000)u"m/s", # phase velocity of light
                                :ε0 => T(8.8541878f-12)u"F/m", # permitivitty free space
                                :μ0 => T(4 * pi * 10 ^ -7)u"N/A^2", # permeability free space
                                :q  => T(-1.602176634f-19)u"C", # electron charge
                                )

      metal_properties = Dict(:τ => T(1.5f-14)u"s", # relaxation time
                              :ρ0 => T(-1f10)u"C/m^3", # bulk carrier charge density
                              )
      ωp = sqrt(physical_constants[:qm] *
                                  metal_properties[:ρ0] / physical_constants[:ε0])

      Δx = 2.0f0u"nm"
      Δt = Δx / (physical_constants[:vp] * 64)
      runtime = 125f0u"fs"
      STEPS = Int(round(runtime / Δt))
      drive_ω = 2 * pi * physical_constants[:vp] / 1200f0u"nm"

      periodicity = (false, true, true)
      desired_size = (512, 1, 1)

      drive_signal(step, Δt) = 2f7u"V/m" * sin(drive_ω * step * Δt) * exp(-((step * Δt - runtime / 2) / (runtime / 10)) ^ 2)
      result = simulate(merge(physical_constants, metal_properties),
                        periodicity,
                        desired_size,
                        90,
                        Δx,
                        Δt,
                        STEPS,
                        [((128, 0, 0), (256, 1, 1))],
                        [],
                        nothing,
                        0u"N/m",
                        0f1u"N/m^2",
                        6,
                        (100:101, :, :, 2),
                        drive_signal,
                        (((:, :, :, :), 32),),
                        (((:, :, :, :), 32),);
                        #linear_only=true
                        )
      serialize("data/$(Int(ceil(time())))", result)
end

main()
