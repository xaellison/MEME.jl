include("../3D.jl")
function main()
      T = Float32

      physical_constants = Dict(:qm => T(-1.75882f11)u"C/kg", # electron charge mass ratio
                                :vp => T(299792000)u"m/s", # phase velocity of light
                                :ε0 => T(8.8541878f-12)u"F/m", # permitivitty free space
                                :μ0 => T(4 * pi * 10 ^ -7)u"N/A^2", # permeability free space
                                :q  => T(-1.602176634f-19)u"C",
                                )

      metal_properties = Dict(:τ => T(1.5f-14)u"s", # relaxation time
                              :ρ0 => T(-1f10)u"C/m^3", # bulk carrier charge density
                              )

      Δx = 2.0f0u"nm"
      Δt = Δx / (physical_constants[:vp] * 32)
      runtime = 1.2* 2048 * Δx  /physical_constants[:vp]
      STEPS = Int(round(runtime / Δt))
      drive_ω = 2 * pi * physical_constants[:vp] / (16 * Δx)

      periodicity = (false, false, true)
      desired_size = (2048, 1024, 1)

      drive_signal(step, Δt) = 1f1u"V/m" * sin(2 * pi * drive_ω * step * Δt) #* (-1 + 2 * exp(drive_freq * step * Δt / 30) / (exp(drive_freq * step * Δt / 30) + 1))

      result = simulate(merge(physical_constants, metal_properties),
                        periodicity,
                        desired_size,
                        64,
                        Δx,
                        Δt,
                        STEPS,
                        [],
                        [],
                        scalar_size->thin_lens(660f0, 792f0, 0f0, 2*643.5f0, 2, T, scalar_size, physical_constants[:ε0]),
                        0f1u"N/m",
                        0f1u"N/m^2",
                        6,
                        (80:81, 64:1024-64, :, 3),
                        drive_signal,
                        (((1:2:2048, 1:2:1024, :, 3), STEPS/200 - 1),),
                        ();
                        linear_only=true,
                        remarks="lens, drive ω = $drive_ω"
                        )
      serialize("data/$(Int(ceil(time())))", result)
end

main()
