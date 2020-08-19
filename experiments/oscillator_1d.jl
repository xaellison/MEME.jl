"""
https://www.nature.com/articles/s41598-018-21850-8
Quote:
We stress that we did not attempt to optimize our structures so as to maximize
the contribution of the bulk effects to the SHG. Nevertheless, our analysis
suggests that it is conceivable that, at least in the case of all-dielectric
nanoparticles, one can design structures for which the bulk effects are
comparable or even larger than the surface ones. This means that care must be
taken when experimental results pertaining to SHG in all-dielectric
nanostructures made of centrosymmetric materials are theoretically interpreted,
as our analysis suggests that the validity of the commonly used practice to
neglect the bulk contribution to SHG might break down in this instance.

This simulation produces a SHG in bulk.
"""

include("../3D.jl")

function main()
      T = Float32

      physical_constants = Dict(:qm => T(-1.75882f11)u"C/kg", # electron charge mass ratio
                                :vp => T(299792000)u"m/s", # phase velocity of light
                                :ε0 => T(8.8541878f-12)u"F/m", # permitivitty free space
                                :μ0 => T(4 * pi * 10 ^ -7)u"N/A^2", # permeability free space
                                :q  => T(-1.602176634f-19)u"C"
                                )

      metal_properties = Dict(:τ => T(1.5f-14)u"s", # relaxation time
                              :ρ0 => T(-1f10)u"C/m^3", # bulk carrier charge density
                              )
      ωp = sqrt(physical_constants[:qm] *
                                  metal_properties[:ρ0] / physical_constants[:ε0])


          metal_properties = Dict(:τ => T(1.5f-14)u"s", # relaxation time
                                  :ρ0 => T(-1f10)u"C/m^3" # bulk carrier charge density
                                  )


      Δx = 2.0f0u"nm"
      Δt = Δx / (physical_constants[:vp] * 32)
      runtime = 500f0u"fs"
      STEPS = Int(round(runtime / Δt))
      drive_ω = 2 * pi * physical_constants[:vp] / 512u"nm"

      periodicity = (false, true, true)
      desired_size = (2^14, 1, 1)

      drive_signal(step, Δt) = 2f1u"V/m" * sin(drive_ω * step * Δt) * exp(-((step * Δt - runtime / 2) / (runtime / 10)) ^ 2)
      result = simulate(merge(physical_constants, metal_properties),
                        periodicity,
                        desired_size,
                        512,
                        Δx,
                        Δt,
                        STEPS,
                        [],
                        [((7000, 0, 0), (8000, 1, 1))],
                        nothing,
                        -49u"N/m",
                        0f1u"N/m^2",
                        6,
                        (520:521, :, :, 2),
                        drive_signal,
                        (((:, :, :, :), STEPS/100-1), ((10000, 1, 1, :), 1)),
                        ();
                        linear_only=true,
                        remarks="nonlinear dielectric attempt, high temporal rez"
                        )
      serialize("./$(Int(ceil(time()))%1000)", result)
end

main()
