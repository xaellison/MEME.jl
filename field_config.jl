#Copyright 2020 Alexander Ellison
function PML(η, Δx, dimensions, periodicity, thickness, m=3, R=10^-6)
    # TODO this implementation is lazy and slow
    unit_E = η / Δx
    unit_M = unit_E / η ^ 2
    T_E = typeof(unit_E)
    T_M = typeof(unit_M)
    σ_E = zeros(T_E, dimensions)
    σ_M = zeros(T_M, dimensions)
    σ0_E = - (m + 1) * log(R) / (2 * 1 * thickness) * unit_E
    σ0_M = - (m + 1) * log(R) / (2 * 1 * thickness) * unit_M

    for x in 1:dimensions[1], y in 1:dimensions[2], z in 1:dimensions[3]
        v = (x, y, z)
        candidates = []
        for dim in 1:3
            if !periodicity[dim]
                push!(candidates, v[dim])
                push!(candidates, dimensions[dim] + 1 - v[dim])
            end
        end
        filter!(x -> x <= thickness, candidates)

        if length(candidates) > 0
            σ_E[x, y, z] = ((thickness - minimum(candidates)) / thickness) ^ m * σ0_E
            σ_M[x, y, z] = ((thickness - minimum(candidates)) / thickness) ^ m * σ0_M
        end
    end
    return σ_E, σ_M
end

function fuzz(field, target, increment)
    delta = zeros(typeof(target), size(field))
    X, Y, Z = size(field)
    for x in 1:X, y in 1:Y, z in 1:Z
        if isapprox(field[x, y, z], zero(typeof(target)))
            if any(x -> isapprox(x, target), (field[mod(x - 2, X) + 1, y, z], field[x % X + 1, y, z],
                                              field[x, mod(y - 2, Y) + 1, z], field[x, y % Y + 1, z],
                                              field[x, y, mod(z - 2, Z) + 1], field[x, y, z % Z + 1]))
                 delta[x, y, z] = target - increment
            end
        end
    end
    return delta
end

function metal_box_carriers(T, world_size, boxes, carrier_density, fuzz_size)
    out = zeros(typeof(carrier_density), world_size)
    for (origin, dimensions) in boxes
        for x in 1:dimensions[1]
            for y in 1:dimensions[2]
                for z in 1:dimensions[3]
                    out[(origin .+ (x, y, z))...] = carrier_density
                end
            end
        end
    end

    my_zero = zero(typeof(carrier_density))
    target = carrier_density
    increment = carrier_density / fuzz_size

    for i in 1:fuzz_size
        out = out + fuzz(out, target, increment)
        target = carrier_density - i * increment
    end

    return CuArray(out)
end


function oscillators(T, world_size, boxes)
    out = zeros(T, world_size...)
    for (origin, dimensions) in boxes
        for x in 1:dimensions[1]
            for y in 1:dimensions[2]
                for z in 1:dimensions[3]
                    out[(origin .+ (x, y, z))...] = true
                end
            end
        end
    end
    return CuArray(out)
end


function thin_lens(R1, R2, x1, x2, n, T, scalar_size, ε0)

    out = ones(T, scalar_size) * ε0

    for i in 1:scalar_size[1]
        for j in 1:scalar_size[2]
            for k in 1:scalar_size[3]
                y = j - scalar_size[2] / 2
                if (i - x2) ^ 2 + y ^ 2 <= R2 ^ 2 && (i - x1) ^ 2 + y ^ 2 <= R1 ^ 2
                    out[i, j, k] *= n
                end
            end
        end
    end

    return CuArray(out)
end
