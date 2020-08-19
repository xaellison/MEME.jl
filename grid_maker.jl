#Copyright 2020 Alexander Ellison
using Logging
using Primes
using Test

function reassemble_factors(factorizations, assignments)
    out = ones(length(factorizations))
    for i in 1:length(factorizations)
        for j in 1:length(factorizations[i])
            if assignments[i][j] > 0
                out[i] *= factorizations[i][j][1] ^ assignments[i][j]
            end
        end
    end
    return out
end

function weight(factorizations, assignments)
    return prod(reassemble_factors(factorizations, assignments))
end

function opt!(factorizations, cap, dest, temp; inner_index=1, outer_index=1)

    factorization = factorizations[outer_index]

    n, n_max = factorization[inner_index]

    # for all the times we can include this factor, from `outer_index`-th number
    for i in 0:n_max
        temp[outer_index][inner_index] = i
        # if we're on the last factor of the `outer_index`-th number...
        if inner_index == length(factorization)
            # if we're on the last number which can be factored
            if outer_index == length(factorizations)
                # if this is the best collection of factors yet found
                if weight(factorizations, dest) < weight(factorizations, temp) <= cap
                    # deepcopy() isn't being helpful
                    for i in 1:length(temp)
                        for j in 1:length(temp[i])
                            dest[i][j] = temp[i][j]
                        end
                    end
                end
            else
                # start on the first factor of the next number
                opt!(factorizations, cap, dest, temp;
                    inner_index=1,
                    outer_index=outer_index+1)
            end
        else
            # go on to the next factor of this number
            opt!(factorizations, cap, dest, temp;
                inner_index=inner_index + 1,
                outer_index=outer_index)
        end
    end
end


function fit(size_tuple, cap)
    dicts = collect(map(x -> factor(Dict, x), size_tuple))
    tuples = Tuple(collect(Tuple(collect((k, d[k]) for k in keys(d))) for d in dicts))
    tuples = map(x -> x == () ? ((1, 1),) : x, tuples)
    dest = collect(zeros(length(a)) for a in tuples)
    temp = collect(zeros(length(a)) for a in tuples)
    opt!(tuples, cap, dest, temp)
    return reassemble_factors(tuples, dest)
end

function validate(tuple, cap)
    solution = fit(tuple, cap)
    @test prod(solution) <= cap
    for (i, x) in enumerate(solution)
        @test x <= tuple[i]
    end
end

validate((1024,), 512)
validate((600, 1600), 1024)
validate((20, 30, 40), 760)
validate((20, 30, 40), 300)


function my_grid(dims; thread_max=1024, block_max=32)
    my_threads = map(Int, fit(dims, thread_max))
    total_blocks = map(Int, dims ./ my_threads)
    launch_blocks = fit(map(Int, dims ./ my_threads), block_max)
    loops = map(Int, total_blocks ./ launch_blocks)
    launch_blocks = map(Int, launch_blocks)
    @assert Tuple(my_threads .* launch_blocks .* loops) == dims
    return map(Tuple, (loops, launch_blocks, my_threads))
end

#=
my_grid((1600, 600))
my_grid((1600, 500))
my_grid((200, 320, 128))
my_grid((2000,))
=#
