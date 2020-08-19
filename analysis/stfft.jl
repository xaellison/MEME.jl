"""
Short time FFT. Weights input signal by a Gaussian. Useful for producing
spectrograms
"""

using FFTW
using Makie
using Serialization
using Unitful


function stfft(array, σ; overlap=1)
    N = length(array)

    n_std_dev = 5
    # window size
    W = 2 * n_std_dev * σ
    overlap_int = Int(round(σ * overlap))
    center = n_std_dev * σ
    kernel = collect(exp(-0.5 * ((i - center) / σ) ^ 2 ) for i in 1:W+1)
    samples = 2 * center:overlap_int:N - 2 * center

    window(x) = array[x - Int(W / 2):x + Int(W / 2)] .* kernel
    output = Array{ComplexF32}(undef, length(samples), Int(W / 2) + 1)

    for (i, c) in enumerate(samples)
        output[i, :] = rfft(window(c))
    end
    return output
end
