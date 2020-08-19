#Amazingly, we can have Unitful quantities and run them on GPU!
using CUDAdrv, CUDAnative, CuArrays
using Unitful

a = [Float32(1.), Float32(2.)]u"V"
b = [Float32(3.), Float32(4.)]u"V"
c = [Float32(0.), Float32(0.)]u"V"
c_a = CuArray(a)
c_b = CuArray(b)
c_c = CuArray(c)
function k(a, b, c)
    idx = threadIdx().x
    c[idx] = a[idx] - b[idx]
    return nothing
end
@cuda blocks=1 threads=1 k(c_a, c_b, c_c)
synchronize()
println(Array(c_c))
