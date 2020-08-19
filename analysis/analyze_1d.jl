using AbstractPlotting, GLMakie, Serialization, Unitful

function gauss(E, x1, x2)
    return -E[x1, 1, 1, 1] + E[x2, 1, 1, 1]
end

raw = deserialize("./data/11th_hr_263")
meta = raw[:meta]
println(meta[:linear_only])
println(meta[:remarks])
data = raw[:E][:,:,:,:]

times = sort!(collect(keys(data)))
MAX = maximum(maximum(data[times[i]][:, 1, 1, 2]) for i in 1:4:length(times))
MIN = minimum(minimum(data[times[i]][:, 1, 1, 2]) for i in 1:4:length(times))
limit = FRect(0, MIN|>ustrip, size(data[times[1]])[1], ustrip(MAX-MIN))
frames = [collect(map(x->x|>ustrip, data[times[i]][:, 1, 1, 2])) for i in 1:1:length(times)]
#max_val = 10 * maximum(map(abs, map(ustrip, deserialize("D:\\ResearchData\\beep\\E\\100")[:, 1, :, 1])))
scene = lines(frames[1][:, 1], limits=limit)
s = scene[end] # last plot in scene
record(scene, "270nm.mp4", frames) do m

    s[1] = m[:, 1]
end
