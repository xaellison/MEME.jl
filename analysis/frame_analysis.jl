using AbstractPlotting, GLMakie, Serialization, Unitful
using ProgressMeter


function main()

    data = collect(values(deserialize(".\\data\\1597857445")[:E]))[1]#[(:, :, :, 3)]
    tz = sort!(collect(keys(data)))
    scene = Scene()
    frame = map(ustrip, data[tz[end]][128:700, :, 1])
    #max_val = 10 * maximum(map(abs, map(ustrip, deserialize("D:\\ResearchData\\beep\\E\\100")[:, 1, :, 1])))
    heatmap!(scene, frame)
    AbstractPlotting.record(scene, "output.mp4", 1:length(tz)) do i

            frame = map(ustrip, data[tz[i]])[128:700, :, 1]
            #println(maximum(frame))
            frame = map(x->x |> ustrip|> abs, frame)
            #frame = map(x->min(max_val, abs(x)), frame)
            #frame = map(x -> sign(x), frame)
            #surface!(scene, 1:size(frame)[1], 1:size(frame)[2], (x, y) -> frame[x, y])
            heatmap!(scene, frame)
            #sleep(1)
    end
end

main()
function axis_comparison()
    data = deserialize(".\\data\\1597848717")[:E][(:, :, :)]
    tz = sort!(collect(keys(data)))
    scene = Scene()
    frame = map(ustrip, data[tz[end]][:, :, 1, 3])
    L = Int(size(frame)[1] / 2)
    vert = zip(temp[1:L, L, 1, 3], collect(1:L) .- L) |> collect |> x -> map(i->(ustrip(i[1]), i[2]), x)
    diag = zip(collect(temp[i, i, 1, 3] for i in 1:L), (collect(1:L) .- L)  .* sqrt(2)) |> collect |> x -> map(i->(ustrip(i[1]), i[2]), x)

    x_diag = collect(L-1:-1:0) .* sqrt(2)
    y_diag = collect(ustrip(frame[i, i]) for i in 1:L)
    x_vert = collect(L-1:-1:0)
    y_vert = collect(ustrip(frame[i, L]) for i in 1:L)
    lines!(scene, x_diag, y_diag, color=:red)
    lines!(scene, x_vert, y_vert, color=:blue)
end
#axis_comparison()
