using Pkg
Pkg.activate("./src")
Pkg.instantiate()
using StatsPlots
using PGFPlotsX
using LaTeXStrings
using BSON
include("../rcg.jl")
include("../setups/all_setups.jl")
include("../rcg_benchmarking.jl")



BSON.@load "result_small_gap_2.bson" results





N_steps = 28
n_runs = 20
tol = 1e-8

a_min = 10.0
a_max = 11.75
vals = fill(NaN, N_steps + 1, n_runs, length(results[a_min]) - 1)
xs = []

methods = []
for j in 1:(length(keys(results[a_min])) - 1)
    method_name = collect(keys(results[a_min]))[j]
    push!(methods, method_name)
end

conv_runs = fill(0, N_steps + 1,  length(results[a_min]) - 1)

i = 1
for (a, result) in results
    push!(xs, a)

    j = 1
    for method_name = methods
        for k = 1:n_runs
            cb = result[method_name][k][1]
            val = cb.n_iter - 2
            if (cb.norm_residuals[cb.n_iter - 2] > tol)
                #extrapolate
                println(method_name)
                println(val)
                val *= log(tol)/log(cb.norm_residuals[cb.n_iter - 2])
                println(val)
                println("---")
            end

            if ((cb.norm_residuals[cb.n_iter - 2] < tol || cb.norm_residuals[cb.n_iter - 1] < tol) && all([!isnothing(cb.norm_residuals[cb.n_iter - l]) for l in [0,1,2,3]]))
                conv_runs[i, j] += 1
            end
            vals[i, k, j] = log2(val) 
        end
        j += 1
    end
    i += 1
end

# avgs = []
# j = 1
# for j in 1:(length(keys(results[10.0])) - 1)
#     method_name = collect(keys(result))[j]
#     avg = sum(results[a_min][method_name][k].n_iter - 2 for k = 1:n_runs)/(n_runs)
#     push!(avgs, avg)
# end
# j = 1
# for (method_name, method) in methods_rcg
#     #vals[:, :, j] *= 1/avgs[j]
#     j += 1
# end

p = sortperm(xs)
pgfplotsx()
plt1 = plot(;yticks = [3,4,5,6], xlabel = L"a", ylabel = L"\\log_2(iter)")
i = 1
for method_name = methods
    errorline!(xs[p], vals[p,:,i], errorstyle=:ribbon, label=method_name)
    i += 1
end
plot!()
plt2 = plot(; xlabel = L"a", ylabel = "Number of convergent runs")
for (i,method_name) = zip(1:length(methods), methods)
    plot!(xs[p], conv_runs[p,i],label=method_name)
    i += 1
end
plot!()

# ys = Array{Int64}([])
# for y in conv_runs[p, :]
#     push!(ys, y)
# end
# sx = repeat(methods , inner = N_steps + 1)
# nam = repeat(xs[p], outer = length(methods))
# nam = [a for a in nam]
# #groupedbar(Array(nam), Array(ys), group = Array(sx), ylabel = L"a", 
# #        title = "Number of convergent runs")
# groupedbar(nam, ys, group = sx, ylabel = L"a", 
#         title = "Number of convergent runs")

savefig(plt1, "plt1.tex")

savefig(plt2, "plt2.tex")

