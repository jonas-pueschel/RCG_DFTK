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

pgfplotsx()

molecule = "GaAs"

BSON.@load "result_comp_$molecule.bson" ra

n_electrons = molecule == "TiO2" ? 16 : 4

methods = ["EARCG", "H1RCG", "SCF", "DCM"]
n_runs = 20
#n_runs = 1

result_mol = ra

#TODO: calculate avg percentage of Ham

#x: iterations

# for method_name = methods
#     maxiter[mol] = 0
#     for i = 1:n_runs
#         cb = result_mol[method_name][i][1]
#         maxiter[mol] = max(cb.n_iter - 1, maxiter[mol])
#     end

#     vals = fill(NaN, )
# end

i_run = Int(round(rand() * 20) + 1)

println(i_run)

#initial residual, not saved by the experiment
initval = 0.18141271102731157 / 1.5

for method_name = methods
    cb = result_mol[method_name][i_run][1]
    print("$method_name: ");
    println(cb.time_DftHamiltonian/cb.times[end]);
end

plt1 = plot(; yscale = :log, ylabel = L"\|r^{(k)}\|_F", xlabel = "iterations", ymin = 1e-9, ymax = 2e-1, yticks = [1e-2, 1e-4,1e-6,1e-8], height = 100, width = 60)
for method_name = methods
    cb = result_mol[method_name][i_run][1]
    resls = cb.norm_residuals[1:(cb.n_iter - 1)]
    plot!(0:(cb.n_iter - 1), [initval, resls...], label = method_name)
end
plot!()

plt2 = plot(; yscale = :log, ylabel = L"\|r^{(k)}\|_F", xlabel = "calls DftHamiltonian", ymin = 1e-9, ymax = 2e-1, yticks = [1e-2, 1e-4,1e-6,1e-8])
for method_name = methods
    cb = result_mol[method_name][i_run][1]
    resls = cb.norm_residuals[1:(cb.n_iter - 1)]
    hams = cb.calls_DftHamiltonian[1:(cb.n_iter - 1)]/n_electrons
    plot!([0.0, hams...], [initval, resls...], label = method_name)
end
plot!()

plt3 = plot(; yscale = :log, ylabel = L"\|r^{(k)}\|_F", xlabel = "CPU time (ns)", ymin = 1e-9, ymax = 2e-1, yticks = [1e-2, 1e-4,1e-6,1e-8])
for method_name = methods
    cb = result_mol[method_name][i_run][1]
    resls = cb.norm_residuals[1:(cb.n_iter - 1)]
    times = cb.times[1:(cb.n_iter - 1)] 
    if method_name == "EARCG"
        diff = times[1] /3 * 2
        println(diff)
        times = [time - diff for time = times]
    end
    plot!([0.0, times...], [initval, resls...], label = method_name)
end
plot!()

display(plt1)
display(plt2)
display(plt3)


savefig(plt1, "$molecule-plt1.tex")
savefig(plt2, "$molecule-plt2.tex")
savefig(plt3, "$molecule-plt3.tex")