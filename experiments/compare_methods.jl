using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

# this script forces precompliation for all methods, ensuring comparability in runtime
include("precompile_methods.jl");
precompile_methods();

model_name = "TiO2"

if model_name == "silicon"
    include("setups/silicon_setup.jl")
    # Silicon lattice constant in Bohr
    model, basis = silicon_setup(; Ecut = 30, kgrid = [4,4,4]);
elseif  model_name == "TiO2"
    include("setups/TiO2_setup.jl")
    model, basis = TiO2_setup(; Ecut = 60, kgrid = [2,2,2]);
elseif model_name == "GaAs"
    include("setups/GaAs_setup.jl")
    model, basis = GaAs_setup(; Ecut = 60, kgrid = [2,2,2]);
end


p = model.n_electrons/2

# Convergence tolerance
tol = 1.0e-8;

# number of runs per method, times_tot is averaged over these
n_runs = 10;
# seed for the RNG, reset before every run (DFTK fills missing initial bands with random vectors)
rng_seed = 1234;

default_callback = RcgDefaultCallback()


scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;


es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis)

init_norm_res = RCG_DFTK.norm_DFTK(basis, res0)
init_e = es0.total

#default callback
defaultCallback = RcgDefaultCallback();

# runs `run_method(callback)` n_runs times (same initial value) and returns a callback
# whose times_tot is averaged over all runs. The other fields are identical for all runs.
function run_averaged(name, run_method)
    println("\n$name")
    cbs_run = TrackResTimeCallback[]
    for i = 1:n_runs
        cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
        DFTK.Random.seed!(rng_seed)
        DFTK.reset_timer!(DFTK.timer)
        run_method(cb)
        println("Run $i/$n_runs, Time_tot (s): $((cb.times_tot[end])/ 1e9)")
        push!(cbs_run, cb)
    end
    n_iter = length(cbs_run[1].times_tot)
    all(length(cb.times_tot) == n_iter for cb in cbs_run) ||
        error("$name: runs have different iteration counts, cannot average times_tot")
    cb_avg = cbs_run[1]
    cb_avg.times_tot = sum(cb.times_tot for cb in cbs_run) ./ n_runs
    println("Avg Time_tot (s): $((cb_avg.times_tot[end])/ 1e9)")
    return cb_avg
end

callback_earcg1 = run_averaged("EARCG-St", cb -> energy_adaptive_riemannian_conjugate_gradient(basis;
            ψ = ψ1, ρ = ρ1, μ = 0.01, tol,
            callback = cb,
            iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.1)));

callback_earcg2 = run_averaged("EARCG-Gr", cb -> energy_adaptive_riemannian_conjugate_gradient(basis;
            ψ = ψ1, ρ = ρ1, μ = 0, tol,
            callback = cb,
            iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.1)));

callback_h1rcg = run_averaged("H1RCG", cb -> h1_riemannian_conjugate_gradient(basis;
            ψ = ψ1, ρ = ρ1,
            tol, callback = cb,
            iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.25)));

callback_l2rcg = run_averaged("L2RCG", cb -> l2_riemannian_conjugate_gradient(basis;
            ψ = ψ1, ρ = ρ1,
            tol, callback = cb,
            maxiter = 100,
            iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.25)));

callback_scf = run_averaged("SCF", cb -> self_consistent_field(basis; ψ = ψ1, ρ = ρ1, tol,
            callback = cb));

cbs = [callback_earcg1, callback_earcg2, callback_h1rcg, callback_l2rcg, callback_scf]


plt1 = plot(; yscale = :log, ylabel = "norm res", xlabel = "iter")
plt2 = plot(; yscale = :log, ylabel = "norm res", xlabel = "hams")
plt3 = plot(; yscale = :log, ylabel = "norm res", xlabel = "times")
for (cb, name) = zip(cbs, ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"])
    iters = 0:(length(cb.norm_residuals)-2)
    plot!(plt1, iters, cb.norm_residuals[1:end-1], label = name)
    plot!(plt2, cb.hams[1:end-1]./p, cb.norm_residuals[1:end-1], label = name)
    plot!(plt3, cb.times_tot[1:end-1]/1e9, cb.norm_residuals[1:end-1], label = name)
    open("experiments/data/general_$(model_name)_$(name)_iters.dat", "w") do io
        println(io, "x y")
        writedlm(io, [iters cb.norm_residuals[1:end-1]], ' ')
    end
    open("experiments/data/general_$(model_name)_$(name)_hams.dat", "w") do io
        println(io, "x y")
        writedlm(io, [cb.hams[1:end-1]./p cb.norm_residuals[1:end-1]], ' ')
    end
    open("experiments/data/general_$(model_name)_$(name)_times.dat", "w") do io
        println(io, "x y")
        writedlm(io, [cb.times_tot[1:end-1]/1e9 cb.norm_residuals[1:end-1]], ' ')
    end
end
display(plt1)
display(plt2)
display(plt3)

