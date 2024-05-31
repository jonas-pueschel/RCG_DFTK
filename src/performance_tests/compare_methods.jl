using Pkg
Pkg.activate("./src")
Pkg.instantiate()

using BSON
using Optim
using LineSearches

using Plots

using DFTK
using LinearAlgebra
using Plots

include("../rcg.jl")
include("../rcg_benchmarking.jl")
include("../setups/all_setups.jl")
include("./compare_methods_tikz.jl")

#general params
tol = 1e-9;
generate_plots = false;
show_plots = true;

#Ecut 90 sind die korrekten Daten

#model, basis, molecule = (silicon_setup(; Ecut = 90)..., "Silicon");
model, basis, molecule = (GaAs_setup(; Ecut = 90)..., "GaAs");

Nk = length(basis.kpoints);

run_rcg_EA      = true
run_rcg_inEA_3  = true
run_rcg_inEA_8  = true
run_rcg_H1      = true
run_EAR         = true
run_inEAR_3     = true
run_inEAR_8     = true
run_H1R         = false
run_scf_opt     = true
run_scf_naive   = false
run_pdcm_cg     = false
run_pdcm_lbfgs  = true

#calculate initial value
init_iter = 1;

function find_ψ1()
    println("find ψ1...");
    while true
        try
            scfres_start = DFTK.self_consistent_field(basis; maxiter = init_iter, nbandsalg = DFTK.FixedBands(model));
            ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
    
            #try if the chosen value works
            DFTK.self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
            
            println("found.");
            return ψ1
        catch e
            println(e);
            println("failed, trying again...");
            continue;
        end
    end
end
ψ1 = find_ψ1();


defaultCallback = RcgDefaultCallback();
resls = []
times = []
const_hams = []
method_names = []

function  callback_estimate_time(callback, molecule)
    if (molecule == "GaAs")
        β = [ 4.83820040322115e8, 2.883595696697497e9, 2.3567767525293593e9, 3.7665307878829656e9] / 4.83820040322115e8
    elseif (molecule == "Silicon")
        β = [ 1.641290230943121e8, 7.237020217004423e8, 3.5292680939192266e9, 3.7200176048407316e8] / 1.641290230943121e8
    end

    ress = callback.norm_residuals
    hams = callback.calls_DftHamiltonian
    hesss = callback.calls_apply_K
    ess = callback.calls_energy_hamiltonian
    dss = callback.calls_compute_density
    xs = []
    for i = 1:length(ress)

        v = [
            hams[i]/ (Nk * size(ψ1[1])[2]), 
            hesss[i],
            ess[i]/Nk,
            dss[i]/size(ψ1[1])[2]
        ]
        push!(xs, β'v)

    end
    return xs
end


if (run_rcg_EA)
    method_name = "RCG_EA"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), 
        gradient = EAGradient(basis, 100, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 10)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_rcg_inEA_3)
    method_name = "RCG_inEA_3"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), 
        gradient = EAGradient(basis, 3, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_rcg_H1)
    method_name = "RCG_H1"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 150, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ApproxHessianStep(2.5), 
        gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(ArmijoRule(0.05, 0.5), 0)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_EAR)
    method_name = "EAR"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = BarzilaiBorweinStep(1e-4, 1.0, 1.0), 
        gradient = EAGradient(basis, 100, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(NonmonotoneRule(0.95, 1e-4, 0.5), 20)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_inEAR_3)
    method_name = "inEAR_3"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = BarzilaiBorweinStep(1e-4, 1.0, 1.0), 
        gradient = EAGradient(basis, 3, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(NonmonotoneRule(0.95, 1e-4, 0.5), 20)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_H1R)
    method_name = "H1R"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = BarzilaiBorweinStep(1e-4, 1.0, 0.1), 
        gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(NonmonotoneRule(0.95, 1e-4, 0.5), 20)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_pdcm_cg)
    method_name = "PDCM_CG"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalPDCM());
    DFTK.direct_minimization(basis; ψ = ψ1, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        optim_method = Optim.ConjugateGradient, 
        linesearch = LineSearches.HagerZhang()
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals)
    push!(times, callback.times)
    push!(const_hams, callback_estimate_time(callback, molecule))
    push!(method_names, method_name)
end

if (run_pdcm_lbfgs)
    method_name = "PDCM_LBFGS"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalPDCM());
    DFTK.direct_minimization(basis; ψ = ψ1, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        optim_method = Optim.LBFGS, 
        linesearch = LineSearches.HagerZhang()
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals)
    push!(times, callback.times)
    push!(const_hams, callback_estimate_time(callback, molecule))
    push!(method_names, method_name)
end

if (run_scf_opt)
    method_name = "SCF_opt"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF());
    DFTK.self_consistent_field(
        basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_scf_naive)
    method_name = "SCF_naive"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF());
    DFTK.self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback),
        mixing = DFTK.SimpleMixing(), 
        nbandsalg = DFTK.FixedBands(model), 
        solver = DFTK.scf_damping_solver()
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_rcg_inEA_8)
    method_name = "RCG_inEA_8"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = ExactHessianStep(2.5), 
        gradient = EAGradient(basis, 8, CorrectedRelativeΛShift(1.1)), 
        backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 100)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (run_inEAR_8)
    method_name = "inEAR_8"

    println(method_name)
    DFTK.reset_timer!(DFTK.timer);
    callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG());
    riemannian_conjugate_gradient(basis, ψ1; maxiter = 75, 
        callback = callback, 
        is_converged = ResidualEvalConverged(tol, callback),
        stepsize = BarzilaiBorweinStep(1e-4, 1.0, 1.0), 
        gradient = EAGradient(basis, 8, CorrectedRelativeΛShift(1.1)), 
        backtracking = StandardBacktracking(NonmonotoneRule(0.95, 1e-4, 0.5), 20)
    );
    println(DFTK.timer)

    push!(resls, callback.norm_residuals[1:end-1])
    push!(times, callback.times[1:end-1])
    push!(const_hams, callback_estimate_time(callback, molecule)[1:end-1])
    push!(method_names, method_name)
end

if (generate_plots)
    #generate plots
    generate_tikz_plots(molecule, resls, times, cost_hams, method_names, init_res, tol)
end


if (show_plots)
    #plot results
    p = plot(title = "Iterations" * " " * molecule)
    q = plot(title = "Times" * " " * molecule)
    r = plot(title = "Hamiltonian equivalent" * " " * molecule)
    for (res, time, const_ham, method) = zip(resls, times, const_hams, method_names)
        plot!(p , 1:length(res), res, label = method)
        plot!(q, time, res, label = method)
        plot!(r, const_ham, res, label = method)
    end
    plot!(p, yscale=:log10, minorgrid=true);
    display(p);
    plot!(q, yscale=:log10, minorgrid=true);
    display(q);
    plot!(r, yscale=:log10, minorgrid=true);
    display(r);
end







