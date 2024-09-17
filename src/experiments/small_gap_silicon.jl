using Pkg
Pkg.activate("./src")
Pkg.instantiate()
using Optim
using LineSearches
using TimerOutputs

using DFTK
using LinearAlgebra
using Krylov
using BSON
#using Plots

include("../rcg.jl")
include("../setups/all_setups.jl")
include("../rcg_benchmarking.jl")

a_min = 10.625
a_max = 11.75
N_steps = 28
#a_step = (a_max - a_min)/N_steps
a_step = 0.0625

n_runs = 20

tol = 1e-8
Ecut = 50

defaultCallback = RcgDefaultCallback();

results = Dict()


methods_rcg = Dict([
    "EARCG" => (basis, ψ1, ρ1, callback) -> riemannian_conjugate_gradient(basis; ψ = ψ1, ρ =  ρ1, maxiter = 50, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
            gradient = EAGradient(basis, CorrectedRelativeΛShift(; μ = 0.01);
            rtol = 2.5e-2,
            itmax = 10,
            h_solver = NestedHSolver(basis, GalerkinInnerSolver;
                    is_converged_InnerSolver = is_converged_res),
            krylov_solver = Krylov.cg,
            Pks = [PreconditionerInnerSolver(basis, kpt, 1) for kpt in basis.kpoints]
        ),
        backtracking = AdaptiveBacktracking(
            WolfeHZRule(0.0, 0.1, 0.5),
            ExactHessianStep(), 5
        )),
    "H1RCG" => (basis, ψ1, ρ1, callback) -> riemannian_conjugate_gradient(basis; ψ = ψ1, ρ =  ρ1, maxiter = 100, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        gradient = H1Gradient(basis), 
        backtracking = StandardBacktracking(
            WolfeHZRule(0.0, 0.9, 0.5), 
            ApproxHessianStep(), 0)
        ),
    "SCF" => (basis, ψ1, ρ1, callback) -> self_consistent_field(basis; ψ = ψ1, ρ =  ρ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback)),
    "DCM" => (basis, ψ1, ρ1, callback) -> DFTK.direct_minimization(basis; ψ = ψ1,
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
        linesearch = LineSearches.HagerZhang(), #performs better than BackTracking
    ),

]);

for a  = a_min:a_step:a_max

    println("##################")
    print("a = "); println(a);
    println("##################")

    model, basis = silicon_setup(; Ecut, a);
    ra = Dict()
    results[a] = ra
    for (method_name, method) in methods_rcg
        ra[method_name] = []
    end

    ra["initvals"] = []

    for i = 1:n_runs
        #Initial value
        scfres_start = self_consistent_field(basis; tol = 1e-1,  nbandsalg = DFTK.FixedBands(model));
        ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
        ρ1 = scfres_start.ρ;

        push!(ra["initvals"], (nothing, nothing))

        for (method_name, method) in methods_rcg

            println(method_name)
            mth = endswith(method_name, "RCG") ? EvalRCG() : startswith(method_name, "SCF") ? EvalSCF() : EvalPDCM();
            callback = ResidualEvalCallback(;defaultCallback, method = mth);
            err = nothing
            DFTK.reset_timer!(DFTK.timer);
            try
                result = method(basis, ψ1, ρ1, callback);println();
            catch e
                err = e 
            end
  
            println(DFTK.timer)

            push!(ra[method_name], (callback, err))
        end
    end

    BSON.@save "result_small_gap_3_$a.bson" results
end


BSON.@save "result_small_gap_3.bson" results

