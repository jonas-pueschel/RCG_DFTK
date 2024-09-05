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

n_runs = 20
tol = 1e-8

result_path = ""


defaultCallback = RcgDefaultCallback();

results = Dict()

mol_setups = Dict([
    #"H2" =>  () -> H2_setup(; Ecut = 30),
    #"graphene" => () -> graphene_setup(;Ecut = 30),
    #"silicon" => () -> silicon_setup(;Ecut = 50),
    #"TiO2" => () -> TiO2_setup(;Ecut = 60),
    "GaAs" => () -> GaAs_setup(;Ecut = 60),
])

methods_rcg = Dict([
    "EARCG" => (basis, ψ1, ρ1, callback) -> riemannian_conjugate_gradient(basis; ψ = ψ1, ρ =  ρ1, maxiter = 50, 
        callback = callback, is_converged = ResidualEvalConverged(tol, callback),
            gradient = EAGradient(basis, CorrectedRelativeΛShift(; μ = 0.01); 
            rtol = 2.5e-2,
            itmax = 10,
            h_solver = LocalOptimalHSolver(basis, GalerkinLOIS;
                    is_converged_lois = is_converged_res),
            krylov_solver = Krylov.cg,
            Pks = [PreconditionerLOIS(basis, kpt, 1) for kpt in basis.kpoints]
        ),
        backtracking = AdaptiveBacktracking(
            WolfeHZRule(0.05, 0.1, 0.5),
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

for (mol, setup) in mol_setups

    println("##################")
    println(mol);
    println("##################")

    model, basis = setup()
    ra = Dict()
    results[mol] = ra
    for (method_name, method) in methods_rcg
        ra[method_name] = []
    end

    for i = 1:n_runs
        #Initial value
        scfres_start = self_consistent_field(basis; tol = 1e-1,  nbandsalg = DFTK.FixedBands(model));
        ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
        ρ1 = scfres_start.ρ;

        for (method_name, method) in methods_rcg

            println(method_name)
            mth = endswith(method_name, "RCG") ? EvalRCG() : startswith(method_name, "SCF") ? EvalSCF() : EvalPDCM();
            callback = ResidualEvalCallback(;defaultCallback, method = mth);
            err = nothing
            DFTK.reset_timer!(DFTK.timer);
            try
                result = method(basis, ψ1, ρ1, callback);println();
            catch e
                println(e)
                #err = e 
            end
  
            println(DFTK.timer)

            push!(ra[method_name], (callback, err))
        end
    end

    BSON.@save "result_comp_$mol.bson" ra
end


BSON.@save "result_comp.bson" results

