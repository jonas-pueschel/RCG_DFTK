using Pkg
Pkg.activate("./src")
Pkg.instantiate()
#using Optim
using LineSearches
using TimerOutputs

using DFTK
using LinearAlgebra
using Krylov
#using Plots

include("./rcg.jl")
include("./setups/all_setups.jl")
include("./rcg_benchmarking.jl")


model, basis = GaAs_setup(; Ecut = 50);


# Convergence we desire in the residual
tol = 1e-8;


#Initial value
scfres_start = self_consistent_field(basis; tol = 1e-1,  nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

defaultCallback = RcgDefaultCallback();

#EARCG
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

shift = CorrectedRelativeΛShift(; μ = 0.01)
gradient = EAGradient(basis, shift; 
        rtol = 2.5e-2,
        itmax = 10,
        h_solver = LocalOptimalHSolver(basis, GalerkinLOIS;
                is_converged_lois = is_converged_res),
        krylov_solver = Krylov.cg,
        Pks = [PreconditionerLOIS(basis, kpt, 1) for kpt in basis.kpoints]
        ) 

backtracking = AdaptiveBacktracking(
        WolfeHZRule(0.0, 0.1, 0.5),
        ExactHessianStep(), 10);

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        cg_param = ParamFR_PRP(),
        callback = callback, is_converged = is_converged,
        gradient = gradient, backtracking = backtracking);
println(DFTK.timer)


#EARG
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

shift = CorrectedRelativeΛShift()       
gradient = EAGradient(basis, shift; 
        rtol = 2.5e-2,
        itmax = 10,
        h_solver = LocalOptimalHSolver(basis, GalerkinLOIS;
                is_converged_lois = is_converged_res),
        krylov_solver = Krylov.minres,
        Pks = [PreconditionerLOIS(basis, kpt, 1) for kpt in basis.kpoints]
        ) 
stepsize = BarzilaiBorweinStep(0.1, 2.5, 1.0);
backtracking = StandardBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), stepsize, 10)
cg_param = ParamZero();

DFTK.reset_timer!(DFTK.timer)
scfres_rg1 = riemannian_conjugate_gradient(basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        transport_η = NoTransport(),
        transport_grad = NoTransport(),
        callback = callback, is_converged = is_converged,
        cg_param = cg_param,
        gradient = gradient, backtracking = backtracking);
println(DFTK.timer)

#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

gradient = H1Gradient(basis)
stepsize = ApproxHessianStep()
backtracking = StandardBacktracking(WolfeHZRule(0.05, 0.4, 0.5), stepsize, 10)


DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        callback = callback, is_converged = is_converged,
        gradient = gradient, 
        #cg_param = ParamZero(),
        backtracking = backtracking);
println(DFTK.timer)

#SCF
callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
scfres_scf = self_consistent_field(basis; tol, 
        callback, 
        is_converged,
        ψ = ψ1, ρ = ρ1,
        maxiter = 100);print("");
println(DFTK.timer)

#naive SCF
# callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
# is_converged = ResidualEvalConverged(tol, callback)

# DFTK.reset_timer!(DFTK.timer)
# self_consistent_field(basis;  
#         ψ = ψ1, ρ = ρ1, 
#         tol, maxiter = 100,
#         callback, is_converged = is_converged = ResidualEvalConverged(tol, callback),
#         mixing = DFTK.SimpleMixing(), nbandsalg = DFTK.FixedBands(model), solver = DFTK.scf_damping_solver());
# println(DFTK.timer)

#PDCM
callback = ResidualEvalCallback(;defaultCallback, method = EvalPDCM())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
scfres_dcm = DFTK.direct_minimization(basis,
        ψ = ψ1;
        tol, maxiter = 100,
        linesearch = LineSearches.HagerZhang(),
        callback = callback, is_converged = is_converged);
println(DFTK.timer)
#methods:  LBFGS, GradientDescent, ConjugateGradient

