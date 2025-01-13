using Pkg
Pkg.activate("./src")
Pkg.instantiate()
#using Optim
using LineSearches
using TimerOutputs

using DFTK
using LinearAlgebra
using Krylov
using Plots
using BSON

include("./rcg.jl")
include("./setups/all_setups.jl")
include("./rcg_benchmarking.jl")

#model, basis = silicon_setup(;Ecut = 50, kgrid = [6,6,6], supercell_size = [1,1,1]);
model, basis = GaAs_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);
#model, basis = TiO2_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);

# Convergence we desire in the residual
tol = 1e-8;


#Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1,  nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

defaultCallback = RcgDefaultCallback();

#EARCG
println("EARCG")
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

shift = CorrectedRelativeΛShift(; μ = 0.01)

gradient = EAGradient(basis, shift; 
        rtol = 2.5e-2,
        itmax = 10,
        h_solver = GlobalOptimalHSolver(),
        krylov_solver = Krylov.minres,
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 

backtracking = AdaptiveBacktracking(
        WolfeHZRule(0.05, 0.1, 0.5),
        ExactHessianStep(), 10);

# backtracking = StandardBacktracking(
#         WolfeHZRule(0.05, 0.1, 0.5),
#         ApproxHessianStep(), 10);

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        cg_param = ParamFR_PRP(),
        callback = callback, is_converged = is_converged,
        gradient = gradient, backtracking = backtracking);
println(DFTK.timer)

println("H1RCG")
#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

gradient = H1Gradient(basis)
# backtracking = AdaptiveBacktracking(
#         WolfeHZRule(0.05, 0.1, 0.5),
#         ExactHessianStep(), 10);

backtracking = StandardBacktracking(
        WolfeHZRule(0.05, 0.1, 0.5),
        ApproxHessianStep(), 10);


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
println("SCF")
callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
scfres_scf = self_consistent_field(basis; tol, 
        callback, 
        is_converged,
        #ψ = ψ1, ρ = ρ1,
        maxiter = 100);print("");
println(DFTK.timer)

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

