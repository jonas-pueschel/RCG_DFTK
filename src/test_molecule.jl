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

#model, basis = silicon_setup(;Ecut = 40, kgrid = [3,3,3], supercell_size = [1,1,1]);
model, basis = silicon_setup(;Ecut = 50, kgrid = [6,6,6], supercell_size = [2,2,2]);
#model, basis = GaAs_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);
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
callback_earcg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback_earcg)

shift = CorrectedRelativeΛShift(; μ = 0.01)

gradient = EAGradient(basis, shift; 
        tol = 2.5e-2,
        itmax = 10,
        h_solver = GlobalOptimalHSolver(),
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 

backtracking = AdaptiveBacktracking(
        ModifiedSecantRule(0.05, 0.1, 1e-12, 0.5),
        ExactHessianStep(basis), 10);

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        cg_param = ParamFR_PRP(),
        callback = callback_earcg, is_converged = is_converged,
        gradient = gradient, iteration_strat = backtracking);
println(DFTK.timer)

# println("H1RCG")
# #RCG: H1 Gradient
# callback_h1rcg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
# is_converged = ResidualEvalConverged(tol, callback_h1rcg)

# gradient = H1Gradient(basis)
# backtracking = AdaptiveBacktracking(
#         ModifiedSecantRule(0.05, 0.1, 1e-12, 0.5),
#         ExactHessianStep(basis), 10);

# DFTK.reset_timer!(DFTK.timer)
# scfres_rcg2 = riemannian_conjugate_gradient(basis; 
#         ψ = ψ1, ρ = ρ1,
#         tol, maxiter = 100, 
#         transport_η = DifferentiatedRetractionTransport(),
#         transport_grad = DifferentiatedRetractionTransport(),
#         callback = callback_h1rcg, is_converged = is_converged,
#         gradient = gradient, 
#         iteration_strat = backtracking);
# println(DFTK.timer)

println("L2RCG")
#RCG: L2 Gradient
callback_l2rcg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback_l2rcg)

gradient = L2Gradient()
backtracking = StandardBacktracking(
        ModifiedSecantRule(0.05, 0.1, 1e-12, 0.5),
        ApproxHessianStep(), 10);
# backtracking = AdaptiveBacktracking(
#         ModifiedSecantRule(0.01, 0.9, 1e-12, 0.5),
#         ExactHessianStep(basis), 10);

DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 200, 
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        callback = callback_l2rcg, is_converged = is_converged,
        gradient = gradient, 
        iteration_strat = backtracking);
println(DFTK.timer)

#SCF
println("SCF")
callback_scf = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback_scf)

DFTK.reset_timer!(DFTK.timer)
scfres_scf = self_consistent_field(basis; tol, 
        callback = callback_scf, 
        is_converged = is_converged,
        #ψ = ψ1, ρ = ρ1,
        maxiter = 100);print("");
println(DFTK.timer)

#PDCM
callback_pdcm = ResidualEvalCallback(;defaultCallback, method = EvalPDCM())
is_converged = ResidualEvalConverged(tol, callback_pdcm)

DFTK.reset_timer!(DFTK.timer)
scfres_dcm = DFTK.direct_minimization(basis,
        ψ = ψ1;
        tol, maxiter = 100,
        linesearch = LineSearches.HagerZhang(),
        callback = callback_pdcm, is_converged = is_converged);
println(DFTK.timer)
#methods:  LBFGS, GradientDescent, ConjugateGradient

plot_callbacks(
        [callback_scf, callback_l2rcg, callback_pdcm, callback_earcg],
        ["SCF", "L2RCG", "PDCM", "EARCG"], 
        ψ1, basis)