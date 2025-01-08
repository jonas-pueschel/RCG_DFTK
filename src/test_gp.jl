using Pkg
Pkg.activate("./src")
using Optim
using LineSearches

using DFTK
using LinearAlgebra
using Krylov
#using Plots

include("./rcg.jl")
include("./setups/all_setups.jl")
include("./rcg_benchmarking.jl")


model, basis = gp2D_setup(Ecut = 400);

# Convergence we desire in the residual
tol = 1e-8;

filled_occ = DFTK.filled_occupation(model);
n_spin = model.n_spin_components;
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
ψ1 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
ρ1 = guess_density(basis)

defaultCallback = RcgDefaultCallback(); #no callback because >1000 iterations

#EARCG
println("EARCG")
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

shift = ConstantShift(0.0)

gradient = EAGradient(basis, shift; 
        rtol = 0.05,
        itmax = 25,
        h_solver = NaiveHSolver(),
        krylov_solver = Krylov.cg,
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 

backtracking = AdaptiveBacktracking(
        WolfeHZRule(0.05, 0.1, 0.5),
        ExactHessianStep(), 10);

cg_param = ParamFR_PRP()

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 1000, 
        cg_param,
        callback = callback, is_converged = is_converged,
        gradient = gradient, backtracking = backtracking);
println(DFTK.timer)

println("H1RCG")
#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

gradient = H1Gradient(basis)
stepsize = ApproxHessianStep()
#backtracking = StandardBacktracking(WolfeHZRule(0.05, 0.4, 0.5), stepsize, 10)
backtracking = AdaptiveBacktracking(
        WolfeHZRule(0.05, 0.25, 0.5),
        ExactHessianStep(), 10);

DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 1000, 
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        callback = callback, is_converged = is_converged,
        gradient = gradient, 
        #cg_param = ParamZero(),
        backtracking = backtracking);
println(DFTK.timer)

#PDCM
callback = ResidualEvalCallback(;defaultCallback, method = EvalPDCM())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
scfres_dcm = DFTK.direct_minimization(basis,
        ψ = ψ1;
        tol, maxiter = 1000,
        linesearch = LineSearches.HagerZhang(),
        callback = callback, is_converged = is_converged);
println(DFTK.timer)
#methods:  LBFGS, GradientDescent, ConjugateGradient

