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


defaultCallback = RcgDefaultCallback(); #no callback because >1000 iterations

#EARCG
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

shift = CorrectedRelativeΛShift(; μ = 1.0)
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
        ψ = ψ1, 
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
        ψ = ψ1,
        tol, maxiter = 100, 
        transport_η = NoTransport(),
        transport_grad = NoTransport(),
        callback = callback, is_converged = is_converged,
        cg_param = cg_param,
        gradient = gradient, backtracking = backtracking);
println(DFTK.timer)


#PDCM
callback = ResidualEvalCallback(;defaultCallback);
is_converged = ResidualEvalConverged(tol, callback);

DFTK.reset_timer!(DFTK.timer);
scfres_dcm = DFTK.direct_minimization(basis; ψ = ψ1, maxiter = 10000,
        optim_method = Optim.LBFGS, 
        linesearch = LineSearches.HagerZhang(),
        callback = callback, is_converged = is_converged);
println(DFTK.timer);
#methods:  LBFGS, GradientDescent, ConjugateGradient