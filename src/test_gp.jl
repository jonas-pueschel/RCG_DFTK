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


defaultCallback = nothing; #no callback because >1000 iterations

#EARCG
callback = ResidualEvalCallback(;defaultCallback);
is_converged = ResidualEvalConverged(tol, callback);

shift = RelativeEigsShift(0.0);
gradient = EAGradient(3, shift; krylov_solver = Krylov.cg); #Ham is spd --> cg
stepsize = ExactHessianStep(2.5);
backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 10, 10);

DFTK.reset_timer!(DFTK.timer);
scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 10000, 
        callback = callback, is_converged = is_converged,
        stepsize = stepsize, gradient = gradient, backtracking = backtracking);
println(DFTK.timer);

#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback);
is_converged = ResidualEvalConverged(tol, callback);

gradient = H1Gradient(basis);
stepsize = ApproxHessianStep(2.5);
parameter = ParamFR_PRP();
backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 10, 10);

DFTK.reset_timer!(DFTK.timer);
scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 10000, 
        callback = callback, is_converged = is_converged,
        cg_param = parameter, stepsize = stepsize, gradient = gradient, 
        backtracking = backtracking);
println(DFTK.timer);

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