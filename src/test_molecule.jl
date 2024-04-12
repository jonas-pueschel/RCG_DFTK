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


model, basis = silicon_setup(; Ecut = 50);

# Convergence we desire in the residual
tol = 1e-8;

filled_occ = DFTK.filled_occupation(model);
n_spin = model.n_spin_components;
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);


ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
#SCF Iterations here to 2 or 3
scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = 1);
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;


defaultCallback = RcgDefaultCallback()

#EARCG
callback = ResidualEvalCallback(;defaultCallback)
is_converged = ResidualEvalConverged(tol, callback)

shift = RelativeΛShift(1.1) #shift Hk with -1.1 * Λk to try to make Ham pd
gradient = EAGradient(3, shift; krylov_solver = Krylov.minres) #Ham may not be pd --> minres
stepsize = ExactHessianStep(2.5)
backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 10, 10)

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 100, 
        callback = callback, is_converged = is_converged,
        stepsize = stepsize, gradient = gradient, backtracking = backtracking);
println(DFTK.timer)


#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback)
is_converged = ResidualEvalConverged(tol, callback)

gradient = H1Gradient(basis)
stepsize = ApproxHessianStep(2.5)
parameter = ParamFR_PRP()
backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 10, 3)

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 100, 
        callback = callback, is_converged = is_converged,
        cg_param = parameter, stepsize = stepsize, gradient = gradient, 
        backtracking = backtracking);
println(DFTK.timer)


#SCF
callback = ResidualEvalCallback(;defaultCallback)
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
        scfres_scf = self_consistent_field(basis; tol, callback, is_converged, ψ = ψ1, maxiter = 100);print("");
println(DFTK.timer)

#PDCM
callback = ResidualEvalCallback(;defaultCallback, method = EvalPDCM())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
scfres_dcm = DFTK.direct_minimization(basis; ψ = ψ1, maxiter = 50,
        optim_method = Optim.LBFGS, 
        linesearch = LineSearches.HagerZhang(),
        callback = callback, is_converged = is_converged);
println(DFTK.timer)
#methods:  LBFGS, GradientDescent, ConjugateGradient