using Pkg
Pkg.activate("./src")
Pkg.instantiate()
using Optim
using LineSearches

using DFTK
using LinearAlgebra
using Krylov
#using Plots

include("./rcg.jl")
include("./setups/all_setups.jl")
include("./rcg_benchmarking.jl")


#model, basis = graphene_setup(; Ecut = 50);

model, basis, molecule = (silicon_setup(; Ecut = 50, a = 11.90)..., "Silicon");


# Convergence we desire in the residual
tol = 1e-8;

filled_occ = DFTK.filled_occupation(model);
n_spin = model.n_spin_components;
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);

#SCF Iterations here to 2 or 3
scfres_start = self_consistent_field(basis; maxiter = 2,  nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;

defaultCallback = RcgDefaultCallback()

#EARCG
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)


shift = CorrectedRelativeΛShift(1.1) #shift Hk with -1.1 * Λk to try to make Ham pd
gradient = EAGradient(basis, shift; 
        itmax = 50,
        rtol = 2,
        krylov_solver = Krylov.minres,
        Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        #Pks = [MatrixShiftedTPA(basis, kpt) for kpt in basis.kpoints]
        ) #Ham may not be pd --> minres
#stepsize = BarzilaiBorweinStep(0.1, 2.5, 1.0);
#backtracking = GreedyBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), 10, 1)

stepsize = ExactHessianStep(2.5);
backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), 0, 10)

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 100, 
        callback = callback, is_converged = is_converged,
        #cg_param = ParamZero(),
        stepsize = stepsize, gradient = gradient, backtracking = backtracking);
println(DFTK.timer)

#RCG: H1 Gradient
callback = ResidualEvalCallback(;defaultCallback, method = EvalRCG())
is_converged = ResidualEvalConverged(tol, callback)

gradient = H1Gradient(basis)
stepsize = ApproxHessianStep(2.5)
parameter = ParamFR_PRP()
backtracking = GreedyBacktracking(ArmijoRule(0.05, 0.5), 0, 1)

DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = riemannian_conjugate_gradient(basis, ψ1; maxiter = 100, 
        callback = callback, is_converged = is_converged,
        cg_param = parameter, stepsize = stepsize, gradient = gradient, 
        backtracking = backtracking);
println(DFTK.timer)

#SCF
callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
        scfres_scf = self_consistent_field(basis; tol, callback, is_converged,
        ψ = ψ1, 
        maxiter = 100);print("");
println(DFTK.timer)

#naive SCF
callback = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback)

DFTK.reset_timer!(DFTK.timer)
self_consistent_field(basis;  ψ = ψ1, maxiter = 100,
        callback, is_converged = is_converged = ResidualEvalConverged(tol, callback),
        mixing = DFTK.SimpleMixing(), nbandsalg = DFTK.FixedBands(model), solver = DFTK.scf_damping_solver());
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

