using DFTK
using RCG_DFTK
using LineSearches


include("../setups/silicon_setup.jl")
include("../setups/GaAs_setup.jl")
include("../setups/TiO2_setup.jl")

model, basis = silicon_setup(;Ecut = 30, kgrid = [4,4,4], supercell_size = [1,1,1]);
#model, basis = GaAs_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);
#model, basis = TiO2_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);

# Convergence we desire in the residual
tol = 1e-8;

#Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1,  nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

defaultCallback = RcgDefaultCallback();

#H1RCG
# println("H1RCG")
# callback_h1rcg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())

# DFTK.reset_timer!(DFTK.timer)
# scfres_rcg2 = h1_riemannian_conjugate_gradient(basis; 
#         ψ = ψ1, ρ = ρ1,
#         tol, maxiter = 100, 
#         callback = callback_h1rcg);
# println(DFTK.timer)

#EARCG
println("EARCG")
callback_earcg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        callback = callback_earcg);
println(DFTK.timer)

#EARG
println("EARG")
callback_earg = ResidualEvalCallback(;defaultCallback, method = EvalRCG())

DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = energy_adaptive_riemannian_gradient(basis; 
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100, 
        callback = callback_earg);
println(DFTK.timer)

#SCF
println("SCF")
callback_scf = ResidualEvalCallback(;defaultCallback, method = EvalSCF())
is_converged = ResidualEvalConverged(tol, callback_scf)

DFTK.reset_timer!(DFTK.timer)
scfres_scf = self_consistent_field(basis; tol, 
        callback = callback_scf, 
        is_converged = is_converged,
        ψ = ψ1, ρ = ρ1,
        maxiter = 100);
println(DFTK.timer)

# In order to make this plot independent from precompule time,
# one should make it run twice
RCG_DFTK.plot_callbacks(
        [callback_scf, callback_h1rcg, callback_earcg, callback_earg],
        ["SCF", "H1RCG", "EARCG", "EARG"], 
        ψ1, basis)

