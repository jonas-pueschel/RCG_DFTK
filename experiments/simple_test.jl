using DFTK
using RCG_DFTK
using PseudoPotentialData


# this script forces precompliation for all methods, ensuring comparability in runtime
include("precompile_methods.jl");
precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr
model, basis = silicon_setup(; Ecut = 30, kgrid = [4,4,4]);

# Convergence tolerance
tol = 1.0e-8;

# Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

#default callback
defaultCallback = RcgDefaultCallback();

# we note that there is a discrepancy between the Time_tot we print and the sums of Δtime 
# this is caused by the fact that time_tot is read from DFTK.timer and Δtime uses time.ns()
# Thus, Δtime also accounts for time not "spent in" the DFTK.timer, like calculation of 
# the residual for SCF and other overhead caused by the benchmarking tools. 


# EARCG
println("\nEARCG-Gr")
callback_earcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())
DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, tol, 
                                                            callback = callback_earcg);
println("Time_tot (s): $((callback_earcg.times_tot[end])/ 1e9)")

# H1RCG
println("\nH1RCG")
callback_h1rcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())
DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
                                               callback = callback_h1rcg);
println("Time_tot (s): $((callback_h1rcg.times_tot[end])/ 1e9)")

# SCF
println("\nSCF")
callback_scf = ResidualEvalCallback(; defaultCallback, method = EvalSCF())
DFTK.reset_timer!(DFTK.timer)
scfres_scf = self_consistent_field(basis; ψ = ψ1, ρ = ρ1, tol,
                                   callback = callback_scf);
println("Time_tot (s): $((callback_scf.times_tot[end])/ 1e9)")