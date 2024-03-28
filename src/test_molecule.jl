using DFTK
using LinearAlgebra
#using Plots

include("./rcg.jl")
include("setups/all_setups.jl")

model, basis = GaAs_setup(Ecut = 15);

# Convergence we desire in the density
tol = 1e-6

filled_occ = DFTK.filled_occupation(model)
n_spin = model.n_spin_components
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)


ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
#Sometimes SCF fcks up number of bands (or similar?) so we may need to increase the number of
#SCF Iterations here to 2 or 3
scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = 3);
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;

# rcg quite heavily benefits from one scf step to get a good initial ψ
# number of iterations gets more than halved in almost all cases!ψ


#scfres_rcg1 = rcg(basis, ψ0; tol = tol, maxiter = 100);print("");



scfres_scf = self_consistent_field(basis; tol, ψ = ψ1);print("");

