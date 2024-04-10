using DFTK
using LinearAlgebra
using StaticArrays
using Plots

include("./rcg.jl")
include("setups/all_setups.jl")

model, basis = gp2D_setup(Ecut = 400);

# Convergence we desire in the density
tol = 1e-4

filled_occ = DFTK.filled_occupation(model)
n_spin = model.n_spin_components
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
ψ1 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]

#scfres_start = self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
#ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
#scfres_start = direct_minimization(basis; tol = tol^(1.2), ψ = ψ1, maxiter = 20);print("");
#ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;



gradient = H1Gradient(basis)
retraction = RetractionQR();

scfres_rcg1 = rcg(basis, ψ1; tol = tol, maxiter = 1000, retraction = retraction);print("");




scfres_scf = direct_minimization(basis; tol = tol^(1.2), ψ = ψ1, maxiter = 10);print("");

