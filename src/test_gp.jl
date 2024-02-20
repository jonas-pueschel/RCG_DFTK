using DFTK
using LinearAlgebra
using StaticArrays
#using Plots

include("./rcg.jl")
include("setups/all_setups.jl")

model, basis = gp2D_setup(Ecut = 200);

# Convergence we desire in the density
tol = 1e-4

filled_occ = DFTK.filled_occupation(model)
n_spin = model.n_spin_components
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]

scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = 2);
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;


options1 = default_options(basis, ψ1)
options1.gradient_options.gradient = "H1"
options1.retraction = "qr"
options1.β_cg = "FR-PRP"
options1.step_size_options.bt_iter = 10;
options1.step_size_options.step_size = "eH"
options1.transport_η = "L2-proj"
options1.transport_res = "L2-proj"

@time begin   
    scfres_rcg1 = rcg(basis, ψ1; tol = tol, maxiter = 200, options = options1);print("");
end
#heatmap(scfres_rcg1.ρ[:, :, 1, 1], c=:blues)

options2 = default_options(basis, ψ1)
options2.gradient_options.gradient = "ea"
options2.gradient_options.inner_iter = 10
options2.retraction = "qr"
options2.β_cg = "HS"
options2.step_size_options.bt_iter = 10;
options2.step_size_options.step_size = "eH"

@time begin   
    scfres_rcg1 = rcg(basis, ψ1; tol = tol, maxiter = 200, options = options2);print("");
end

#Tol needs to be squared (?)

@time begin   
    scfres = direct_minimization(basis, ψ1, tol = tol^1.3 , maxiter = 200);print("");
end
#heatmap(scfres.ρ[:, :, 1, 1], c=:blues)