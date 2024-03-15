using DFTK
using LinearAlgebra
#using Plots

include("./rcg.jl")
include("setups/all_setups.jl")

model, basis = silicon_setup(Ecut = 15);

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


options1 = default_options(basis, ψ1)
options1.gradient_options.gradient = "H1"
options1.retraction = "qr"
options1.β_cg = "FR-PRP"
options1.step_size_options.bt_iter = 0
options1.step_size_options.step_size = "eH"

@time begin   
    scfres_rcg1 = rcg(basis, ψ0; tol = tol, maxiter = 100, options = options1);print("");
end

#heatmap(scfres_rcg1.ρ[:, :, 1, 1], c=:blues)

options2 = default_options(basis, ψ1)
options2.gradient_options.gradient = "ea"
options2.gradient_options.inner_iter = 3
options2.retraction = "polar"
options2.β_cg = "FR-PRP"
options2.step_size_options.bt_iter = 10;
#options2.μ = 2.0;
options2.step_size_options.step_size = "eH"
options2.gradient_options.shift = 0.0

#Shifts: TiO2: 0.0, H2: 0.5, graphene: 0.0, iron: don't, GaAs: 
#Stepsize:     eH       eH             aH         don't

@time begin   
    scfres_rcg2 = rcg(basis, ψ1; tol = tol, maxiter = 100, options = options2);print("");
end

@time begin
    scfres_scf = self_consistent_field(basis; tol, ψ = ψ1);print("");
end

#heatmap(scfres_rcg1.ρ[:, :, 1, 1], c=:blues)