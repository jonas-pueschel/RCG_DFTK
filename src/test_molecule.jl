using DFTK
using LinearAlgebra
#using Plots

include("./rcg.jl")
include("setups/all_setups.jl")

model, basis = TiO2_setup(Ecut = 100);

# Convergence we desire in the density
tol = 1e-6

filled_occ = DFTK.filled_occupation(model)
n_spin = model.n_spin_components
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
ψ0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];


#Sometimes SCF fcks up number of bands (or similar?) so we may need to increase the number of
#SCG Iterations here to 2 or 3
scfres_start = self_consistent_field(basis; ψ = ψ0 , maxiter = 1);
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
#TODO sometimes scf fcks up the size

# rcg quite heavily benefits from one scf step to get a good initial ψ
# number of iterations gets more than halved in almost all cases!ψ

options1 = default_options(basis, ψ1)
options1.gradient_options.gradient = "H1"
options1.retraction = "polar"
options1.β_cg = "FR-PRP"
options1.step_size_options.bt_iter = 10;
options1.step_size_options.step_size = "aH"

println(size(ψ1[size(ψ1)[1]]))
println(size(ψ0[size(ψ0)[1]]))

@time begin   
    scfres_rcg1 = rcg(basis, ψ0; tol = tol, maxiter = 200, options = options1);print("");
end


#heatmap(scfres_rcg1.ρ[:, :, 1, 1], c=:blues)

#options2 = default_options(basis, ψ1)
#options2.gradient_options.gradient = "ea"
#options2.gradient_options.inner_iter = 100
#options2.retraction = "polar"
#options2.β_cg = "FR-PRP"
#options2.step_size_options.bt_iter = 10;
#options2.step_size_options.step_size = "eH"

#@time begin   
#    scfres_rcg1 = rcg(basis, ψ1; tol = tol, maxiter = 200, options = options2);print("");
#end

@time begin
    scfres_scf = self_consistent_field(basis; tol, ψ = ψ0);print("");
end

#heatmap(scfres_rcg1.ρ[:, :, 1, 1], c=:blues)