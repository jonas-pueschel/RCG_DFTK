using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr
#Ecuts = [10, 16, 25, 40, 63, 101, 160, 250, 400, 630, 1010]
Ecuts = [10, 25, 63, 160, 400, 1010]
model, basis_arr = silicon_setup(; Ecut = Ecuts, kgrid = [4,4,4]); 
model_name = "silicon"
basis_c = basis_arr[1]

# Initial value
scfres_start = self_consistent_field(basis_c; tol = 1e-8);
ψ_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ_c = scfres_start.ρ

# Convergence tolerance
tol = 1.0e-7;
tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
tols_scf = 0.1 .* [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# Reference solution on the finest grid, needed during the runs to track the distance.
# The silent RFMG run provides the initial guess and also compiles the methods, such that
# the timings below do not contain compilation time.
println("\nReference")
scfres_ref = riemannian_full_multigrid(basis_arr; ψ = ψ_c, ρ = ρ_c, tol = 1e-13, #callback = info -> nothing,
 coarse_model_tol_outer = 1e-2);
ψ_ref = scfres_ref.ψ

# RFMG
println("\nRFMG")
cb_rfmg = RfmgDefaultCallback(;basis_arr, ψ_ref)
res_rfmg = riemannian_full_multigrid(basis_arr; ψ = ψ_c, ρ = ρ_c, tol, callback = cb_rfmg,
 coarse_model_tol_outer = 1e-1,
 gradients  = [H1Gradient(basis) for basis = basis_arr],
 iteration_strat_outer = RestartedGreedySecantStrategy(ConstantStep(1.0), 0.1));


# nested EARCG
println("\nnested EARCG")
cb_nearcg = RfmgDefaultCallback(;basis_arr, ψ_ref)
res_nearcg = nested_earcg(basis_arr, tols; ψ = ψ_c, ρ = ρ_c, callback = cb_nearcg);

# nested SCF
println("\nnested SCF")
cb_nscf = RfmgDefaultCallback(;basis_arr, ψ_ref)
res_nscf = nested_scf(basis_arr, tols_scf; ψ = ψ_c, ρ = ρ_c, callback = cb_nscf);


function get_occ(basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]
    return occupation
end

basis_f = basis_arr[end]
cb_ml = RfmgDefaultCallback(;basis_arr, ψ_ref)
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ_c)
ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))
result = multilevel_riemannian_optimization(basis_arr; ψ = ψ1, ρ = ρ1, tol,
    callback = cb_ml,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45)


# dist vs. time
plt = plot(; yscale = :log10, xscale = :log10, ylabel = "dist", xlabel = "time (s)")
mkpath("experiments/data")
for (cb, name) = zip([cb_rfmg, cb_nearcg, cb_nscf, cb_ml], ["RFMG", "nEARCG", "nSCF", "RML"])
    plot!(plt, cb.times, cb.dists, label = name, marker = :circle, markersize = 2)

    open("experiments/data/rfmg_$(model_name)_$(name)_dist_time.dat", "w") do io
        println(io, "x y Ecut")
        writedlm(io, [cb.times cb.dists cb.Ecuts], ' ')
    end
end
display(plt)

function get_occ(basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]
    return occupation
end

basis_f = basis_arr[5]
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ_c)
ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))

cb_test = RfmgDefaultCallback(;basis_arr, ψ_ref)
energy_adaptive_riemannian_conjugate_gradient(basis_f; ψ = ψ1, ρ = ρ1, 
    tol = 1e-10,
    callback = cb_test)