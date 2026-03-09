using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")


Ecuts = [10, 16, 25, 40, 63, 101, 160]
model, basis_arr = silicon_setup(; Ecut = Ecuts, kgrid = [4,4,4]);
basis_c = basis_arr[1]
basis_f = basis_arr[end]

# Initial value
# scfres_start = self_consistent_field(basis_c; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(basis_c.model));
# ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
# ρ1_c = scfres_start.ρ
# ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
# ρ1 = DFTK.interpolate_density(ρ1_c, basis_c, basis_f)
scfres_start = self_consistent_field(basis_f; tol = 1e-1, nbandsalg = DFTK.FixedBands(basis_f.model));
ψ1_f = DFTK.select_occupied_orbitals(basis_f, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_f = scfres_start.ρ
ψ1 = ψ1_f
ρ1 = ρ1_f

# scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
# e_ref = scfres_ref.energies.total 

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
init_e = es0.total

# Convergence tolerance
tol = 1.0e-6;
default_callback = RcgDefaultCallback()

println("\nH1 7L")
cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg0 = multilevel_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    callback = cb0,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr])
println(cb0.times_tot[end] / 1e9)

println("\nH1 7L")
cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg0 = multilevel_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    callback = cb0,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr])
println(cb0.times_tot[end] / 1e9)

println("\nH1 2L")
cb1 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg1 = two_level_riemannian_optimization(basis_c, basis_f; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    iteration_strat_fine = StandardBacktracking(
        ArmijoRule(0.1, 0.5),
        ConstantStep(1.0), 10
    ),
    callback = cb1,
    gradient = H1Gradient(basis_f),
    coarse_solvers = [rcg_coarse_solver(basis_c, H1Gradient(basis_c),10)],
    );
println(cb1.times_tot[end] / 1e9)

println("\nH1 A2L")
cb2 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg2 = two_level_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    callback = cb2,
    gradient = H1Gradient(basis_f),
    coarse_solvers = [rcg_coarse_solver(basis_c, H1Gradient(basis_c),10) for basis_c = basis_arr[1:end-1]],
    );
println(cb2.times_tot[end] / 1e9)


println("\nEA 4L")
cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg3 = multilevel_riemannian_optimization(basis_arr[1:end]; ψ = ψ1_f, ρ = ρ1_f, 
    tol,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    callback = cb3,
    gradients = [EAGradient(basis) for basis = basis_arr])
println(cb3.times_tot[end] / 1e9)

println("\nEA 2L")
cb4 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg4 = two_level_riemannian_optimization(basis_arr[[1,end]]; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    callback = cb4,
    gradient = EAGradient(basis_f),
    coarse_solvers = [rcg_coarse_solver(basis_arr[1], EAGradient(basis_arr[1]),10)]
);
println(cb4.times_tot[end] / 1e9)

println("\nEA A2L")
cb5 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_mg5 = two_level_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    callback = cb5,
    gradient = EAGradient(basis_f),
    coarse_solvers = [rcg_coarse_solver(basis_c, EAGradient(basis_c),10) for basis_c = basis_arr[1:end-1]],
    );
println(cb5.times_tot[end] / 1e9)

println("\nH1RCG")
cb6 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_h1rcg = RCG_DFTK.h1_riemannian_conjugate_gradient(basis_f;
    callback = cb6, ψ = ψ1, ρ = ρ1, tol);
println(cb6.times_tot[end] / 1e9)

println("\nEARCG")
cb7 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_earcg = energy_adaptive_riemannian_conjugate_gradient(basis_f;
    callback = cb7, ψ = ψ1, ρ = ρ1, tol);
println(cb7.times_tot[end] / 1e9)

println("\nSCF")
cb8 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_scf = self_consistent_field(basis_f;
    callback = cb8, ψ = ψ1, ρ = ρ1, tol = 0.5 * tol);

# cb9 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# scfres_ref = energy_adaptive_riemannian_conjugate_gradient(basis_f;
#     callback = cb9, ψ = scfres_earcg.ψ, ρ = scfres_earcg.ρ, tol = 1e-14);



cbs = [cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8]
emin = min([min(cb.Es...)  for cb = cbs]...)
results = [scfres_mg0, scfres_mg1, scfres_mg2, scfres_mg3, scfres_mg4, scfres_mg5, nothing, nothing, nothing]
ccs = []
for res = results
    if isnothing(res)
        push!(ccs, nothing)
    else
        push!(ccs, res.coarse_corrections)
    end
end

#e_ref += 1e-14 * min([min(cb.Es...)  for cb = cbs]...)
#postprocessing
err = 1e-15
refval = max([cbb.Es[1] for cbb = cbs]...)
for cb = cbs
    if emin < 0
        cb.Es .+= (err - emin)
        continue
    end
    if abs(cb.Es[end]) < 1e-12
        continue
    end
    if cb.Es[1] != refval
        cb.Es .+= (refval - cb.Es[1])
    else
        cb.Es .+= (err - emin)
    end
end

cb0.Es
using BSON
BSON.@save "results_GaAs.bson" cbs ccs


