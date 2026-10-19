using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots
using BSON

#include("plot_ml_test.jl")
include("multires.jl")

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")


Ecuts = [10, 16, 25, 40, 63, 101, 160]

model, basis_arr = silicon_setup(; Ecut = Ecuts, kgrid = [4,4,4]); 
model_name = "silicon"
basis_c = basis_arr[1]
basis_f = basis_arr[end]

# Initial value
scfres_start = self_consistent_field(basis_c; tol = 1e-8);
ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_c = scfres_start.ρ
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))
#DFTK.interpolate_density(ρ1_c, basis_c, basis_f)
# scfres_start = self_consistent_field(basis_f; tol = 1e-1, nbandsalg = DFTK.FixedBands(basis_f.model));
# ψ1 = DFTK.select_occupied_orbitals(basis_f, scfres_start.ψ, scfres_start.occupation).ψ;
# ρ1 = scfres_start.ρ

# scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
# e_ref = scfres_ref.energies.total 

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
init_e = es0.total


default_callback = RcgDefaultCallback()

cbs = []
ccs_arr = []
names = []
emin = 1000
# Convergence tolerance
tol = 1.0e-8;

result2ccs(result) = haskey(result, :coarse_corrections) ? result.coarse_corrections : nothing;

cbs = []
ccs_arr = []
names = []

ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_arr[3], ψ1_c)
ρ1 = DFTK.compute_density(basis_arr[3], ψ1, get_occ(basis_arr[3]))

name = "H1 2L algebraic"
println("\n$name")
cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
result = multilevel_riemannian_optimization(basis_arr[[1,3]]; ψ = ψ1, ρ = ρ1, tol,
    callback = cb,
    multilevel_map_functor = (point_restriction) -> RCG_DFTK.AlgebraicMap(point_restriction),
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr[[1,3]]])
println(cb.times_tot[end] / 1e9)
push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)


name = "H1 2L geometric"
println("\n$name")
cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
result = multilevel_riemannian_optimization(basis_arr[[1,3]]; ψ = ψ1, ρ = ρ1, tol,
    callback = cb,
    multilevel_map_functor = (point_restriction) -> RCG_DFTK.GeometricMap(point_restriction),
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr[[1,3]]])
println(cb.times_tot[end] / 1e9)
push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

name = "H1 2L projection"
println("\n$name")
cb = TrackResTimeCallback(default_callback, init_norm_res, init_e)
result = multilevel_riemannian_optimization(basis_arr[[1,3]]; ψ = ψ1, ρ = ρ1, tol,
    callback = cb,
    multilevel_map_functor = (point_restriction) -> RCG_DFTK.ProjectionMap(),
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
    gradients = [H1Gradient(basis) for basis = basis_arr[[1,3]]])
println(cb.times_tot[end] / 1e9)
push!(cbs, cb); push!(ccs_arr, result2ccs(result)); push!(names, name)

