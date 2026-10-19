using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots
using BSON

include("setups/GaAs_setup.jl")

include("./multires.jl")

Ecuts = [10, 16, 25, 40, 63, 101, 160]

model, basis_arr = GaAs_setup(; Ecut = Ecuts, kgrid = [4,4,4]); 
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


# Convergence tolerance
tol = 1.0e-8;


cb = function (info)
    return
    is_cc = haskey(info, :coarse_corrections) ? info.coarse_corrections[end] : false
    println("$(info.basis.Ecut): cc = $is_cc")
end

name = "H1 3L"
lvls = [1,4,7]
println("\n$name")
result = multilevel_riemannian_optimization(basis_arr[lvls]; ψ = ψ1, ρ = ρ1, tol,
    callback = cb,
    callback_mid = cb,
    callback_coarse = cb,
    maxiter = 100,
    coarse_model_tol = 1e-2,
    coarse_cond_tol = 0.45,
        coarse_tol_functor = (tol) -> RCG_DFTK.RelativeResTolerance(1e-2, tol),
    gradients = [H1Gradient(basis) for basis = basis_arr[lvls]]);