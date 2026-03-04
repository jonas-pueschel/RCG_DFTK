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

model, basis_arr = silicon_setup(; Ecut = [15, 30, 45, 60], kgrid = [4,4,4]);
basis_cc = basis_arr[1]
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

gradient_functor = (basis) -> H1Gradient(basis)
default_callback = RcgDefaultCallback()
println("\nH1 4g")
cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = multilevel_riemannian_optimization(basis_arr[3:end]; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb0,
    gradient_functor
    );
println(cb0.times_tot[end] / 1e9)
println("\nH1 2g")
cb1 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = multilevel_riemannian_optimization(basis_arr[[1,end]]; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb1,
    gradient_functor 
    );
println(cb1.times_tot[end] / 1e9)

println("\nH1 A2g")
cb2 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = two_level_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb2,
    gradient = gradient_functor(basis_arr[end]),
    coarse_solver = (basis_c) -> h1_coarse_solver(basis_c, 10),
    );
println(cb2.times_tot[end] / 1e9)

gradient_functor = (basis) -> EAGradient(basis)

println("\nEA 4g")
cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = multilevel_riemannian_optimization(basis_arr[1:end]; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb3,
    gradient_functor
    );
println(cb3.times_tot[end] / 1e9)

println("\nEA 2g")
cb4 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = two_level_riemannian_optimization(basis_arr[[1,end]]; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb4,
    gradient = gradient_functor(basis_arr[end]),
    coarse_solver = (basis_c) -> h1_coarse_solver(basis_c, 10),
);
println(cb4.times_tot[end] / 1e9)

println("\nEA A2g")
cb5 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = two_level_riemannian_optimization(basis_arr; ψ = ψ1_f, ρ = ρ1_f, tol,
    coarse_model_tol = 1e-2,
    callback = cb5,
    gradient = gradient_functor(basis_arr[end]),
    coarse_solver = (basis_c) -> h1_coarse_solver(basis_c, 10),
    );
println(cb5.times_tot[end] / 1e9)



# # cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# # scfres_rcg3 = RCG_DFTK.h1_riemannian_gradient(basis_f;
# #     callback = cb3, ψ = ψ1, ρ = ρ1, tol);
# # println(cb3.times_tot[end] / 1e9)

# cb4 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# scfres_rcg4 = RCG_DFTK.energy_adaptive_riemannian_conjugate_gradient(basis_f;
#     callback = cb4, ψ = ψ1, ρ = ρ1, tol);
# println(cb4.times_tot[end] / 1e9)

# # cb5 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# # scfres_scf = self_consistent_field(basis_f;
# #     callback = cb5, ψ = ψ1, ρ = ρ1, tol = 1e-7);
# # println(cb5.times_tot[end] / 1e9)

# e_ref = min(cb0.Es..., cb4.Es..., ) - 1e-14
# ml_times0 = [cb0.times_tot[pk] for pk = 2:(length(cb0.times_tot)-1) if scfres_rcg0.coarse_corrections[pk-1]]
# ml_resls0 = [cb0.norm_residuals[pk] for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
# ml_times = [cb1.times_tot[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
# ml_resls = [cb1.norm_residuals[pk] for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
# plt1 = plot(; yscale = :log, ylabel = "norm res", xlabel = "CPU time in s")
# plot!(cb1.times_tot / 1e9, cb1.norm_residuals, label = "H1ML")
# scatter!(ml_times/ 1e9, ml_resls, label = "coarse corr H1")
# plot!(cb0.times_tot / 1e9, cb0.norm_residuals, label = "EAML")
# scatter!(ml_times0/ 1e9, ml_resls0, label = "coarse corr EA")
# plot!(cb2.times_tot / 1e9, cb2.norm_residuals, label = "H1RCG")
# plot!(cb4.times_tot / 1e9, cb4.norm_residuals, label = "EARCG")
# #plot!(cb3.times_tot / 1e9, cb3.norm_residuals, label = "H1RG")
# #plot!(cb5.times_tot / 1e9, cb5.norm_residuals, label = "SCF")
# display(plt1)

# ml_Es0 = [cb0.Es[pk] - e_ref for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
# ml_Es = [cb1.Es[pk] - e_ref for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
# plt = plot(; yscale = :log, ylabel = "ΔE", xlabel = "CPU time in s")
# plot!(cb1.times_tot / 1e9, cb1.Es.-e_ref, label = "H1ML")
# scatter!(ml_times/ 1e9, ml_Es, label = "coarse corr H1")
# plot!(cb0.times_tot / 1e9, cb0.Es.-e_ref, label = "EAML")
# scatter!(ml_times0/ 1e9, ml_Es0, label = "coarse corr EA")
# plot!(cb2.times_tot / 1e9, cb2.Es.-e_ref, label = "H1RCG")
# plot!(cb4.times_tot / 1e9, cb4.Es.-e_ref, label = "EARCG")
# #plot!(cb3.times_tot / 1e9, cb3.Es.-e_ref, label = "H1RG")
# #plot!(cb5.times_tot / 1e9, cb5.Es.-e_ref, label = "SCF")
# display(plt)

# iter = [i for i = 0:(length(cb1.norm_residuals)-1)]
# iter0 = [i for i = 0:(length(cb0.norm_residuals)-1)]
# plt2 = plot(; yscale = :log, ylabel = "ΔE", xlabel = "iter")
# plot!(iter0, cb0.Es.-e_ref, label = "EAML")
# plot!(iter, cb1.Es.-e_ref, label = "H1ML")
# ml_iters = [iter[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
# ml_iters0 = [iter0[pk] for pk = 2:(length(cb0.times_tot)-1) if scfres_rcg0.coarse_corrections[pk-1]]
# scatter!(ml_iters0, ml_Es0, label = "coarse corr EA")
# scatter!(ml_iters, ml_Es, label = "coarse corr H1")

# display(plt2)