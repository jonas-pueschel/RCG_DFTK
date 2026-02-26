using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr
model_f, basis_f = silicon_setup(; Ecut = 70, kgrid = [4,4,4]);
model_c, basis_c = silicon_setup(; Ecut = 15, kgrid = [4,4,4]);


# Convergence tolerance
tol = 1.0e-6;

# Initial value
scfres_start = self_consistent_field(basis_c; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(basis_c.model));
ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_c = scfres_start.ρ
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
ρ1 = DFTK.interpolate_density(ρ1_c, basis_c, basis_f)

scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
e_ref = scfres_ref.energies.total 

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
init_e = es0.total

mg_prec = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
    ψ = ψ1, ρ = ρ1, tol = 1e-3, callback = (info) -> nothing)

# multilevel
println("\nMultilevel")
default_callback = RCG_DFTK.RcgDefaultCallback()

cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
    ψ = ψ1, ρ = ρ1, tol, 
    callback = cb0,
    multilevel_map = ProjectionMap(),
    check_convergence_early = true,
    coarse_density = RCG_DFTK.RecalculateDensity(),
    maxiter = 100,
    coarse_tol = RCG_DFTK.RelativeResTolerance(1e-3, tol),
    coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.3, tol),
    );
println(cb0.times_tot[end] / 1e9)

cb1 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg1 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
    ψ = ψ1, ρ = ρ1, tol, 
    callback = cb1,
    check_convergence_early = true,
    coarse_density = RCG_DFTK.RecalculateDensity(),
    maxiter = 100,
    coarse_tol = RCG_DFTK.RelativeResTolerance(1e-1, tol),
    coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.3, tol),
    );
println(cb1.times_tot[end] / 1e9)


cb2 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg2 = RCG_DFTK.h1_riemannian_conjugate_gradient(basis_f;
    callback = cb2, ψ = ψ1, ρ = ρ1, tol);
println(cb2.times_tot[end] / 1e9)

cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg3 = RCG_DFTK.h1_riemannian_gradient(basis_f;
    callback = cb3, ψ = ψ1, ρ = ρ1, tol);
println(cb3.times_tot[end] / 1e9)


plt1 = plot(; yscale = :log, ylabel = "norm res", xlabel = "CPU time in s")
plot!(cb0.times_tot / 1e9, cb0.norm_residuals, label = "MultiLevel Proj")
ml_times0 = [cb0.times_tot[pk] for pk = 2:(length(cb0.times_tot)-1) if scfres_rcg0.coarse_corrections[pk-1]]
ml_resls0 = [cb0.norm_residuals[pk] for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
plot!(cb1.times_tot / 1e9, cb1.norm_residuals, label = "MultiLevel PsInv")
ml_times = [cb1.times_tot[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
ml_resls = [cb1.norm_residuals[pk] for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
plot!(cb2.times_tot / 1e9, cb2.norm_residuals, label = "H1RCG")
plot!(cb3.times_tot / 1e9, cb3.norm_residuals, label = "H1RG")
scatter!(ml_times0/ 1e9, ml_resls0, label = "coarse corr Proj")
scatter!(ml_times/ 1e9, ml_resls, label = "coarse corr PsInv")
display(plt1)


plt = plot(; yscale = :log, ylabel = "ΔE", xlabel = "CPU time in s")
plot!(cb0.times_tot / 1e9, cb0.Es.-e_ref, label = "MultiLevel Proj")
ml_Es0 = [cb0.Es[pk] - e_ref for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
plot!(cb1.times_tot / 1e9, cb1.Es.-e_ref, label = "MultiLevel PsInv")
ml_Es = [cb1.Es[pk] - e_ref for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
plot!(cb2.times_tot / 1e9, cb2.Es.-e_ref, label = "H1RCG")
plot!(cb3.times_tot / 1e9, cb3.Es.-e_ref, label = "H1RG")
scatter!(ml_times0/ 1e9, ml_Es0, label = "coarse corr Proj")
scatter!(ml_times/ 1e9, ml_Es, label = "coarse corr PsInv")
display(plt)

iter = [i for i = 0:(length(cb1.norm_residuals)-1)]
plt2 = plot(; yscale = :log, ylabel = "ΔE", xlabel = "iter")
plot!(iter, cb0.Es.-e_ref, label = "Proj")
plot!(iter, cb1.Es.-e_ref, label = "PsInv")
ml_iters = [iter[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
ml_iters0 = [iter[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
scatter!(ml_iters0, ml_Es0, label = "coarse corr Proj")
scatter!(ml_iters, ml_Es, label = "coarse corr PsInv")

display(plt2)