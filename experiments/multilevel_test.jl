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
tol = 1.0e-8;

# Initial value
scfres_start = self_consistent_field(basis_c; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(basis_c.model));
ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_c = scfres_start.ρ
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
ρ1 = DFTK.interpolate_density(ρ1_c, basis_c, basis_f)

init_norm_res = RCG_DFTK.init_norm_res(ψ1, basis_f)

# multilevel
println("\nMultilevel")
default_callback = RCG_DFTK.RcgDefaultCallback()

cb1 = TrackResTimeCallback(default_callback, init_norm_res)
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


cb2 = TrackResTimeCallback(default_callback, init_norm_res)
scfres_rcg2 = RCG_DFTK.h1_riemannian_conjugate_gradient(basis_f;
    callback = cb2, ψ = ψ1, ρ = ρ1, tol);
println(cb2.times_tot[end] / 1e9)


plt = plot(; yscale = :log, ylabel = "norm res", xlabel = "CPU time in s")
plot!(cb1.times_tot / 1e9, cb1.norm_residuals, label = "MultiLevel")
ml_times = [cb1.times_tot[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
ml_resls = [cb1.norm_residuals[pk] for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
plot!(cb2.times_tot / 1e9, cb2.norm_residuals, label = "H1RCG")
scatter!(ml_times/ 1e9, ml_resls, label = "coarse corr")

