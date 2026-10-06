using DFTK
using RCG_DFTK
using PseudoPotentialData
using LinearAlgebra
using Plots

# this script forces precompliation for all methods, ensuring comparability in runtime
include("precompile_methods.jl");
precompile_methods();

include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")
# Silicon lattice constant in Bohr
# model, [basis_c, basis_f] = silicon_setup(; Ecut = [15, 70], kgrid = [4,4,4]);
model, [basis_c, basis_f] = silicon_setup(; Ecut = [15, 70], kgrid = [2,2,2]);
# model, [basis_c, basis_f] = TiO2_setup(; Ecut = [15, 70], kgrid = [2,2,2]);

# Convergence tolerance
tol = 1.0e-6;

# Initial value
# scfres_start = self_consistent_field(basis_c; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(basis_c.model));
# ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
# ρ1_c = scfres_start.ρ
# ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
# ρ1 = DFTK.interpolate_density(ρ1_c, basis_c, basis_f)
scfres_start = self_consistent_field(basis_f; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(basis_c.model));
ψ1_f = DFTK.select_occupied_orbitals(basis_f, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1_f = scfres_start.ρ
ψ1 = ψ1_f
ρ1 = ρ1_f

# scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
# e_ref = scfres_ref.energies.total 

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
init_e = es0.total


# multilevel
default_callback = RCG_DFTK.RcgDefaultCallback()

cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg0 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
    ψ = ψ1, ρ = ρ1, tol, 
    callback = cb0,
    #multilevel_map = ProjectionMap(),
    gradient = EAGradient(basis_f, CorrectedRelativeΛShift(μ = 0.0)),
    check_convergence_early = true,
    maxiter = 100,
    coarse_tol = RCG_DFTK.RelativeResTolerance(1e-1, tol),
    coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.45, tol),
    iteration_strat_fine = StandardBacktracking(
        ArmijoRule(0.1, 0.5),
        ConstantStep(1.0), 10
    ),
        coarse_solver = (basis_c) -> ea_coarse_solver(basis_c, 10),
    do_rayleigh_ritz = false,
    );
println(cb0.times_tot[end] / 1e9)

cb1 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg1 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
    ψ = ψ1, ρ = ρ1, tol, 
    callback = cb1,
    gradient = H1Gradient(basis_f),
    check_convergence_early = true,
    coarse_density = RCG_DFTK.RecalculateDensity(),
    maxiter = 100,
    coarse_tol = RCG_DFTK.RelativeResTolerance(1e-1, tol),
    coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.3, tol),
    iteration_strat_fine = StandardBacktracking(
        ArmijoRule(0.1, 0.5),
        ConstantStep(1.0), 10
    )
    );
println(cb1.times_tot[end] / 1e9)


cb2 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg2 = RCG_DFTK.h1_riemannian_conjugate_gradient(basis_f;
    callback = cb2, ψ = ψ1, ρ = ρ1, tol);
println(cb2.times_tot[end] / 1e9)

# cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# scfres_rcg3 = RCG_DFTK.h1_riemannian_gradient(basis_f;
#     callback = cb3, ψ = ψ1, ρ = ρ1, tol);
# println(cb3.times_tot[end] / 1e9)

cb4 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
scfres_rcg4 = RCG_DFTK.energy_adaptive_riemannian_conjugate_gradient(basis_f;
    callback = cb4, ψ = ψ1, ρ = ρ1, tol);
println(cb4.times_tot[end] / 1e9)

# cb5 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
# scfres_scf = self_consistent_field(basis_f;
#     callback = cb5, ψ = ψ1, ρ = ρ1, tol = 1e-7);
# println(cb5.times_tot[end] / 1e9)

e_ref = min(cb0.Es..., cb4.Es..., ) - 1e-14
ml_times0 = [cb0.times_tot[pk] for pk = 2:(length(cb0.times_tot)-1) if scfres_rcg0.coarse_corrections[pk-1]]
ml_resls0 = [cb0.norm_residuals[pk] for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
ml_times = [cb1.times_tot[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
ml_resls = [cb1.norm_residuals[pk] for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
plt1 = plot(; yscale = :log, ylabel = "norm res", xlabel = "CPU time in s")
plot!(cb1.times_tot / 1e9, cb1.norm_residuals, label = "H1ML")
scatter!(ml_times/ 1e9, ml_resls, label = "coarse corr H1")
plot!(cb0.times_tot / 1e9, cb0.norm_residuals, label = "EAML")
scatter!(ml_times0/ 1e9, ml_resls0, label = "coarse corr EA")
plot!(cb2.times_tot / 1e9, cb2.norm_residuals, label = "H1RCG")
plot!(cb4.times_tot / 1e9, cb4.norm_residuals, label = "EARCG")
#plot!(cb3.times_tot / 1e9, cb3.norm_residuals, label = "H1RG")
#plot!(cb5.times_tot / 1e9, cb5.norm_residuals, label = "SCF")
display(plt1)

ml_Es0 = [cb0.Es[pk] - e_ref for pk = 2:(length(cb0.norm_residuals)-1) if scfres_rcg0.coarse_corrections[pk-1]]
ml_Es = [cb1.Es[pk] - e_ref for pk = 2:(length(cb1.norm_residuals)-1) if scfres_rcg1.coarse_corrections[pk-1]]
plt = plot(; yscale = :log, ylabel = "ΔE", xlabel = "CPU time in s")
plot!(cb1.times_tot / 1e9, cb1.Es.-e_ref, label = "H1ML")
scatter!(ml_times/ 1e9, ml_Es, label = "coarse corr H1")
plot!(cb0.times_tot / 1e9, cb0.Es.-e_ref, label = "EAML")
scatter!(ml_times0/ 1e9, ml_Es0, label = "coarse corr EA")
plot!(cb2.times_tot / 1e9, cb2.Es.-e_ref, label = "H1RCG")
plot!(cb4.times_tot / 1e9, cb4.Es.-e_ref, label = "EARCG")
#plot!(cb3.times_tot / 1e9, cb3.Es.-e_ref, label = "H1RG")
#plot!(cb5.times_tot / 1e9, cb5.Es.-e_ref, label = "SCF")
display(plt)

iter = [i for i = 0:(length(cb1.norm_residuals)-1)]
iter0 = [i for i = 0:(length(cb0.norm_residuals)-1)]
plt2 = plot(; yscale = :log, ylabel = "ΔE", xlabel = "iter")
plot!(iter0, cb0.Es.-e_ref, label = "EAML")
plot!(iter, cb1.Es.-e_ref, label = "H1ML")
ml_iters = [iter[pk] for pk = 2:(length(cb1.times_tot)-1) if scfres_rcg1.coarse_corrections[pk-1]]
ml_iters0 = [iter0[pk] for pk = 2:(length(cb0.times_tot)-1) if scfres_rcg0.coarse_corrections[pk-1]]
scatter!(ml_iters0, ml_Es0, label = "coarse corr EA")
scatter!(ml_iters, ml_Es, label = "coarse corr H1")

display(plt2)