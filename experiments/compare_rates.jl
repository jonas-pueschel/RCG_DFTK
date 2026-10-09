using DFTK
using RCG_DFTK
using PseudoPotentialData

include("calculate_cond.jl")

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr
model, basis = silicon_setup(; Ecut = 30, kgrid = [4,4,4]);

include("setups/TiO2_setup.jl")
# TiO2 lattice constant in Bohr
model, basis = TiO2_setup(; Ecut = 50, kgrid = [2,2,2]);


# Convergence tolerance
tol = 1.0e-8;

# Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis)

init_norm_res = RCG_DFTK.norm_DFTK(basis, res0)
init_e = es0.total

#default callback
default_callback = RcgDefaultCallback();

# we note that there is a discrepancy between the Time_tot we print and the sums of Δtime 
# this is caused by the fact that time_tot is read from DFTK.timer and Δtime uses time.ns()
# Thus, Δtime also accounts for time not "spent in" the DFTK.timer, like calculation of 
# the residual for SCF and other overhead caused by the benchmarking tools. 


# reference sol
println("\nReference sol")
DFTK.reset_timer!(DFTK.timer)
reference = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, 
    tol = 1e-10,
    );

ψ = reference.ψ
ρ = reference.ρ
H = reference.ham
Hψ = H * ψ
Λ = [Hψ[ik]'ψ[ik] for ik = 1:length(basis.kpoints)]
Λ = 0.5 * [Λ[ik]' + Λ[ik] for  ik = 1:length(basis.kpoints)]
occupation = reference.occupation

κ_ea, _, _ = calculate_cond(inv_ea_metric(basis, ψ, Hψ, H, Λ; tol = 1e-6, itmax = 30), basis, ψ, ρ, H, Λ, occupation;
    κtol = 1e-4, rtol = 1e-6)

κ_h1, _, _ = calculate_cond(inv_h1_metric(basis, ψ), basis, ψ, ρ, H, Λ, occupation; 
    κtol = 1e-4, rtol = 1e-6)

κ_l2, _, _ = calculate_cond(inv_l2_metric(), basis, ψ, ρ, H, Λ, occupation;
    κtol = 1e-4, rtol = 1e-6)



function expected_steps_cg(κ::Real; κ_l2::Real = κ, r0=1e-1, rtol=1e-8)
    κ >= 1 || throw(ArgumentError("κ must be ≥ 1"))
    κ == 1 && return 1                      # converges in one step
    ε = rtol / r0
    C = 2 * sqrt(κ_l2)
    k = log(C / ε) / log((s + 1) / (s - 1))
    return max(ceil(Int, k), 0)
end

function expected_steps_gd(κ::Real; r0=1e-1, rtol=1e-8)
    κ >= 1 || throw(ArgumentError("κ must be ≥ 1"))
    κ == 1 && return 1
    ε = rtol / r0
    C = sqrt(κ) #TODO, this can be made smaller if all other kappas are known
    k = log(C / ε) / log((κ + 1) / (κ - 1))
    return max(ceil(Int, k), 0)
end


# EARCG-Gr
println("\nEARCG-Gr")
callback_earcg_gr = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, tol, 
    iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.0),
    callback = callback_earcg_gr);
println("Time_tot (s): $((callback_earcg_gr.times_tot[end])/ 1e9)")
println("$(scfres_rcg1.n_iter) steps (expected: $(expected_steps_cg(κ_ea; r0 = init_norm_res)))")

# H1RCG
println("\nH1RCG")
callback_h1rcg = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rcg2 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
    callback = callback_h1rcg,
    iteration_strat = GreedySecantStrategy(ApproxHessianStep(), 0.0));
println("Time_tot (s): $((callback_h1rcg.times_tot[end])/ 1e9)")
println("$(scfres_rcg2.n_iter) steps (expected: $(expected_steps_cg(κ_h1; r0 = init_norm_res)))")

# L2RCG
println("\nL2RCG")
callback_l2rcg = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rcg3 = l2_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
    iteration_strat = GreedySecantStrategy(ApproxHessianStep(), 0.0),
    callback = callback_l2rcg);
println("Time_tot (s): $((callback_l2rcg.times_tot[end])/ 1e9)")
println("$(scfres_rcg3.n_iter) steps (expected: $(expected_steps_cg(κ_l2; r0 = init_norm_res)))")

# EARGD-Gr
println("\nEARGD-Gr")
callback_eargd_gr = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rgd1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, μ = 0, tol, 
    callback = callback_eargd_gr,
    iteration_strat = GreedySecantStrategy(ConstantStep(1.0), 0.0),
    cg_param = ParamZero());
println("Time_tot (s): $((callback_eargd_gr.times_tot[end])/ 1e9)")
println("$(scfres_rgd1.n_iter) steps (expected: $(expected_steps_gd(κ_ea; r0 = init_norm_res)))")

# H1RGD
println("\nH1RGD")
callback_h1rgd = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rgd2 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
    callback = callback_h1rgd,
    iteration_strat = GreedySecantStrategy(ApproxHessianStep(), 0.0),
    cg_param = ParamZero());
println("Time_tot (s): $((callback_h1rgd.times_tot[end])/ 1e9)")
println("$(scfres_rgd2.n_iter) steps (expected: $(expected_steps_gd(κ_h1; r0 = init_norm_res)))")

# L2RGD
println("\nL2RGD")
callback_l2rgd = TrackResTimeCallback(default_callback, init_norm_res, init_e)
DFTK.reset_timer!(DFTK.timer)
scfres_rgd3 = l2_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol,
    callback = callback_l2rgd,
    iteration_strat = GreedySecantStrategy(ApproxHessianStep(), 0.0),
    cg_param = ParamZero());
println("Time_tot (s): $((callback_l2rgd.times_tot[end])/ 1e9)")
println("$(scfres_rgd1.n_iter) steps (expected: $(expected_steps_gd(κ_l2; r0 = init_norm_res)))")
