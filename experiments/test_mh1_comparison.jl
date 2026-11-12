using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots

include("setups/multigrid_setup.jl")

# Playground for a single RFMG run. The model (setup function and Ecut_ref of the reference
# solution) comes from MULTIGRID_MODELS, the Ecuts of the levels can be chosen freely.
model_name = "TiO2"  # "TiO2", "GaAs" or "silicon"
(; setup, Ecut_ref) = MULTIGRID_MODELS[model_name]

Ecuts = [50.0, 127.3, 240.0, 400.0]
#Ecuts = [50.0, 84.0, 127.3, 179.2, 240.0, 600.0]
#Ecuts = [10.0, 39.0, 87.0, 154.0, 240.0]
#Ecuts = get_Ecuts(700; Ecut_min = 9, mode = :sqrt, Δ_pre = 1.55, Δ_post = 6, Ecut_kink = 100)
Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1)
#Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :offset, offset = 1.0)

model, basis_arr = setup(; Ecut = Ecuts)
_, basis_arr_ref = setup(; Ecut = Ecuts_ref)

# Initial value on the coarsest basis
ψ_c, ρ_c = initial_value(basis_arr[1]; tol = 1e-4)

# Reference solution (loaded if reference_solutions/<model_name>_ref.BSON has the same Ecut_ref)
ψ_ref, ρ_ref, basis_ref = reference_solution(model_name, setup, Ecuts, Ecut_ref, ψ_c, ρ_c)

# Convergence tolerance
tol = 1.0e-9
refinement_tol = 0.05

println("\nRFMG")
cb_rfmg = RfmgDefaultCallback(; basis_ref, ψ_ref)
res_rfmg = riemannian_full_multigrid(basis_arr, basis_arr_ref;
    ψ = ψ_c, ρ = ρ_c,
    tol, callback = cb_rfmg,
    callback_outer = cb_rfmg,
    #callback_coarse = RcgDefaultCallback(),
    maxiter = 20,
    maxiter_inner = 7,
    refinement_tol = 0.05,
    coarse_tol_outer = ReduceErrorRatioTolerance(0.05),
    coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(5.0) &
                  MinStepsCoarseCondition(1),
    #coarse_cond = ToleranceMinStepCoarseCondition(0.45),
    #gradients = [H1Gradient(basis) for basis in basis_arr],
    #cycle = WCycle(1, 1),
    #coarse_density = InterpolateDensity(),
    basis_distance = 1,
    terminate_on_non_descent = true,
)
println(cb_rfmg.times[end])

cb_rfmg.dists

cb_rfmg.times
multilevel_tikz(res_rfmg.levels,algorithm = :rfmg, file = "multilevel_pattern.tex")

# nested H1RCG
println("\nnested H1RCG")
cb_nh1rcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref;
    tol = 1e-8, refinement_tol,
    check_every = 1, ψ = ψ_c, ρ = ρ_c, callback_inner = cb_nh1rcg);
println(cb_nh1rcg.times[end])

# H1RCG
basis_f = basis_arr[end]
ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ_c)
ρ1 = RCG_DFTK.compute_density(basis_f, ψ1)
is_converged = NestedItConvergenceErrEstimate(
        EstimateNorm(refinement_tol), tol)
RCG_DFTK.reset!(is_converged, basis_arr_ref[end], true)
println("\nH1RCG")
cb_h1rcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_h1rcg = h1_riemannian_conjugate_gradient(basis_arr[end];
    ψ = ψ1, ρ = ρ1, callback = cb_h1rcg,
    is_converged);
println(cb_h1rcg.times[end])

# nested SCF
println("\nnested SCF")
cb_nscf = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nscf = nested_scf(basis_arr, basis_arr_ref;
    tol = 1e-8, refinement_tol, check_every = 1,
    ψ = ψ_c, ρ = ρ_c,
    #fermialg = RCG_DFTK.FillLowestBands(),
    callback_inner = cb_nscf);
println(cb_nscf.times[end])

# SCF
is_converged = NestedItConvergenceErrEstimate(
        EstimateNorm(refinement_tol), tol)
RCG_DFTK.reset!(is_converged, basis_arr_ref[end], true)
println("\nH1RCG")
cb_scf = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_scf = self_consistent_field(basis_arr[end];
    ψ = ψ1, ρ = ρ1, callback = cb_scf,
    is_converged);
println(cb_scf.times[end])

# dist vs. time
plt_dist = plot(; yscale = :log10, xscale = :log, xlabel = "time (s)", ylabel = "dist")
for (cb, name) in zip([cb_rfmg, cb_nh1rcg, cb_h1rcg, cb_nscf, cb_scf], ["RFMG", "nH1RCG", "H1RCG", "nSCF", "SCF"])
    plot!(plt_dist, cb.times, cb.dists; marker = :circle, markersize = 2, label = name)
end
display(plt_dist)