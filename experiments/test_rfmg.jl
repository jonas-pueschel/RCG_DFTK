using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots

include("setups/multigrid_setup.jl")

# Playground for a single RFMG run. The model (setup function and Ecut_ref of the reference
# solution) comes from MULTIGRID_MODELS, the Ecuts of the levels can be chosen freely.
model_name = "TiO2"  # "TiO2", "GaAs" or "silicon"
(; setup, Ecut_ref) = MULTIGRID_MODELS[model_name]

#Ecuts = [50.0, 127.3, 240.0, 600.0]
Ecuts = [50.0, 179.2, 600.0]
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

println("\nRFMG")
cb_rfmg = RfmgDefaultCallback(; basis_ref, ψ_ref)
res_rfmg = riemannian_full_multigrid(basis_arr, basis_arr_ref;
    ψ = ψ_c, ρ = ρ_c,
    tol, callback = cb_rfmg,
    #callback_coarse = RcgDefaultCallback(),
    maxiter = 20,
    maxiter_inner = 7,
    refinement_tol = 0.05,
    coarse_tol_outer = ReduceErrorRatioTolerance(0.05),
    coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(5.0) &
                  MinStepsCoarseCondition(1),
    #coarse_cond = ToleranceMinStepCoarseCondition(0.45),
    gradients = [EAGradient(basis) for basis in basis_arr],
    #gradients = [H1Gradient(basis) for basis in basis_arr],
    #cycle = WCycle(1, 1),
    #coarse_density = InterpolateDensity(),
    basis_distance = 1,
    terminate_on_non_descent = true,
)
println(cb_rfmg.times[end])

# nested SCF on the same basis hierarchy, the method to beat
println("\nnested SCF")
cb_nscf = RfmgDefaultCallback(; basis_ref, ψ_ref)
res_nscf = nested_scf(basis_arr, basis_arr_ref;
    ψ = ψ_c, ρ = ρ_c,
    tol = 1e-8, callback = cb_nscf,
    refinement_tol = 0.05,
    check_every = 1,
)
println(cb_nscf.times[end])

# nested EARCG on the same basis hierarchy
println("\nnested EARCG")
cb_nearcg = RfmgDefaultCallback(; basis_ref, ψ_ref)
res_nearcg = nested_earcg(basis_arr, basis_arr_ref;
    ψ = ψ_c, ρ = ρ_c,
    tol = 1e-8, callback = cb_nearcg,
    refinement_tol = 0.05,
    check_every = 1,
)
println(cb_nearcg.times[end])

# dist vs. time
plt_dist = plot(; yscale = :log10, xscale = :log, xlabel = "time (s)", ylabel = "dist")
for (cb, name) in zip([cb_rfmg, cb_nscf, cb_nearcg], ["RFMG", "nSCF", "nEARCG"])
    plot!(plt_dist, cb.times, cb.dists; marker = :circle, markersize = 2, label = name)
end
display(plt_dist)

# Ecut of the basis of every gradient calculation (nested SCF records no levels)
plt_levels = plot(res_rfmg.levels; yscale = :log10, xlabel = "gradient calculation",
    ylabel = "Ecut", label = "RFMG", yticks = (Ecuts, string.(round.(Int, Ecuts))))
display(plt_levels)

plt_levels = plot(res_nearcg.levels; yscale = :log10, xlabel = "gradient calculation",
    ylabel = "Ecut", label = "nEARCG", yticks = (Ecuts, string.(round.(Int, Ecuts))))
display(plt_levels)