using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles
using BSON

include("setups/ecuts.jl")
include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")

include("./plotting_scripts/plot_iterates.jl")

model_name = "TiO2"
setup(; Ecut) = TiO2_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [1,1,1], 
    model_terms = :pbe)

Ecuts = get_Ecuts(800; Ecut_min = 10, mode = :factor, factor = 1.7)
Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1)

# Ecuts = get_Ecuts(1000; Ecut_min = 9, mode = :sqrt, Δ_pre = 1.55, Δ_post = 6, Ecut_kink = 100)
# Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :offset, offset = 0.57)


model, basis_arr = setup(; Ecut = Ecuts); 
_, basis_arr_ref = setup(; Ecut = Ecuts_ref); 



basis_c = basis_arr[1]
# Initial value
scfres_start = h1_riemannian_gradient(basis_c; tol = 1e-4, callback = (info) -> nothing);
ψ_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ_c = scfres_start.ρ

# Convergence tolerance
tol = 1.0e-9;


basis_f = basis_arr[end]

println("\nReference")
rcgres_ref1 = riemannian_full_multigrid(basis_arr, basis_arr_ref;
 ψ = ψ_c, ρ = ρ_c, 
 tol = 0.0,
#  callback_coarse = RfmgDefaultCallback(),
#  callback_outer = RfmgDefaultCallback(),
#  callback_mid = RfmgDefaultCallback(),
 maxiter = 30,
 maxiter_inner = 7,
 #coarse_model_tol_outer = 0.0, 
 coarse_tol_outer = ReduceErrorRatioTolerance(0.1),
 refinement_tol = 0.1,
 #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
 coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(10.0) & MinStepsCoarseCondition(1),
 gradients = [EAGradient(basis) for basis = basis_arr], 
 terminate_on_non_descent = true,
);

println("Calculated intermediate reference solution in $((rcgres_ref1.runtime_ns)/1e9) seconds")
plot_ψ(rcgres_ref1.ψ, rcgres_ref1.basis, Ecuts; name = "$(model_name)-reference-1")


rcgres_ref  = energy_adaptive_riemannian_conjugate_gradient(rcgres_ref1.basis, 
    ψ = rcgres_ref1.ψ, ρ = rcgres_ref1.ρ, tol = 1e-9)

println("Refined reference solution in $((rcgres_ref.runtime_ns)/1e9) seconds")
ψ_ref = rcgres_ref.ψ
ρ_ref = rcgres_ref.ρ
basis_ref = rcgres_ref.basis
Ecut_ref = rcgres_ref.basis.Ecut
BSON.@save "experiments/reference_solutions/$(model_name)_ref.BSON" ψ_ref ρ_ref Ecut_ref

function sqrt_eq_steps(first, last, n_steps)
    fs = sqrt(first)
    ls = sqrt(last)
    stepsize = (ls - fs)/(n_steps-1)
    return [(fs + a * stepsize)^2 for a = 0:(n_steps-1)]
end

# # silicon and GaAs
# Ecuts = [10.0, 21.0, 38.0, 58.8, 84.8, 117.5, 202.8, 385.7, 713.4]
# TiO2
Ecuts = [10.0, 39.0, 87.0, 154.0, 240.0, 640.0]
Ecuts = [50.0, 84.0, 127.3, 179.2, 240.0, 600]
# #Ecuts = get_Ecuts(600; Ecut_min = 9, mode = :factor, factor = 1.7)
Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1)

plot_ψ(ψ_ref, basis_ref, Ecuts; name = "$(model_name)-reference")

# Ecuts = get_Ecuts(1000; Ecut_min = 9, mode = :sqrt, Δ_pre = 1.55, Δ_post = 6, Ecut_kink = 100)
# Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :offset, offset = 0.57)


model, basis_arr = setup(; Ecut = Ecuts); 
_, basis_arr_ref = setup(; Ecut = Ecuts_ref); 

basis_c = basis_arr[1]
# Initial value
scfres_start = h1_riemannian_gradient(basis_c; tol = 1e-4, callback = (info) -> nothing);
ψ_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ_c = scfres_start.ρ



cb_rfmga = PlotDiffCallback("EARFMG-A", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
rfmga = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
        ψ = ψ_c, ρ = ρ_c, tol = 1e-10, 
        refinement_tol = 0.1,
        coarse_tol_outer = ReduceErrorRatioTolerance(1e-2),
        coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(5.0) & MinStepsCoarseCondition(1),
        callback_outer = cb_rfmga);


display(plot(1:length(rfmga.levels), rfmga.levels))

cb_nscf = PlotDiffCallback("nSCF", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nscf = nested_scf(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c, 
    refinement_tol = 0.1,
    callback_inner = cb_nscf
    );


cb_nea = PlotDiffCallback("nEARCG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nearcg = nested_earcg(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c,
    refinement_tol = 0.1,
    callback_inner = cb_nea
    );



cb_nh1 = PlotDiffCallback("nH1RCG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c,
    refinement_tol = 0.1,
    #is_converged_inner = (info) -> (info.n_iter >= 10),
    callback_inner = cb_nh1
    );

plt1 = plot(1:length(cb_rfmgv.norm_diffs), cb_rfmgv.norm_diffs; label = "rfmgv", yscale = :log10)
plot!(plt1, 1:length(cb_rfmga.norm_diffs), cb_rfmga.norm_diffs; label = "rfmga", yscale = :log10)
plot!(plt1, 1:length(cb_nscf.norm_diffs), cb_nscf.norm_diffs; label = "nscf", yscale = :log10)
plot!(plt1, 1:length(cb_nea.norm_diffs), cb_nea.norm_diffs; label = "nea", yscale = :log10)
plot!(plt1, 1:length(cb_nh1.norm_diffs), cb_nh1.norm_diffs; label = "nh1", yscale = :log10)
display(plt1)

# intres = riemannian_full_multigrid(basis_arr[1:end-1], basis_arr_ref[1:end-1];
#  ψ = ψ_c, ρ = ρ_c, 
#  tol = 1e-11,
#  #callback_coarse = RcgDefaultCallback(),
#  maxiter = 30,
#  maxiter_inner = 7,
#  #coarse_model_tol_outer = 0.0, 
#  coarse_tol_outer = ReduceErrorRatioTolerance(0.2),
#  refinement_tol = 0.2,
#  #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
#  coarse_cond = EstimateErrorCoarseCondition(0.25),
#  gradients = [EAGradient(basis) for basis = basis_arr], 
#  basis_distance = 2,
# );
# rcgres  = energy_adaptive_riemannian_conjugate_gradient(intres.basis, 
#     ψ = intres.ψ, ρ = intres.ρ, tol = 1e-10)

# ψ_fine = RCG_DFTK.interpolate_c2f(rcgres.basis, basis_f, intres.ψ)

# out_fine = compute_all(basis_f, ψ_fine)
# res_fine, ρ_fine, Λ_fine = out_fine.res, out_fine.ρ, out_fine.Λ
# H_fine, Hψ_fine = out_fine.H, out_fine.Hψ
# err_est_fine = RCG_DFTK.invert_refinement_metric_improved(basis_f, ψ_fine, res_fine) 

# diff = RCG_DFTK.grassmann_diff_DFTK(ψ_fine, rcgres_ref.ψ)

# #RCG_DFTK.norm_DFTK(basis_f, res_fine)

# ea_grad = RCG_DFTK.calculate_gradient(ψ_fine, Hψ_fine, H_fine, Λ_fine, res_fine,
#                                       EAGradient(basis_f; tol = 1e-2))

# h1_grad = RCG_DFTK.calculate_gradient(ψ_fine, Hψ_fine, H_fine, Λ_fine, res_fine,
#                                       H1Gradient(basis_f))

# # ea_grad_step = RCG_DFTK.calculate_retraction(ψ_fine, -ea_grad, 1.0, RetractionPolar())
# # h1_grad_step = RCG_DFTK.calculate_retraction(ψ_fine, -h1_grad, 1.0, RetractionPolar())
# # err_est_step = RCG_DFTK.calculate_retraction(ψ_fine, -err_est_fine, 1.0, RetractionPolar())
# ea_grad_step = energy_adaptive_riemannian_conjugate_gradient(basis_f; ψ = ψ_fine, ρ = ρ_fine, tol = 0, maxiter = 1).ψ;
# h1_grad_step = h1_riemannian_conjugate_gradient(basis_f; ψ = ψ_fine, ρ = ρ_fine, tol = 0, maxiter = 1).ψ;
# scf_step = self_consistent_field(basis_f; ψ = ψ_fine, ρ = ρ_fine)


# as, es = calculate_energy_distribution(rcgres_ref.ψ, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2))


# as, es = calculate_energy_distribution(ψ_fine, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2))

# as, es = calculate_energy_distribution(diff, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2), label = "diff")

# as, es = calculate_energy_distribution(ea_grad, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2), label = "ea grad")

# as, es = calculate_energy_distribution(h1_grad, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2), label = "h1 grad")

# as, es = calculate_energy_distribution(err_est_fine, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2), label = "err est")


# as, es = calculate_energy_distribution(res_fine, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-13, 2), label = "res")

# as, es = calculate_energy_distribution(diff, basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-15, 1e-5), label = "diff")

# RCG_DFTK.grassmann_dist_DFTK(basis_f, ψ_ref, ea_grad_step)

# as, es = calculate_energy_distribution(RCG_DFTK.grassmann_diff_DFTK(ψ_ref, ea_grad_step), basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# as = max.(as, 1e-20)
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-15, 1e-5), label = "ea step")

# RCG_DFTK.grassmann_dist_DFTK(basis_f, ψ_ref, h1_grad_step)

# as, es = calculate_energy_distribution(RCG_DFTK.grassmann_diff_DFTK(ψ_ref, h1_grad_step), basis_f)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# as = max.(as, 1e-20)
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-15, 1e-5), label = "h1 step")

# as, es = calculate_energy_distribution(intres.ψ, intres.basis)
# xs = sqrt.(2 .* es)          # |k+G|
# Eticks = Ecuts
# as = max.(as, 1e-20)
# scatter(xs, as, yscale = :log10,
#         xticks = (sqrt.(2 .* Eticks), string.(Eticks)),
#         xlabel = "E_kin (√ scale)", ylims = (1e-15, 1e-5), label = "intres")
