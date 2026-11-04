using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

include("setups/ecuts.jl")
include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")

include("./plotting_scripts/plot_iterates.jl")

model_name = "silicon"
setup(; Ecut) = silicon_setup(; Ecut,  kgrid = [4,4,4], supercell_size = [1,1,1], 
    model_terms = :lda)

Ecuts = get_Ecuts(1000; Ecut_min = 9, mode = :factor, factor = 1.7)
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
rcgres_ref = riemannian_full_multigrid(basis_arr, basis_arr_ref;
 ψ = ψ_c, ρ = ρ_c, 
 tol = 0.0,
 #callback_coarse = RcgDefaultCallback(),
 maxiter = 30,
 maxiter_inner = 7,
 #coarse_model_tol_outer = 0.0, 
 coarse_tol_outer = ReduceErrorRatioTolerance(0.2),
 refinement_tol = 0.2,
 #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
 coarse_cond = EstimateErrorCoarseCondition(0.25),
 gradients = [EAGradient(basis) for basis = basis_arr], 
 basis_distance = 2,
);
println("Calculated reference solution in $(rcgres_ref.runtime_ns/1e9) seconds")
rcgres_ref  = energy_adaptive_riemannian_conjugate_gradient(rcgres_ref.basis, 
    ψ = rcgres_ref.ψ, ρ = rcgres_ref.ρ, tol = 1e-13)
ψ_ref = rcgres_ref.ψ
ρ_ref = rcgres_ref.ρ
basis_ref = rcgres_ref.basis


Ecuts = [10.0, 21.0, 38.0, 58.8, 84.8, 117.5, 202.8, 385.7, 713.4]
#Ecuts = get_Ecuts(600; Ecut_min = 9, mode = :factor, factor = 1.7)
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



cb_rfmgv = PlotDiffCallback("EARFMG-V", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts;
    #plotting = [:distribution]
    )
rfmgv = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
        ψ = ψ_c, ρ = ρ_c, 
        maxiter_inner = 3,
        tol = 0,
        is_converged_outer = (tol) -> (info -> false),
        cycle = VCycle(1,1),
        callback_outer = cb_rfmgv);

plt1 = plot(1:length(cb_rfmgv.norm_diffs), cb_rfmgv.norm_diffs; label = "rfmgv", yscale = :log10)

cb_rfmga = PlotDiffCallback("EARFMG-A", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
rfmga = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
        ψ = ψ_c, ρ = ρ_c, tol = 1e-10, 
        refinement_tol = 1e-1,
        coarse_tol_outer = ReduceErrorRatioTolerance(1e-2),
        coarse_cond = EstimateErrorCoarseCondition(0.25),
        callback_outer = cb_rfmga);


cb_nscf = PlotDiffCallback("nSCF", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nscf = nested_scf(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c, 
    is_converged_inner = (info) -> (info.n_iter >= 6),
    callback_inner = cb_nscf
    );


cb_nea = PlotDiffCallback("nEARCG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nearcg = nested_earcg(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c,
    is_converged_inner = (info) -> (info.n_iter >= 4),
    callback_inner = cb_nea
    );



cb_nh1 = PlotDiffCallback("nH1RCG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    )
nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c,
    is_converged_inner = (info) -> (info.n_iter >= 10),
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
