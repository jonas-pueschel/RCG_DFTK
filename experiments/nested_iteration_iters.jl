using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

include("setups/multigrid_setup.jl")
include("./plotting_scripts/plot_iterates.jl")

model_name = "TiO2"  # "TiO2", "GaAs" or "silicon"
(; model, Ecuts, Ecuts_ref, basis_arr, basis_arr_ref, ψ_c, ρ_c, ψ_ref, ρ_ref, basis_ref) =
    multigrid_setup(model_name)

# data_path = "./experiments/data/$model_name-nested-it/"
# img_path = "./experiments/imgs/$model_name-nested-it/"

data_path = "./experiments/data/$model_name-rfmg/"
img_path = "./experiments/imgs/$model_name-rfmg/"

plot_ψ(ψ_ref, basis_ref, Ecuts; name = "$(model_name)-reference", data_path, img_path)

# cb_nh1 = PlotDiffCallback("nH1RCG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
#     #plotting = [:distribution],
#     img_path, data_path,
#     heatmap_lims = (1e-9, 1e-2)
#     )
# nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c,
#     #refinement_tol = 0.05,
#     is_converged_inner = (info) -> (info.n_iter >= 10),
#     callback_inner = cb_nh1
#     );

cb_rfmga = PlotDiffCallback("H1RFMG", model_name, ψ_ref, ρ_ref, basis_ref, Ecuts; 
    #plotting = [:distribution]
    img_path, data_path,
    heatmap_lims = (1e-9, 1e-2)
    )
rfmga = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
        ψ = ψ_c, ρ = ρ_c, tol = 1e-10, 
        refinement_tol = 0.05,
        gradients = [H1Gradient(basis) for basis in basis_arr],
        coarse_tol_outer = ReduceErrorRatioTolerance(0.05),
        coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(5.0) & MinStepsCoarseCondition(1),
        callback_outer = cb_rfmga);

# norm of the difference to the reference solution per iteration (x = 1, 2, 3, ...), for tikz
# mkpath(data_path)
# open("$(data_path)$(model_name)-H1RCG-norm-diffs.dat", "w") do io
#     println(io, "x y")
#     writedlm(io, zip(eachindex(cb_nh1.norm_diffs), cb_nh1.norm_diffs), ' ')
# end
# norm of the difference to the reference solution per iteration (x = 1, 2, 3, ...), for tikz
mkpath(data_path)
open("$(data_path)$(model_name)-H1RFMG-norm-diffs.dat", "w") do io
    println(io, "x y")
    writedlm(io, zip(eachindex(cb_rfmga.norm_diffs), cb_rfmga.norm_diffs), ' ')
end
