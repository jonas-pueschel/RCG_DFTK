using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
include("setups/GaAs_setup.jl")
include("setups/TiO2_setup.jl")

# keep the system awake while this Julia process runs (Windows only)
# const ES_CONTINUOUS      = 0x80000000
# const ES_SYSTEM_REQUIRED = 0x00000001
# keep_awake()  = ccall((:SetThreadExecutionState, "kernel32"), UInt32, (UInt32,), ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
# allow_sleep() = ccall((:SetThreadExecutionState, "kernel32"), UInt32, (UInt32,), ES_CONTINUOUS)

# keep_awake()

# Silicon lattice constant in Bohr
#Ecuts = [10, 16, 25, 40, 63, 101, 160, 250, 400, 630, 1010]
#Ecuts = [10, 40, 160, 630]
model_name = "silicon"
setup(; Ecut) = silicon_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [2,2,2])
#setup(; Ecut) = TiO2_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [1,1,1])
Ecuts = [10, 16, 25, 40, 63, 101, 160, 250, 400]
#Ecuts = [10,  25,  63, 160,  400]
model, basis_arr = setup(; Ecut = Ecuts); 
_, basis_arr_ref = setup(; Ecut = 1.1 .* Ecuts); 

basis_c = basis_arr[1]
# Initial value
scfres_start = self_consistent_field(basis_c; tol = 1e-2);
ψ_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
ρ_c = scfres_start.ρ

# Convergence tolerance
tol = 1.0e-9;
ψ_ref = nothing
basis_ref = nothing

# Reference solution on the finest grid, needed during the runs to track the distance.
# The silent RFMG run provides the initial guess and also compiles the methods, such that
# the timings below do not contain compilation time.
println("\nReference")
scfres_ref = riemannian_full_multigrid(basis_arr, basis_arr_ref; ψ = ψ_c, ρ = ρ_c, tol = 1e-9,
 maxiter = 20,
 maxiter_inner = 3,
 coarse_model_tol_outer = 0.0,
 refinement_tol = 0.2,
 gradients = [H1Gradient(basis) for basis = basis_arr], 
 only_rcg = false,
 cycle = VCycle(1,1), cycle_finest = VCycle(1,1),
 basis_distance = 1,
);
ψ_ref = scfres_ref.ψ

basis_ref = basis_arr[end]
basis_arr = basis_arr[1:end-1]
basis_arr_ref = basis_arr_ref[1:end-1]


# ψ_c2 = RCG_DFTK.prolongate_point(basis_arr[1], basis_arr[2], ψ_c)
# # check that there are no virtual orbitals
# model = basis_arr[2].model
# filled_occ = DFTK.filled_occupation(model)
# n_spin = model.n_spin_components
# n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
# # number of kpoints and occupation
# Nk = length(basis_arr[2].kpoints)

# occupation = [filled_occ * ones(Float64, n_bands) for ik in 1:Nk]

# ρ_c2 = DFTK.compute_density(basis_arr[2], ψ_c2 , occupation)
# RFMG
println("\nRFMG")
cb_rfmg1 = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_rfmg1 = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
 ψ = ψ_c, ρ = ρ_c, 
 tol, callback = cb_rfmg1,
 #callback_coarse = RcgDefaultCallback(),
 maxiter = 20,
 maxiter_inner = 5,
 #coarse_model_tol_outer = 0.0, 
 coarse_tol_outer = ReduceErrorRatioTolerance(0.2),
 refinement_tol = 0.2,
 #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
 coarse_cond = EstimateErrorCoarseCondition(0.25),
 gradients = [basis.Ecut < 500 ? H1Gradient(basis) : EAGradient(basis) for basis = basis_arr], 
 #coarse_cost_residual = GreedyCostResidual,
 #only_rcg = true,
 #force_post_smoothing = false,
 #cycle = VCycle(0,1), cycle_finest = VCycle(1,0),
 basis_distance = 2,
 );
println(cb_rfmg1.times[end])


# function strip_levels(levels)
#     lmin = min(levels...)
#     last_was_min = true
#     levels_new = []
#     for l = levels
#         if l == lmin 
#             last_was_min && continue
#             last_was_min = true
#         else
#             last_was_min = false
#         end
#         push!(levels_new, l)
#     end
#     return levels_new
# end
levels = strip_levels(res_rfmg1.levels)
plot(1:length(levels), levels, ;yscale = :log10)

# # RFMG
# println("\nRFMG")
# cb_rfmg2 = RfmgDefaultCallback(;basis_ref, ψ_ref)
# res_rfmg2 = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
#  ψ = ψ_c, ρ = ρ_c, 
#  tol, callback = cb_rfmg2,
#  #callback_coarse = RcgDefaultCallback(),
#  maxiter = 20,
#  maxiter_inner = 10,
#  #coarse_model_tol_outer = 0.0, 
#  coarse_tol_outer = ReduceErrorRatioTolerance(0.2),
#  refinement_tol = 0.4,
#  #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
#  coarse_cond = EstimateErrorCoarseCondition(0.25),
#  gradients = [H1Gradient(basis) for basis = basis_arr], 
#  #coarse_cost_residual = GreedyCostResidual,
#  #only_rcg = true,
#  cycle = VCycle(1,1), cycle_finest = VCycle(1,0),
#  basis_distance = 1,
#  );
# println(cb_rfmg2.times[end])

# levels = strip_levels(res_rfmg2.levels)
# plot(1:length(levels), levels, ;yscale = :log10)

# nested EARCG
println("\nnested EARCG")
cb_nearcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nearcg = nested_earcg(basis_arr, basis_arr_ref; 
    tol = 1e-8, refinement_tol = 0.2, 
    check_every = 1, ψ = ψ_c, ρ = ρ_c, callback = cb_nearcg);
println(cb_nearcg.times[end])

# nested H1RCG
# println("\nnested H1RCG")
# cb_nh1rcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
# res_nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref; tol = 1e-8, refinement_tol = 0.2, 
#     check_every = 1, ψ = ψ_c, ρ = ρ_c, callback = cb_nh1rcg);
# println(cb_nh1rcg.times[end])

# nested SCF
println("\nnested SCF")
cb_nscf = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nscf = nested_scf(basis_arr, basis_arr_ref; tol = 1e-8, 
refinement_tol = 0.2, check_every = 1,
ψ = ψ_c, ρ = ρ_c, 
callback = cb_nscf);
println(cb_nscf.times[end])

# function get_occ(basis)
#     model = basis.model
#     filled_occ = DFTK.filled_occupation(model)
#     n_spin = model.n_spin_components
#     n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
#     Nk = length(basis.kpoints)
#     occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]
#     return occupation
# end

# basis_f = basis_arr[end]
# cb_ml = RfmgDefaultCallback(;basis_arr, ψ_ref)
# ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ_c)
# ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))
# result = multilevel_riemannian_optimization(basis_arr; ψ = ψ1, ρ = ρ1, tol,
#     callback = cb_ml,
#     coarse_model_tol = 1e-2,
#     coarse_cond_tol = 0.45);


# dist vs. time
plt = plot(; yscale = :log10, xscale = :log, ylabel = "dist", xlabel = "time (s)")
mkpath("experiments/data")
for (cb, name) = zip([cb_rfmg1, cb_nearcg, cb_nscf], ["RFMG", "nEARCG", "nSCF"])
    plot!(plt, cb.times, max.(cb.dists), label = name, marker = :circle, markersize = 2)
    if @isdefined(model_name) && model_name != ""
        open("experiments/data/rfmg_$(model_name)_$(name)_dist_time.dat", "w") do io
            println(io, "x y Ecut")
            writedlm(io, [cb.times cb.dists cb.Ecuts], ' ')
        end
    end
end
display(plt)

#allow_sleep() 