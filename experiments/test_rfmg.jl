using DFTK
using RCG_DFTK
using PseudoPotentialData
using Plots
using DelimitedFiles

# run each method used below once, such that the timings do not contain compilation time
# (h1rg: initial value; rfmg: reference and RFMG runs; nested_it: nested EARCG, H1RCG and SCF)
include("precompile_methods.jl");
precompile_methods(["rfmg", "nested_it", "h1rg"]);

include("setups/ecuts.jl")
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
setup(; Ecut) = silicon_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [1,1,1], model_terms = :lda, supersampling = 2)
#setup(; Ecut) = silicon_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [1,1,1])
#setup(; Ecut) = TiO2_setup(; Ecut,  kgrid = [2,2,2], supercell_size = [1,1,1])
#Ecuts = [10, 16, 25, 40, 63, 101, 160, 250, 400, 630]
#Ecuts = [10, 25, 63,  160, 400, 500]

Ecuts = get_Ecuts(1000; Ecut_min = 9, mode = :factor, factor = 1.7)
get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1)

Ecuts = get_Ecuts(1000; Ecut_min = 9, mode = :sqrt, Δ_pre = 1.55, Δ_post = 6, Ecut_kink = 100)
get_Ecuts_ref(Ecuts; mode = :offset, offset = 0.57)
#TiO2
#Ecuts = [40, 63, 101, 160, 250, 400, 500]

model, basis_arr = setup(; Ecut = Ecuts); 
_, basis_arr_ref = setup(; Ecut = 1.1 .* Ecuts); 

# Ecuts = [16, 40, 101, 250]

basis_c = basis_arr[1]
# Initial value
#scfres_start = self_consistent_field(basis_c; tol = 1e-2);
scfres_start = h1_riemannian_gradient(basis_c; tol = 1e-4);
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
scfres_ref = riemannian_full_multigrid(basis_arr, basis_arr_ref;
 ψ = ψ_c, ρ = ρ_c, 
 tol = 1e-11,
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
println("Calculated reference solution in $(scfres_ref.runtime_ns/1e9) seconds")
# scfres_ref = energy_adaptive_riemannian_conjugate_gradient(scfres_ref.basis; 
#     ρ = scfres_ref.ρ, ψ = scfres_ref.ψ, tol = 1e-11)
ψ_ref = scfres_ref.ψ
res_ref = scfres_ref.res


basis_ref = basis_arr[end]


Ecuts = get_Ecuts(700; Ecut_min = 9, mode = :factor, factor = 10^(0.2))
get_Ecuts_ref(Ecuts; mode = :factor, factor = 1.1)

# Ecuts = get_Ecuts(700; Ecut_min = 9, mode = :sqrt, Δ_pre = 1.55, Δ_post = 6, Ecut_kink = 100)
# get_Ecuts_ref(Ecuts; mode = :offset, offset = 0.57)

model, basis_arr = setup(; Ecut = Ecuts); 
_, basis_arr_ref = setup(; Ecut = 1.1 .* Ecuts); 
scfres_start = h1_riemannian_gradient(basis_arr[1]; tol = 1e-4 ,callback = (info -> nothing));
ψ_c = DFTK.select_occupied_orbitals(basis_arr[1], scfres_start.ψ, scfres_start.occupation).ψ;
ρ_c = scfres_start.ρ;

println("\nEARFMG")
cb_rfmg2 = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_rfmg2 = riemannian_full_multigrid(basis_arr, basis_arr_ref; 
 ψ = ψ_c, ρ = ρ_c, 
 tol, callback = cb_rfmg2,
 #callback_coarse = RcgDefaultCallback(),
 maxiter = 20,
 maxiter_inner = 7,
 #coarse_model_tol_outer = 0.0, 
 coarse_tol_outer = ReduceErrorRatioTolerance(0.2),
 refinement_tol = 0.2,
 #coarse_cond = ToleranceMinStepCoarseCondition(0.1),
 coarse_cond = EstimateErrorCoarseCondition(0.25),
 gradients = [EAGradient(basis) for basis = basis_arr], 
 #coarse_cost_residual = GreedyCostResidual,
 #only_rcg = true,
 #force_post_smoothing = false,
 basis_distance = 1,
 );
println(cb_rfmg2.times[end])

plot(1:length(cb_rfmg2.dists), cb_rfmg2.dists; yscale = :log10)

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
#fermialg = RCG_DFTK.FillLowestBands(),
callback = cb_nscf);
println(cb_nscf.times[end])


# basis_f = basis_arr[end]
# cb_ml = RfmgDefaultCallback(;basis_arr, ψ_ref)
# ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ_c)
# ρ1 = DFTK.compute_density(basis_f, ψ1, get_occ(basis_f))
# result = multilevel_riemannian_optimization(basis_arr; ψ = ψ1, ρ = ρ1, tol,
#     callback = cb_ml,
#     coarse_model_tol = 1e-2,
#     coarse_cond_tol = 0.45);

function get_occ(basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]
    return occupation
end

function naive_method(method)
    is_converged = NestedItConvergenceErrEstimate(
        EstimateNorm(0.2), tol)
    cb = (; times = [], dists = [], Ecuts = [])
    for (basis_int, basis_int_ref) = zip(basis_arr, basis_arr_ref)
        ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_int, ψ_c)
        ρ1 = DFTK.compute_density(basis_int, ψ1, get_occ(basis_int))

        RCG_DFTK.reset!(is_converged, basis_int_ref, true)
        result = method(basis_int; ρ = ρ1, ψ = ψ1, is_converged)

        ψ_int = DFTK.select_occupied_orbitals(basis_int, result.ψ, get_occ(basis_int)).ψ;
        ψ_fine = RCG_DFTK.interpolate_c2f(basis_int, basis_ref , ψ_int)
        push!(cb.times, result.runtime_ns/1e9)
        push!(cb.dists, RCG_DFTK.grassmann_dist_DFTK(basis_ref, ψ_fine, ψ_ref))
        push!(cb.Ecuts, basis_int.Ecut)
    end
    return cb
end

method(basis; kw...) = energy_adaptive_riemannian_conjugate_gradient(basis; kw...)
cb_naive_earcg = naive_method(method)

method(basis; kw...) = self_consistent_field(basis; kw...)
cb_naive_scf = naive_method(method)

# dist vs. time
plt = plot(; yscale = :log10, xscale = :log, ylabel = "dist", xlabel = "time (s)")
mkpath("experiments/data")
for (cb, name) = zip([cb_rfmg2 , cb_nearcg, cb_nscf,
    cb_naive_earcg, cb_naive_scf], 
            ["EARFMG", "nEARCG", "nSCF", "naive EARCG", "naive SCF"])
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