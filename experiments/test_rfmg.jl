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

basis_c = basis_arr[1]

# Convergence tolerance
tol = 1.0e-9;

# refinement tolerance of all methods
refinement_tol = 0.05

# A: adaptive coarse corrections, V: fixed V(1,1)-cycles, one per outer iteration
rfmg_variants = Dict(
    "A" => (;
        maxiter = 20,
        maxiter_inner = 7,
        coarse_tol_outer = ReduceErrorRatioTolerance(0.05),
        coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(5.0) &
                      MinStepsCoarseCondition(1),
    ),
    "V" => (;
        maxiter = 100,
        maxiter_inner = 3,  # = length of one V(1,1)-cycle
        is_converged_outer = (tol) -> (info -> false),
        cycle = VCycle(1, 1),
    ),
)

# Ecut of the basis of every gradient calculation of each method (if the method records it)
levels_of = Dict{String, Vector{Float64}}()
get_levels(result) = haskey(result, :levels) ? Float64.(result.levels) : Float64[]

function run_rfmg(name, gradient, variant)
    println("\n$name")
    cb = RfmgDefaultCallback(; basis_ref, ψ_ref)
    result = riemannian_full_multigrid(basis_arr, basis_arr_ref;
        ψ = ψ_c, ρ = ρ_c, tol, callback = cb, refinement_tol,
        gradients = [gradient(basis) for basis in basis_arr],
        basis_distance = 1,
        rfmg_variants[variant]...)
    println(cb.times[end])
    levels_of[name] = get_levels(result)
    return cb
end

cb_earfmg_a = run_rfmg("EARFMG-A", EAGradient, "A")
cb_earfmg_v = run_rfmg("EARFMG-V", EAGradient, "V")
cb_h1rfmg_a = run_rfmg("H1RFMG-A", H1Gradient, "A")
cb_h1rfmg_v = run_rfmg("H1RFMG-V", H1Gradient, "V")

# nested EARCG
println("\nnested EARCG")
cb_nearcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nearcg = nested_earcg(basis_arr, basis_arr_ref;
    tol = 1e-8, refinement_tol,
    check_every = 1, ψ = ψ_c, ρ = ρ_c, callback = cb_nearcg);
println(cb_nearcg.times[end])
levels_of["nEARCG"] = get_levels(res_nearcg)

# nested H1RCG
println("\nnested H1RCG")
cb_nh1rcg = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nh1rcg = nested_h1rcg(basis_arr, basis_arr_ref;
    tol = 1e-8, refinement_tol,
    check_every = 1, ψ = ψ_c, ρ = ρ_c, callback = cb_nh1rcg);
println(cb_nh1rcg.times[end])
levels_of["nH1RCG"] = get_levels(res_nh1rcg)

# nested SCF
println("\nnested SCF")
cb_nscf = RfmgDefaultCallback(;basis_ref, ψ_ref)
res_nscf = nested_scf(basis_arr, basis_arr_ref;
    tol = 1e-8, refinement_tol, check_every = 1,
    ψ = ψ_c, ρ = ρ_c,
    #fermialg = RCG_DFTK.FillLowestBands(),
    callback = cb_nscf);
println(cb_nscf.times[end])
levels_of["nSCF"] = get_levels(res_nscf)


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
        EstimateNorm(refinement_tol), tol)
    cb = (; times = [], dists = [], Ecuts = [], levels = Float64[])
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
        append!(cb.levels, get_levels(result))
    end
    return cb
end

method(basis; kw...) = energy_adaptive_riemannian_conjugate_gradient(basis; kw...)
cb_naive_earcg = naive_method(method)
levels_of["naive EARCG"] = cb_naive_earcg.levels

method(basis; kw...) = self_consistent_field(basis; kw...)
cb_naive_scf = naive_method(method)
levels_of["naive SCF"] = cb_naive_scf.levels

# dist vs. time
plt = plot(; yscale = :log10, xscale = :log, ylabel = "dist", xlabel = "time (s)")
mkpath("experiments/data")
for (cb, name) = zip([cb_earfmg_a, cb_earfmg_v, cb_h1rfmg_a, cb_h1rfmg_v,
                      cb_nearcg, cb_nh1rcg, cb_nscf, cb_naive_earcg, cb_naive_scf],
                     ["EARFMG-A", "EARFMG-V", "H1RFMG-A", "H1RFMG-V",
                      "nEARCG", "nH1RCG", "nSCF", "naive EARCG", "naive SCF"])
    plot!(plt, cb.times, max.(cb.dists), label = name, marker = :circle, markersize = 2)
    if @isdefined(model_name) && model_name != ""
        open("experiments/data/rfmg_$(model_name)_$(name)_dist_time.dat", "w") do io
            # levels has one entry per gradient calculation, not per iteration: the shorter
            # columns are padded with NaN
            cols = [Float64.(cb.times), Float64.(cb.dists), Float64.(cb.Ecuts), levels_of[name]]
            n_rows = maximum(length, cols)
            pad(c) = [c; fill(NaN, n_rows - length(c))]
            println(io, "x y Ecut level")
            writedlm(io, reduce(hcat, pad.(cols)), ' ')
        end
    end
end
display(plt)

#allow_sleep() 