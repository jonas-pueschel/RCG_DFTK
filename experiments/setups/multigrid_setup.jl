using DFTK
using RCG_DFTK
using BSON
include(joinpath(@__DIR__, "silicon_setup.jl"))
include(joinpath(@__DIR__, "GaAs_setup.jl"))
include(joinpath(@__DIR__, "TiO2_setup.jl"))

const REFERENCE_SOLUTION_DIR = joinpath(@__DIR__, "..", "reference_solutions")

# cutoff hierarchy and reference cutoff of each model
const MULTIGRID_MODELS = Dict(
    "TiO2" => (;
        setup = (; Ecut) -> TiO2_setup(; Ecut, kgrid = [2, 2, 2], model_terms = :pbe),
        Ecuts = [50.0, 127.3, 240.0, 600.0], #[50.0, 84.0, 127.3, 179.2, 240.0, 600.0],
        Ecut_ref = 800.0,
    ),
    "GaAs" => (;
        setup = (; Ecut) -> GaAs_setup(; Ecut, kgrid = [2, 2, 2], model_terms = :lda),
        Ecuts = [10.0, 21.0, 38.0, 58.8, 84.8, 117.5, 202.8, 380.0, 700.0],
        Ecut_ref = 1000.0,
    ),
    "silicon" => (;
        setup = (; Ecut) -> silicon_setup(; Ecut, kgrid = [4, 4, 4], model_terms = :lda),
        Ecuts = [10.0, 21.0, 38.0, 58.8, 84.8, 117.5, 202.8, 380.0, 700.0],
        Ecut_ref = 1000.0,
    ),
)

"""
    multigrid_setup(model_name; tol_init = 1e-4, ref_factor = 1.1, recompute_reference = false)

Multigrid environment of one of the models in `MULTIGRID_MODELS` ("TiO2", "GaAs", "silicon"):
- `model`, `setup` (the model's `setup(; Ecut)` function), `Ecuts`, `Ecuts_ref`,
  `basis_arr` and `basis_arr_ref` (reference cutoffs `ref_factor * Ecuts`),
- the initial value `ψ_c`, `ρ_c` on `basis_arr[1]`, computed by H1-RCG with tolerance `tol_init`,
- the reference solution `ψ_ref`, `ρ_ref` on `basis_ref` (cutoff `Ecut_ref`), see
  `reference_solution`.
"""
function multigrid_setup(model_name; tol_init = 1e-4, ref_factor = 1.1, recompute_reference = false)
    haskey(MULTIGRID_MODELS, model_name) ||
        throw(ArgumentError("unknown model $model_name, available: $(keys(MULTIGRID_MODELS))"))
    (; setup, Ecuts, Ecut_ref) = MULTIGRID_MODELS[model_name]

    Ecuts_ref = get_Ecuts_ref(Ecuts; mode = :factor, factor = ref_factor)
    model, basis_arr = setup(; Ecut = Ecuts)
    _, basis_arr_ref = setup(; Ecut = Ecuts_ref)

    println("Initial value on Ecut = $(Ecuts[1])")
    ψ_c, ρ_c = initial_value(basis_arr[1]; tol = tol_init)

    ψ_ref, ρ_ref, basis_ref = reference_solution(model_name, setup, Ecuts, Ecut_ref, ψ_c, ρ_c;
                                                 ref_factor, recompute = recompute_reference)

    return (; model_name, model, setup, Ecuts, Ecuts_ref, basis_arr, basis_arr_ref,
              ψ_c, ρ_c, ψ_ref, ρ_ref, basis_ref, Ecut_ref)
end

function initial_value(basis; tol = 1e-4)
    scfres = h1_riemannian_conjugate_gradient(basis; tol, callback = (info) -> nothing)
    ψ = DFTK.select_occupied_orbitals(basis, scfres.ψ, scfres.occupation).ψ
    return ψ, scfres.ρ
end

"""
    reference_solution(model_name, setup, Ecuts, Ecut_ref, ψ_c, ρ_c; ref_factor = 1.1,
                       refinement_tol = 0.05, recompute = false)

Reference solution on the cutoff `Ecut_ref`, loaded from
`reference_solutions/<model_name>_ref.BSON` if that file exists with the same `Ecut_ref`.
Otherwise it is computed and saved: RFMG on `[Ecuts..., Ecut_ref]`, starting from `ψ_c`, `ρ_c`,
runs until the refinement condition (`refinement_tol`) asks for a refinement beyond `Ecut_ref`.
EARCG then reduces the residual norm of this intermediate solution by a factor `1e-3`.
"""
function reference_solution(model_name, setup, Ecuts, Ecut_ref, ψ_c, ρ_c; ref_factor = 1.1,
                            refinement_tol = 0.05, recompute = false)
    file = joinpath(REFERENCE_SOLUTION_DIR, "$(model_name)_ref.BSON")
    _, basis_ref = setup(; Ecut = Ecut_ref)

    if isfile(file) && !recompute
        data = BSON.load(file)
        if data[:Ecut_ref] == Ecut_ref
            println("Loaded reference solution from $file")
            return data[:ψ_ref], data[:ρ_ref], basis_ref
        end
        @warn "$file has Ecut_ref = $(data[:Ecut_ref]) instead of $Ecut_ref, recomputing"
    end

    Ecuts_mg = [Ecuts..., Ecut_ref]
    _, basis_arr = setup(; Ecut = Ecuts_mg)
    _, basis_arr_ref = setup(; Ecut = ref_factor .* Ecuts_mg)

    println("\nReference: RFMG up to Ecut = $Ecut_ref")
    rfmg = riemannian_full_multigrid(basis_arr, basis_arr_ref;
        ψ = ψ_c, ρ = ρ_c,
        tol = 0.0,  # stop only when the refinement condition asks for a cutoff beyond Ecut_ref
        maxiter = 100,
        maxiter_inner = 7,
        refinement_tol,
        coarse_tol_outer = ReduceErrorRatioTolerance(0.1),
        coarse_cond = EstimateErrorCoarseCondition(0.25) & ResidualNormCoarseCondition(10.0) &
                      MinStepsCoarseCondition(1),
        gradients = [EAGradient(basis) for basis in basis_arr],
        terminate_on_non_descent = true,
    )
    rfmg.basis.Ecut == Ecut_ref ||
        @warn "RFMG stopped on Ecut = $(rfmg.basis.Ecut) instead of Ecut_ref = $Ecut_ref"
    println("Intermediate reference solution in $(rfmg.runtime_ns / 1e9) seconds")

    # the intermediate solution lives on rfmg.basis, the RFMG level with cutoff Ecut_ref
    norm_res = residual_norm(rfmg.basis, rfmg.ψ)
    tol = 1e-3 * norm_res
    println("\nReference: EARCG from residual norm $norm_res to $tol")
    rcg = energy_adaptive_riemannian_conjugate_gradient(rfmg.basis; ψ = rfmg.ψ, ρ = rfmg.ρ, tol)
    println("Refined reference solution in $(rcg.runtime_ns / 1e9) seconds")

    ψ_ref, ρ_ref = rcg.ψ, rcg.ρ
    mkpath(REFERENCE_SOLUTION_DIR)
    BSON.bson(file; ψ_ref, ρ_ref, Ecut_ref, Ecuts)
    println("Saved reference solution to $file")
    return ψ_ref, ρ_ref, basis_ref
end

function residual_norm(basis, ψ)
    filled_occ = DFTK.filled_occupation(basis.model)
    occupation = [filled_occ * ones(size(ψk, 2)) for ψk in ψ]
    ρ = DFTK.compute_density(basis, ψ, occupation)
    _, H = energy_hamiltonian(basis, ψ, occupation; ρ)
    _, _, res = RCG_DFTK.compute_residual(H, ψ)
    return RCG_DFTK.norm_DFTK(basis, res)
end
