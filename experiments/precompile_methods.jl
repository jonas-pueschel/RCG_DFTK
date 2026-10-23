using DFTK
using RCG_DFTK
using PseudoPotentialData
include(joinpath(@__DIR__, "setups", "silicon_setup.jl"))

# names accepted by precompile_methods; groups expand to several methods
const PRECOMPILE_GROUPS = Dict(
    "rfmg"      => ["rfmg_ea", "rfmg_h1"],
    "nested_it" => ["nested_earcg", "nested_h1rcg", "nested_scf"],
    "single"    => ["earcg", "h1rcg", "h1rg", "scf"],
)
const PRECOMPILE_METHODS = ["rfmg_ea", "rfmg_h1", "rml",
                            "nested_earcg", "nested_h1rcg", "nested_scf",
                            "earcg", "h1rcg", "h1rg", "scf"]

"""
    precompile_methods(methods = PRECOMPILE_METHODS; verbose = false)

Runs each of the given methods once on a small silicon system (tolerance 1e-5), such that
the timings of later runs do not contain compilation time. `methods` contains method names
and/or groups:
- `"rfmg"` (= `"rfmg_ea"`, `"rfmg_h1"`): RFMG with EA and H1 gradients, Ecuts (10, 25, 40)
- `"rml"`: multilevel Riemannian optimization, Ecuts (10, 25, 40)
- `"nested_it"` (= `"nested_earcg"`, `"nested_h1rcg"`, `"nested_scf"`): Ecuts (10, 25, 40)
- `"single"` (= `"earcg"`, `"h1rcg"`, `"h1rg"`, `"scf"`): single level, Ecut 25
All runs use `RfmgDefaultCallback` (including the distance column), such that its printing
code is compiled as well; output and warnings are suppressed unless `verbose = true`.

Example: `precompile_methods(["rfmg", "nested_it"])`
"""
function precompile_methods(methods = PRECOMPILE_METHODS; verbose = false)
    selected = unique(vcat([get(PRECOMPILE_GROUPS, m, [m]) for m in methods]...))
    unknown = setdiff(selected, PRECOMPILE_METHODS)
    isempty(unknown) || throw(ArgumentError("unknown methods $(unknown), available: " *
        "$(PRECOMPILE_METHODS) and groups $(collect(keys(PRECOMPILE_GROUPS)))"))
    println("run each method once to ensure they are precompiled: $(join(selected, ", "))")

    tol = 1e-5
    Ecuts = [10, 25, 40]
    _, basis_arr = silicon_setup(; Ecut = Ecuts)
    _, basis_arr_ref = silicon_setup(; Ecut = 1.1 .* Ecuts)
    basis_c = basis_arr[1]
    basis_single = basis_arr[2]  # Ecut 25
    basis_fine = basis_arr[end]

    # initial value on the coarsest basis
    scfres_start = self_consistent_field(basis_c; tol = 1e-2, callback = info -> nothing)
    ψ_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ
    ρ_c = scfres_start.ρ
    occupation = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).occupation

    # prolongated initial values for the methods not starting on the coarsest basis
    function prolongated_guess(basis)
        ψ = RCG_DFTK.prolongate_point(basis_c, basis, ψ_c)
        ψ, DFTK.compute_density(basis, ψ, occupation)
    end
    ψ_single, ρ_single = prolongated_guess(basis_single)
    ψ_fine, ρ_fine = prolongated_guess(basis_fine)

    # any point on the finest basis serves as reference, it only has to exercise the distance code
    callback() = RfmgDefaultCallback(; basis_ref = basis_fine, ψ_ref = ψ_fine)

    runs = Dict(
        "rfmg_ea" => () -> riemannian_full_multigrid(basis_arr, basis_arr_ref;
            ψ = ψ_c, ρ = ρ_c, tol, callback = callback(),
            gradients = [EAGradient(basis) for basis in basis_arr]),
        "rfmg_h1" => () -> riemannian_full_multigrid(basis_arr, basis_arr_ref;
            ψ = ψ_c, ρ = ρ_c, tol, callback = callback(),
            gradients = [H1Gradient(basis) for basis in basis_arr]),
        "rml" => () -> multilevel_riemannian_optimization(basis_arr;
            ψ = ψ_fine, ρ = ρ_fine, tol, callback = callback()),
        "nested_earcg" => () -> RCG_DFTK.nested_earcg(basis_arr, basis_arr_ref;
            ψ = ψ_c, ρ = ρ_c, tol, callback = callback()),
        "nested_h1rcg" => () -> RCG_DFTK.nested_h1rcg(basis_arr, basis_arr_ref;
            ψ = ψ_c, ρ = ρ_c, tol, callback = callback()),
        "nested_scf" => () -> RCG_DFTK.nested_scf(basis_arr, basis_arr_ref;
            ψ = ψ_c, ρ = ρ_c, tol, callback = callback()),
        "earcg" => () -> energy_adaptive_riemannian_conjugate_gradient(basis_single;
            ψ = ψ_single, ρ = ρ_single, tol, callback = callback()),
        "h1rcg" => () -> RCG_DFTK.h1_riemannian_conjugate_gradient(basis_single;
            ψ = ψ_single, ρ = ρ_single, tol, callback = callback()),
        "h1rg" => () -> RCG_DFTK.h1_riemannian_gradient(basis_single;
            ψ = ψ_single, ρ = ρ_single, tol, callback = callback()),
        "scf" => () -> self_consistent_field(basis_single;
            ψ = ψ_single, ρ = ρ_single, tol, callback = callback()),
    )

    for m in selected
        verbose && println("\n$m")
        # with tol = 1e-5 on these Ecuts, the multilevel methods stop at the discretization error
        # and warn ("cannot refine anymore", "not converged"): only shown if verbose
        quiet_run() = Base.CoreLogging.with_logger(runs[m], Base.CoreLogging.NullLogger())
        t = @elapsed (verbose ? runs[m]() : redirect_stdout(quiet_run, devnull))
        println("  $m: $(round(t; digits = 1)) s")
    end
    println("done.")
end
