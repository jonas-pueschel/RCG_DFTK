using DFTK
using PseudoPotentialData
include(joinpath(@__DIR__, "ecuts.jl"))

"""
    setup_model(lattice, atoms, positions; model_terms = :lda)

Model with the pseudopotential and, depending on `model_terms`, the nonlinear terms:
`:lda` (Hartree + LDA XC), `:pbe` (Hartree + PBE XC), `:hartree` (Hartree only), `:xc` (LDA XC
only) or `:linear` (none, i.e. a linear eigenvalue problem).
"""
function setup_model(lattice, atoms, positions; model_terms = :lda)
    model_terms == :lda && return model_LDA(lattice, atoms, positions)
    model_terms == :pbe && return model_DFT(lattice, atoms, positions; functionals = PBE())
    extra_terms = Dict(:linear => [], :hartree => [Hartree()], :xc => [LDA()])
    haskey(extra_terms, model_terms) || throw(ArgumentError("unknown model_terms $model_terms"))
    model_atomic(lattice, atoms, positions; extra_terms = extra_terms[model_terms])
end

"""
    psp_family(core, model_terms)

GTH pseudopotential family with the given `core` size ("largecore", "semicore", ...),
generated with PBE for `model_terms = :pbe` and with LDA otherwise.
"""
function psp_family(core, model_terms)
    functional = model_terms == :pbe ? "pbe" : "lda"
    PseudoFamily("cp2k.nc.sr.$functional.v0_1.$core.gth")
end

"""
    supercell_fft_size(model_prim, Ecut, supercell_size; supersampling = 2)

FFT size for a `supercell_size` supercell of `model_prim`: a size sufficient for the primitive
cell (containing the denominators of its fractional translations, as DFTK's default does),
times `supercell_size`. With independently chosen sizes per direction, the ratio of the
supercell can get lost (e.g. (72, 40, 40) for a 2×1×1 silicon supercell). Then DFTK drops the
symmetries relating the supercell axes, which changes the irreducible k-points between bases.
"""
function supercell_fft_size(model_prim, Ecut, supercell_size; supersampling = 2)
    denominators = [denominator(rationalize(sym.w[i]; tol = DFTK.SYMMETRY_TOLERANCE))
                    for sym in model_prim.symmetries for i = 1:3]
    factors = Tuple(intersect((2, 3, 4, 6), denominators))
    fft_size_prim = compute_fft_size(model_prim, Ecut; supersampling, factors)
    return Tuple(supercell_size .* fft_size_prim)
end

"""
    setup_bases(model_prim, model, Ecut, kgrid, supercell_size; supersampling = 2)

Returns `[model, basis]` for a number `Ecut` and `[model, basis_arr]` for an array of cutoffs,
with FFT sizes from `supercell_fft_size`. Warns if a basis does not keep all symmetries.
"""
function setup_bases(model_prim, model, Ecut, kgrid, supercell_size; supersampling = 2)
    function make_basis(e)
        fft_size = supercell_fft_size(model_prim, e, supercell_size; supersampling)
        basis = PlaneWaveBasis(model; Ecut = e, kgrid, fft_size, symmetries_respect_rgrid = true)
        length(basis.symmetries) == length(model.symmetries) ||
            @warn "Ecut = $e: the basis keeps only $(length(basis.symmetries)) of $(length(model.symmetries)) symmetries"
        basis
    end
    if isa(Ecut, Number)
        return [model, make_basis(Ecut)]
    elseif isa(Ecut, Array)
        return [model, [make_basis(e) for e in Ecut]]
    else
        throw("invalid type of Ecut")
    end
end
