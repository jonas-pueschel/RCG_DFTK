"""
    nested_iteration(method, basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
                     refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                     ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback(),
                     callback_inner = info -> nothing)

Solve successively on the bases in `basis_arr` (ordered from coarse to fine). On level `l`,
`method(basis_arr[l]; ρ, ψ, is_converged, callback)` is called, where `is_converged` is a
`NestedItConvergenceErrEstimate` using the reference basis `basis_arr_ref[l]`: as in RFMG,
a level is left once `refinement_cond` decides that the coarse contribution to the estimated
error is small compared to the whole error (by default below `refinement_tol` times the whole
error in the L2 norm, see `AbstractRefinementCondition`; checked in every `check_every`-th
iteration). The resulting `ψ` is prolongated onto the next basis and, together with its
density, used as initial guess there. The whole algorithm terminates once the estimated error is below `tol`,
or on the last level once refinement would be necessary or the coarse contribution to the
error is below `1e-2 tol` (as in RFMG; then it is not reported as converged).
`method` has to return an info object containing `ψ`, `occupation` and `n_iter`.
`callback` is called once per level, after the level is finished, with the final info of the
level as `stage = :iterate` (`n_iter` is the level, `n_iter_total` the iterations of all levels
so far; e.g. for `RfmgDefaultCallback`, one line per Ecut), and once more at the end with
`stage = :finalize` and `converged` the convergence of the whole nested iteration.
`callback_inner` is passed on to `method`, with `n_iter` counted across all levels.
The optional initial guesses `ρ` and `ψ` live in the coarsest basis `basis_arr[1]`.
"""
function nested_iteration(method, basis_arr::Vector, basis_arr_ref::Vector;
                          tol = 1e-6, refinement_tol = 0.45,
                          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                          ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback(),
                          callback_inner = info -> nothing)
    @assert length(basis_arr) == length(basis_arr_ref)
    for l = 1:(length(basis_arr) - 1)
        @assert basis_arr[l].Ecut < basis_arr[l + 1].Ecut
    end

    start_ns = time_ns()
    is_converged = NestedItConvergenceErrEstimate(refinement_cond, tol; check_every)
    info = nothing
    n_iter_prev = 0  # iterations performed on the previous levels
    for (l, (basis, basis_ref)) in enumerate(zip(basis_arr, basis_arr_ref))
        if l > 1
            # extrapolate the previous solution onto the current basis
            ψ = prolongate_point(basis_arr[l - 1], basis, info.ψ)
            ρ = DFTK.compute_density(basis, ψ, info.occupation)
        end
        is_last_level = l == length(basis_arr)
        reset!(is_converged, basis_ref, is_last_level)

        # only pass initial guesses if available, s.t. the method's defaults are used otherwise
        guess = (; (isnothing(ρ) ? (;) : (; ρ))..., (isnothing(ψ) ? (;) : (; ψ))...)
        # continue the iteration count of the previous levels
        callback_level = info_inner -> callback_inner(merge(info_inner, (; n_iter = info_inner.n_iter + n_iter_prev)))
        info = method(basis; guess..., is_converged, callback = callback_level)

        # converged: the final convergence goal was met; the level infos carry the convergence
        # of the level, the returned info the one of the whole method
        (; final_converged, refine, est_err, est_err_norm, est_err_perc) = is_converged
        is_last_level && refine && !final_converged && @warn "cannot refine anymore"
        done = final_converged || is_last_level
        n_iter_prev += info.n_iter
        info = merge(info, (; n_iter = n_iter_prev, level = l, n_levels = length(basis_arr), basis_ref,
                              converged = done ? final_converged : info.converged, final_converged,
                              est_err, est_err_norm, est_err_perc, runtime_nested_ns = time_ns() - start_ns))

        # one callback per level: the info of the method is its final one (stage = :finalize,
        # converged = convergence of the level), which would not be printed as an iteration
        callback(merge(info, (; stage = :iterate, converged = false, n_iter = l, n_iter_total = n_iter_prev,
                                ρin = nothing, ρout = haskey(info, :ρ) ? info.ρ : nothing,
                                τ = nothing, start_ns)))
        done && break
    end
    info = merge(info, (; runtime_ns = time_ns() - start_ns, start_ns))
    callback(info)
    info
end

"""
    nested_earcg(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback(),
          callback_inner = info -> nothing, kwargs...)

Nested iteration with `energy_adaptive_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_earcg(basis_arr::Vector, basis_arr_ref::Vector;
                      tol = 1e-6, refinement_tol = 0.45,
                      refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                      callback = RfmgDefaultCallback(), callback_inner = info -> nothing,
                      ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = energy_adaptive_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ,
                     callback, callback_inner)
end

"""
    nested_h1rcg(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback(),
          callback_inner = info -> nothing, kwargs...)

Nested iteration with `h1_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_h1rcg(basis_arr::Vector, basis_arr_ref::Vector;
                      tol = 1e-6, refinement_tol = 0.45,
                      refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                      callback = RfmgDefaultCallback(), callback_inner = info -> nothing,
                      ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = h1_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ,
                     callback, callback_inner)
end

"""
    nested_scf(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback(),
          callback_inner = info -> nothing, kwargs...)

Nested iteration with DFTK's `self_consistent_field` on each level.
Additional `kwargs` are passed to the solver, e.g. `fermialg = FillLowestBands()` to use the
occupations of the RCG methods instead of DFTK's default (see `FillLowestBands`).
"""
function nested_scf(basis_arr::Vector, basis_arr_ref::Vector;
                    tol = 1e-6, refinement_tol = 0.45,
                    refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                    callback = RfmgDefaultCallback(), callback_inner = info -> nothing,
                    ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = self_consistent_field(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ,
                     callback, callback_inner)
end

"""
    FillLowestBands()

Fermi level algorithm for DFTK's `self_consistent_field` (keyword `fermialg`) with the
occupations of the RCG methods: at every k-point the lowest `n_occ` bands are fully occupied,
without a common Fermi level. Wherever the eigenvalues have a gap between the occupied and the
unoccupied states (over all k-points), this coincides with DFTK's zero temperature default.
Without such a gap (e.g. on coarse, unresolved levels), DFTK's default aborts ("Unable to find
non-fractional occupations ..."), while this keeps filling the lowest bands. The returned `εF`
is only informative (midpoint between highest occupied and lowest unoccupied eigenvalue).
Only for zero temperature.
"""
struct FillLowestBands <: DFTK.AbstractFermiAlgorithm end


function DFTK.compute_occupation(basis::PlaneWaveBasis{T}, eigenvalues::AbstractVector,
                                 ::FillLowestBands; kwargs...) where {T}
    model = basis.model
    @assert iszero(model.temperature) "FillLowestBands is only meant for zero temperature"
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    if model.n_electrons % (n_spin * filled_occ) != 0
        error("$(model.n_electrons) electrons cannot be attained by filling states with " *
              "occupation $filled_occ.")
    end
    n_occ = div(model.n_electrons, n_spin * filled_occ)
    all(length(εk) >= n_occ for εk in eigenvalues) ||
        error("Could not obtain required number of electrons by filling every state. " *
              "Increase n_bands.")

    occupation = map(eigenvalues) do εk
        occ = T[n <= n_occ ? filled_occ : zero(T) for n in eachindex(εk)]
        DFTK.to_device(basis.architecture, occ)
    end
    ε_homo = maximum(εk[n_occ] for εk in eigenvalues)
    ε_lumo = all(length(εk) > n_occ for εk in eigenvalues) ?
             minimum(εk[n_occ + 1] for εk in eigenvalues) : ε_homo
    (; occupation, εF = (ε_homo + ε_lumo) / 2)
end
