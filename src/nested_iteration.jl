"""
    nested_iteration(method, basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
                     refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                     ρ = nothing, ψ = nothing, callback = info -> nothing)

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
`callback` is passed on to `method` (with `n_iter` counted across all levels, as in the
multilevel methods, e.g. for `RfmgDefaultCallback`), and called once more with the final
info of every level (`stage = :finalize`).
The optional initial guesses `ρ` and `ψ` live in the coarsest basis `basis_arr[1]`.
"""
function nested_iteration(method, basis_arr::Vector, basis_arr_ref::Vector;
                          tol = 1e-6, refinement_tol = 0.45,
                          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                          ρ = nothing, ψ = nothing, callback = RfmgDefaultCallback())
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
        # continue the iteration count of the previous levels; the final info of each
        # level is passed to the callback below
        callback_level = function (info_inner)
            info_inner.stage == :finalize && return info_inner
            callback(merge(info_inner, (; n_iter = info_inner.n_iter + n_iter_prev)))
        end
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
        callback(info)
        done && break
    end
    info
end

"""
    nested_earcg(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with `energy_adaptive_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_earcg(basis_arr::Vector, basis_arr_ref::Vector;
                      tol = 1e-6, refinement_tol = 0.45,
                      refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                      callback = RfmgDefaultCallback(),
                      ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = energy_adaptive_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ, callback)
end

"""
    nested_h1rcg(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with `h1_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_h1rcg(basis_arr::Vector, basis_arr_ref::Vector;
                      tol = 1e-6, refinement_tol = 0.45,
                      refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                      callback = RfmgDefaultCallback(),
                      ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = h1_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ, callback)
end

"""
    nested_scf(basis_arr, basis_arr_ref; tol = 1e-6, refinement_tol = 0.45,
          refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
          ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with DFTK's `self_consistent_field` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_scf(basis_arr::Vector, basis_arr_ref::Vector;
                    tol = 1e-6, refinement_tol = 0.45,
                    refinement_cond = EstimateL2Norm(refinement_tol), check_every = 1,
                    callback = RfmgDefaultCallback(),
                    ρ = nothing, ψ = nothing, kwargs...)
    method(basis; kw...) = self_consistent_field(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, basis_arr_ref; tol, refinement_cond, check_every, ρ, ψ, callback)
end
