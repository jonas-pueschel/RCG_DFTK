"""
    nested_iteration(method, basis_arr, tols; ρ = nothing, ψ = nothing, callback = info -> nothing)

Solve successively on the bases in `basis_arr` (ordered from coarse to fine). On level `l`,
`method(basis_arr[l]; ρ, ψ, tol = tols[l], callback)` is called; the resulting `ψ` is
prolongated onto the next basis and, together with its density, used as initial guess there.
`method` has to return an info object containing `ψ`, `occupation` and `n_iter`.
`callback` is passed on to `method` (with `n_iter` counted across all levels, as in the
multilevel methods, e.g. for `RfmgDefaultCallback`), and called once more with the final
info of every level (`stage = :finalize`).
The optional initial guesses `ρ` and `ψ` live in the coarsest basis `basis_arr[1]`.
"""
function nested_iteration(method, basis_arr::Vector, tols::AbstractVector;
                          ρ = nothing, ψ = nothing, callback = info -> nothing)
    @assert length(basis_arr) == length(tols)
    for l = 1:(length(basis_arr) - 1)
        @assert basis_arr[l].Ecut < basis_arr[l + 1].Ecut
    end

    start_ns = time_ns()
    info = nothing
    n_iter_prev = 0  # iterations performed on the previous levels
    for (l, (basis, tol)) in enumerate(zip(basis_arr, tols))
        if l > 1
            # extrapolate the previous solution onto the current basis
            ψ = prolongate_point(basis_arr[l - 1], basis, info.ψ)
            ρ = DFTK.compute_density(basis, ψ, info.occupation)
        end

        # only pass initial guesses if available, s.t. the method's defaults are used otherwise
        guess = (; (isnothing(ρ) ? (;) : (; ρ))..., (isnothing(ψ) ? (;) : (; ψ))...)
        # continue the iteration count of the previous levels; the final info of each
        # level is passed to the callback below
        callback_level = function (info_inner)
            info_inner.stage == :finalize && return info_inner
            callback(merge(info_inner, (; n_iter = info_inner.n_iter + n_iter_prev)))
        end
        info = method(basis; guess..., tol, callback = callback_level)

        n_iter_prev += info.n_iter
        info = merge(info, (; n_iter = n_iter_prev, level = l, n_levels = length(basis_arr),
                              runtime_nested_ns = time_ns() - start_ns))
        callback(info)
    end
    info
end

"""
    nested_earcg(basis_arr, tols; ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with `energy_adaptive_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_earcg(basis_arr::Vector, tols::AbstractVector;
                      ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)
    method(basis; kw...) = energy_adaptive_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, tols; ρ, ψ, callback)
end

"""
    nested_h1rcg(basis_arr, tols; ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with `h1_riemannian_conjugate_gradient` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_h1rcg(basis_arr::Vector, tols::AbstractVector;
                      ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)
    method(basis; kw...) = h1_riemannian_conjugate_gradient(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, tols; ρ, ψ, callback)
end

"""
    nested_scf(basis_arr, tols; ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)

Nested iteration with DFTK's `self_consistent_field` on each level.
Additional `kwargs` are passed to the solver.
"""
function nested_scf(basis_arr::Vector, tols::AbstractVector;
                    ρ = nothing, ψ = nothing, callback = info -> nothing, kwargs...)
    method(basis; kw...) = self_consistent_field(basis; kwargs..., kw...)
    nested_iteration(method, basis_arr, tols; ρ, ψ, callback)
end
