# Parameters and helpers for the riemannian full multigrid (RFMG)

# Prolongates ψ_c and evaluates density, Hamiltonian, energies and residual on the fine grid.
function evaluate_prolongated(basis_c, basis_f, occupation, ψ_c)
    ψ_f = prolongate_point(basis_c, basis_f, ψ_c)
    ρ_f = DFTK.compute_density(basis_f, ψ_f, occupation)
    energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)
    Hψ, Λ, res, cost = calculate_cost_residual(H, ψ_f, energies.total, basis_f, StandardCostResiudal())
    return (; ψ_f, ρ_f, energies, cost, H, Hψ, Λ, res)
end

# Analogue of get_next_rcg for a step along a prolongated coarse direction η_f = P η_c.
# Since retraction and prolongation commute, the step is done on the coarse grid and
# then prolongated; energy and residual are evaluated on the fine grid. The returned
# named tuple contains the fields used by do_step, and additionally the coarse point ψ_c_next.
function get_next_prolongated(basis_c, basis_f, occupation, ψ_c, η_c, τ,
                              retraction::AbstractRetraction, transport::AbstractTransport)
    ψ_c_next = calculate_retraction(ψ_c, η_c, τ, retraction)
    (; ψ_f, ρ_f, energies, cost, H, Hψ, Λ, res) = evaluate_prolongated(basis_c, basis_f, occupation, ψ_c_next)

    # transport of the direction (needed for the slope), again on the coarse grid
    Tη_c = calculate_transport(ψ_c_next, η_c, η_c, τ, ψ_c, transport, retraction; is_prev_dir = true)
    Tη_next = prolongate_vector(basis_c, basis_f, ψ_c_next, ψ_f, Tη_c, ProjectionMap())

    return (; ψ_c_next, ψ_next = ψ_f, ρ_next = ρ_f, energies_next = energies, cost_next = cost,
              H_next = H, Hψ_next = Hψ, Λ_next = Λ, res_next = res, Tη_next, τ)
end

# GreedySecantStrategy that does not re-use the previous step size, i.e. every step
# starts with the initial guess τ_0 (e.g. ConstantStep(1.0))
struct RestartedGreedySecantStrategy <: IterationStrategy
    τ_0::AbstractStepSize
    c2::Real
end
function do_step(basis, ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, cost, get_next,
                 backtracking::RestartedGreedySecantStrategy)
    strat = GreedySecantStrategy(backtracking.τ_0, backtracking.c2)
    return do_step(basis, ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, cost, get_next, strat)
end

"""
Decides, analogously to `is_converged`, whether the error is dominated by the discretization,
i.e. whether to refine: `refinement_cond(info)` with `info` containing the error estimate
`est_err` on `basis_ref` and its restriction `est_err_coarse` on `basis`. Subtypes have a
field `refinement_tol` and define the norm the estimate is measured in via `error_norm`; the
same norm is used for the convergence check against `tol`.
"""
abstract type AbstractRefinementCondition end

function estimate_error_norms(rc::AbstractRefinementCondition, info)
    err = error_norm(rc, info.basis_ref, info.est_err)
    err_coarse = error_norm(rc, info.basis, info.est_err_coarse)
    return (; err, err_coarse, perc = err_coarse / err)
end

# refine if the coarse contribution to the error is small compared to the whole error
(rc::AbstractRefinementCondition)(info) = estimate_error_norms(rc, info).perc < rc.refinement_tol

struct EstimateL2Norm <: AbstractRefinementCondition
    refinement_tol
end
error_norm(::EstimateL2Norm, basis, e) = norm_DFTK(basis, e)

struct EstimateH1Norm <: AbstractRefinementCondition
    refinement_tol
end
error_norm(::EstimateH1Norm, basis, e) = h1_norm_DFTK(basis, e)

struct AlwaysRefine <: AbstractRefinementCondition end
error_norm(::AlwaysRefine, basis, e) = norm_DFTK(basis, e)
(rc::AlwaysRefine)(info) = true


"""
    NestedItConvergenceErrEstimate(refinement_cond, tol; check_every = 1)

Convergence criterion of a single level of `nested_iteration`, using the same error estimator
as RFMG: the occupied orbitals of `info.ψ` are prolongated onto the reference basis `basis_ref`
of the current level, where the residual is evaluated and the refinement metric is inverted.
The level is converged if
- the whole estimated error is below `tol` (then also `final_converged` is set), or
- `refinement_cond` (an `AbstractRefinementCondition`, which also defines the norm of the
  estimate) asks for refinement; on the last level this terminates the whole nested
  iteration, as refining is not possible anymore (as in RFMG), or
- (on the last level) the coarse contribution is below `1e-2 tol` (as in RFMG).
`refine` tells whether the refinement condition triggered at the last estimate.
As the estimate needs a Hamiltonian on the reference basis, it is only computed in every
`check_every`-th call. `basis_ref` and `is_last_level` are set by `nested_iteration`.
"""
mutable struct NestedItConvergenceErrEstimate
    refinement_cond::AbstractRefinementCondition
    tol
    check_every::Int
    basis_ref
    is_last_level::Bool
    n_calls::Int
    final_converged::Bool
    # last estimate, keyed by the identity of info.ψ (e.g. the final is_converged call of RCG)
    ψ_last
    converged_last::Bool
    refine::Bool
    est_err
    est_err_norm
    est_err_perc
end
function NestedItConvergenceErrEstimate(refinement_cond, tol; check_every = 1)
    NestedItConvergenceErrEstimate(refinement_cond, tol, check_every, nothing, false, 0, false,
                                   nothing, false, false, nothing, nothing, nothing)
end

# prepare for the next level of the nested iteration
function reset!(conv::NestedItConvergenceErrEstimate, basis_ref, is_last_level)
    conv.basis_ref = basis_ref
    conv.is_last_level = is_last_level
    conv.n_calls = 0
    conv.final_converged = false
    conv.ψ_last = nothing
    conv.converged_last = false
    conv.refine = false
    conv.est_err = nothing
    conv.est_err_norm = nothing
    conv.est_err_perc = nothing
    conv
end

function (conv::NestedItConvergenceErrEstimate)(info)
    !isnothing(conv.ψ_last) && conv.ψ_last === info.ψ && return conv.converged_last
    conv.n_calls += 1
    mod(conv.n_calls, conv.check_every) == 0 || return false

    basis, basis_ref = info.basis, conv.basis_ref
    # only the occupied orbitals matter for the error estimate (e.g. SCF carries virtual ones)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_bands = div(model.n_electrons, model.n_spin_components * filled_occ, RoundUp)
    ψ = [ψk[:, 1:n_bands] for ψk in info.ψ]
    occupation = [filled_occ * ones(eltype(basis), n_bands) for _ in basis.kpoints]

    # residual on the reference basis, cannot be taken from info (lives on the current basis)
    (; ψ_f, res) = evaluate_prolongated(basis, basis_ref, occupation, ψ)
    est_err = invert_refinement_metric_improved(basis_ref, ψ_f, res)
    est_err_coarse = restrict_vector(basis, basis_ref, ψ, ψ_f, est_err, ProjectionMap())
    est_info = (; basis, basis_ref, est_err, est_err_coarse)
    (; err, err_coarse, perc) = estimate_error_norms(conv.refinement_cond, est_info)

    conv.est_err = est_err
    conv.est_err_norm = err
    conv.est_err_perc = perc
    conv.final_converged = err < conv.tol
    conv.refine = conv.refinement_cond(est_info)
    conv.converged_last = conv.final_converged || conv.refine ||
        (conv.is_last_level && err_coarse < 1e-2 * conv.tol)
    conv.ψ_last = info.ψ
    return conv.converged_last
end
