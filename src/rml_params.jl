mutable struct ApproxCoarseGridCostResidual <: AbstractCostResidual
    ψk
    wk
    function ApproxCoarseGridCostResidual(ψ_c, Rres)
        return new(ψ_c, -Rres)
    end
end

function initialize_cost_residual(Hψ, Λ, res, e_tot, basis, cgcr::ApproxCoarseGridCostResidual)
    Rres = - cgcr.wk
    cgcr.wk += res

    return Hψ, Λ, Rres, e_tot
end

function calculate_cost_residual(H , ψ, e_tot, basis, cgcr::ApproxCoarseGridCostResidual)
    Nk = size(ψ)[1]
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik in 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik in 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik in 1:Nk]

    #iR = invRet(cgcr.ψk, ψ)
    cost = e_tot - inner_product_DFTK(basis, cgcr.wk, ψ) 
    res_c = res - proj_TSt(ψ, cgcr.wk)

    return Hψ, Λ, res_c, cost
end

mutable struct CoarseGridCostResidual <: AbstractCostResidual
    ψk
    wk
    function CoarseGridCostResidual(ψ_c, Rres)
        return new(ψ_c, -Rres)
    end
end

function initialize_cost_residual(Hψ, Λ, res, e_tot, basis, cgcr::CoarseGridCostResidual)
    Rres = - cgcr.wk

    #println("angle ratio (Ecut = $(basis.Ecut)): $(inner_product_DFTK(basis, res, Rres) / (norm_DFTK(basis, Rres) * norm_DFTK(basis, res)))")
    #println("norm res (Ecut = $(basis.Ecut)): $(norm_DFTK(basis, res))")

    cgcr.wk += res
    return Hψ, Λ, Rres, e_tot
end

function calculate_cost_residual(H , ψ, e_tot, basis, cgcr::CoarseGridCostResidual)
    Nk = size(ψ)[1]
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik in 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik in 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik in 1:Nk]

    iR = invRet(cgcr.ψk, ψ)
    cost = e_tot - inner_product_DFTK(basis, iR, ψ) 
    res_c = res - adjDinvRet(cgcr.ψk, ψ, cgcr.wk)

    return Hψ, Λ, res_c, cost
end

mutable struct RcgConvergenceResidualMGH
    tolerance
    max_len
    ψ_c
    last_norm_res
    function RcgConvergenceResidualMGH(tolerance, max_len, ψ_c)
        return new(tolerance, max_len, ψ_c, nothing)
    end
end

function (conv::RcgConvergenceResidualMGH)(info)
    # TODO we calculate iR twice, once here and once in the cost_resiudal function
    iR = RCG_DFTK.invRet(conv.ψ_c, info.ψ)
    check_iR = any([norm(iRk) > conv.max_len for iRk = iR])
    res_increase = isnothing(conv.last_norm_res) ? false : conv.last_norm_res < info.norm_res
    conv.last_norm_res = info.norm_res
    check_tol = info.norm_res < conv.tolerance 
    return check_iR || check_tol || res_increase
end

function interpolate_c2f(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, y) where {T}
    @assert basis_c.kgrid == basis_f.kgrid && basis_c.Ecut <= basis_f.Ecut
    Nk = length(basis_f.kpoints)
    @assert size(y)[1] == Nk

    x = Vector{Matrix{Complex{Float64}}}(undef, length(basis_f.kpoints))

    for (kpt_c, kpt_f, yk, ik) = zip(basis_c.kpoints, basis_f.kpoints, y, 1:Nk)
        @assert size(yk)[1] ==  length(G_vectors(basis_c, kpt_c))
        xk = zeros(Complex{T}, length(G_vectors(basis_f, kpt_f)), size(yk)[2])
        idx_f = 1
        for idx_c = 1:length(G_vectors(basis_c, kpt_c))
            while (kpt_c.G_vectors[idx_c] != kpt_f.G_vectors[idx_f])
                idx_f += 1
            end
            xk[idx_f, :] = yk[idx_c, :]
        end
        x[ik] = xk
    end

    return x
end

function interpolate_f2c(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, x) where {T}
    @assert basis_c.kgrid == basis_f.kgrid && basis_c.Ecut <= basis_f.Ecut
    Nk = length(basis_f.kpoints)
    @assert size(x)[1] == Nk

    y = Vector{Matrix{Complex{Float64}}}(undef, length(basis_c.kpoints))

    for (kpt_c, kpt_f, xk, ik) = zip(basis_c.kpoints, basis_f.kpoints, x, 1:Nk)
        @assert size(xk)[1] ==  length(G_vectors(basis_f, kpt_f))
        yk = zeros(Complex{T}, length(G_vectors(basis_c, kpt_c)), size(xk)[2])
        idx_f = 1
        for idx_c = 1:length(G_vectors(basis_c, kpt_c))
            while (kpt_c.G_vectors[idx_c] != kpt_f.G_vectors[idx_f])
                idx_f += 1
            end
            yk[idx_c, :] = xk[idx_f, :]
        end
        y[ik] = yk
    end

    return y
end

function π_St(x)
    Nk = size(x)[1]
    pf = []
    Sf = []
    Sfinv = []
    for ik in 1:Nk
        xk = x[ik]
        S = xk'xk
        s, U = eigen(S)
        Σ = broadcast(x -> sqrt(abs(x)), s)
        Σ_inv = broadcast(x -> 1.0 / x, Σ)
        Sfinv_k = U * Diagonal(Σ_inv) * U'
        pfk = xk * Sfinv_k
        push!(pf, pfk)
        push!(Sf, U * Diagonal(Σ) * U')
        push!(Sfinv, Sfinv_k)
    end
    return pf, Sf, Sfinv
end

function proj_TSt(ψ, v)
    Nk = size(ψ)[1]
    G = [ψ[ik]'v[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [v[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
end

abstract type AbstractPointRestriction end

mutable struct ProjectiveRestriction <: AbstractPointRestriction
    Sf
    Sfinv
    function ProjectiveRestriction()
        new(nothing, nothing)
    end
end

function restrict_point(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_f, pr::ProjectiveRestriction) where {T}
    y = interpolate_f2c(basis_c, basis_f, ψ_f)
    ψ_c, Sf, Sfinv = π_St(y)
    pr.Sf = Sf
    pr.Sfinv = Sfinv
    return ψ_c
end

function prolongate_point(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c) where {T}
    y = interpolate_c2f(basis_c, basis_f, ψ_c)
    ψ_f, ~, ~ = π_St(y)
    return ψ_f
end

abstract type AbstractVectorMultilevelMap end

struct ProjectionMap <:AbstractVectorMultilevelMap end

function restrict_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, v, ::ProjectionMap) where {T}
    w = interpolate_f2c(basis_c, basis_f, v)
    return proj_TSt(ψ_c, w)
end

function prolongate_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, w, ::ProjectionMap) where {T}
    v = interpolate_c2f(basis_c, basis_f, w)
    return proj_TSt(ψ_f, v)
end

mutable struct AlgebraicMap <: AbstractVectorMultilevelMap
    pr::ProjectiveRestriction
end

function restrict_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, v, am::AlgebraicMap) where {T}
    Nk = length(v)
    w = interpolate_f2c(basis_c, basis_f, v)
    w = [w[ik] * am.pr.Sf[ik] for ik = 1:Nk]
    return proj_TSt(ψ_c, w)
end

function prolongate_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, w, am::AlgebraicMap) where {T}
    Nk = length(w)
    v = interpolate_c2f(basis_c, basis_f, w)
    return [v[ik] * am.pr.Sf[ik] for ik = 1:Nk]
end

mutable struct GeometricMap <: AbstractVectorMultilevelMap
    pr::ProjectiveRestriction
end

function restrict_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, v, gm::GeometricMap) where {T}
    Nk = length(v)
    w = interpolate_f2c(basis_c, basis_f, v)
    Mtx_rhs = [w[ik]' * ψ_c[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    Xv = [lyap(gm.pr.Sfinv[ik], -Mtx_rhs[ik]) for ik = 1:Nk]
    return [(w[ik] - ψ_c[ik] * Xv[ik])* gm.pr.Sfinv[ik]  for ik = 1:Nk]
end

function prolongate_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, w, gm::GeometricMap) where {T}
    Nk = length(w)
    v = interpolate_c2f(basis_c, basis_f, w)
    Iψ = interpolate_c2f(basis_c, basis_f, ψ_c)
    Mtx_rhs = [(ψ_c[ik]' * w[ik]) *  gm.pr.Sfinv[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    v = [v[ik] * gm.pr.Sfinv[ik] - Iψ[ik] *  lyap(gm.pr.Sfinv[ik], -Mtx_rhs[ik]) for ik = 1:Nk]
    return proj_TSt(ψ_f, v)
end

function invRet(ψ,z)
    Nk = size(ψ)[1]
    Mtx_rhs = [2 * I(size(ψ[ik])[2]) for ik = 1:Nk]
    Mtx_lhs = [ψ[ik]'z[ik] for ik = 1:Nk]
    Yz = [lyap(Mtx_lhs[ik], -Mtx_rhs[ik]) for ik = 1:Nk] 
    return [z[ik] * Yz[ik] - ψ[ik] for ik = 1:Nk]
end

function DinvRet(ψ,z,u)
    Nk = size(ψ)[1]
    Mtx_lhs = [ψ[ik]'z[ik] for ik = 1:Nk]

    Mtx_rhs_z = [2 * I(size(ψ[ik])[2]) for ik = 1:Nk]
    Yz = [lyap(Mtx_lhs[ik], -Mtx_rhs_z[ik]) for ik = 1:Nk] 

    Mtx_rhs_u = [ψ[ik]'u[ik] * Yz[ik] for ik = 1:Nk]
    Mtx_rhs_u = [M + M' for M = Mtx_rhs_u]
    Yu = [lyap(Mtx_lhs[ik], Mtx_rhs_u[ik]) for ik = 1:Nk] 
    return [z[ik] * Yu[ik] + u[ik] * Yz[ik] for ik = 1:Nk]
end

function adjDinvRet(ψ,z,u)
    Nk = size(ψ)[1]
    Mtx_lhs = [ψ[ik]'z[ik] for ik = 1:Nk]

    Mtx_rhs_z = [2 * I(size(ψ[ik])[2]) for ik = 1:Nk]
    Yz = [lyap(Mtx_lhs[ik], -Mtx_rhs_z[ik]) for ik = 1:Nk] 

    Mtx_rhs_u = [z[ik]'u[ik] for ik = 1:Nk]
    Mtx_rhs_u = [M + M' for M = Mtx_rhs_u]
    Xzu = [lyap(Mtx_lhs[ik], - Mtx_rhs_u[ik]) for ik = 1:Nk] 
    w = [(u[ik] - ψ[ik] * Xzu[ik]) * Yz[ik] for ik = 1:Nk]
    return proj_TSt(z, w)
end

"""
Decides whether the next direction of `two_level_riemannian_optimization` is a coarse
correction. A condition defines `coarse_condition_holds(ctx, cc)` for a
`CoarseConditionContext` `ctx`, which must not change `cc`, and, if it carries state,
`update_coarse_condition!(cc, ctx)`, which is called once a coarse correction is taken.
Conditions are combined with `&` (all have to hold) and `|` (one has to hold), e.g.
`MinStepsCoarseCondition(1) & ResidualRatioCoarseCondition(0.45)`; they are evaluated from
left to right and only as far as needed, so cheap conditions should come first.
"""
abstract type AbstractCoarseCondition end

"""
Everything a coarse condition may use at the current fine iterate: `basis_c`, `basis_f`, `ψ_c`,
`ψ_f`, the fine residual `res`, the restricted residual `Rres` and the number of performed
steps `n_iter`. The coarse density and the evaluation on the coarse grid (energies,
Hamiltonian and standard residual of the coarse problem at `ψ_c`) are computed on demand by
`coarse_density!` and `coarse_evaluation!` and cached; if a coarse correction is taken, they are
passed on to the coarse solver instead of being computed again.
"""
mutable struct CoarseConditionContext
    basis_c
    basis_f
    ψ_c
    ψ_f
    res
    Rres
    n_iter
    occupation
    ρ_f
    coarse_density  # AbstractCoarseDensity
    ρ_c
    coarse_eval
end
CoarseConditionContext(basis_c, basis_f, ψ_c, ψ_f, res, Rres, n_iter, occupation, ρ_f, coarse_density) =
    CoarseConditionContext(basis_c, basis_f, ψ_c, ψ_f, res, Rres, n_iter, occupation, ρ_f,
                           coarse_density, nothing, nothing)

function coarse_density!(ctx::CoarseConditionContext)
    if isnothing(ctx.ρ_c)
        ctx.ρ_c = calculate_coarse_density(ctx.basis_c, ctx.basis_f, ctx.ρ_f, ctx.ψ_c, ctx.ψ_f,
                                           ctx.coarse_density)
    end
    return ctx.ρ_c
end

# (; energies, H, Hψ, Λ, res) of the (uncorrected) coarse problem at ψ_c, see initial_eval
# of riemannian_conjugate_gradient
function coarse_evaluation!(ctx::CoarseConditionContext)
    if isnothing(ctx.coarse_eval)
        energies, H = energy_hamiltonian(ctx.basis_c, ctx.ψ_c, ctx.occupation; ρ = coarse_density!(ctx))
        Hψ, Λ, res = compute_residual(H, ctx.ψ_c)
        ctx.coarse_eval = (; energies, H, Hψ, Λ, res)
    end
    return ctx.coarse_eval
end

function check_coarse_condition(ctx::CoarseConditionContext, cc::AbstractCoarseCondition)
    coarse_condition_holds(ctx, cc) || return false
    update_coarse_condition!(cc, ctx)
    return true
end

update_coarse_condition!(cc::AbstractCoarseCondition, ctx) = cc

# Called by two_level_riemannian_optimization on its own (deep) copy of the coarse condition,
# with the tolerance tol of that level.
init_coarse_condition!(cc::AbstractCoarseCondition, tol) = cc

# all (AllCoarseConditions) or one (AnyCoarseCondition) of the conditions have to hold; once a
# coarse correction is taken, all conditions are updated
struct AllCoarseConditions <: AbstractCoarseCondition
    conditions::Vector{AbstractCoarseCondition}
end
struct AnyCoarseCondition <: AbstractCoarseCondition
    conditions::Vector{AbstractCoarseCondition}
end
const CombinedCoarseCondition = Union{AllCoarseConditions, AnyCoarseCondition}

all_parts(cc::AllCoarseConditions) = cc.conditions
all_parts(cc::AbstractCoarseCondition) = AbstractCoarseCondition[cc]
any_parts(cc::AnyCoarseCondition) = cc.conditions
any_parts(cc::AbstractCoarseCondition) = AbstractCoarseCondition[cc]
Base.:&(a::AbstractCoarseCondition, b::AbstractCoarseCondition) =
    AllCoarseConditions(vcat(all_parts(a), all_parts(b)))
Base.:|(a::AbstractCoarseCondition, b::AbstractCoarseCondition) =
    AnyCoarseCondition(vcat(any_parts(a), any_parts(b)))

coarse_condition_holds(ctx, cc::AllCoarseConditions) = all(c -> coarse_condition_holds(ctx, c), cc.conditions)
coarse_condition_holds(ctx, cc::AnyCoarseCondition) = any(c -> coarse_condition_holds(ctx, c), cc.conditions)
function update_coarse_condition!(cc::CombinedCoarseCondition, ctx)
    foreach(c -> update_coarse_condition!(c, ctx), cc.conditions)
    return cc
end
function init_coarse_condition!(cc::CombinedCoarseCondition, tol)
    foreach(c -> init_coarse_condition!(c, tol), cc.conditions)
    return cc
end

# the restricted residual is at least η times the fine residual
struct ResidualRatioCoarseCondition <: AbstractCoarseCondition
    η
end
coarse_condition_holds(ctx, cc::ResidualRatioCoarseCondition) =
    norm_DFTK(ctx.basis_c, ctx.Rres) ≥ cc.η * norm_DFTK(ctx.basis_f, ctx.res)

# ψ_c has moved more than ϵ since the last coarse correction
# (ϵ = nothing: use the tolerance of the level the condition is used on, see init_coarse_condition!)
mutable struct MinDistanceCoarseCondition <: AbstractCoarseCondition
    ϵ
    ψ_c
end
MinDistanceCoarseCondition(ϵ = nothing) = MinDistanceCoarseCondition(ϵ, nothing)
function init_coarse_condition!(cc::MinDistanceCoarseCondition, tol)
    isnothing(cc.ϵ) && (cc.ϵ = tol)
    return cc
end
coarse_condition_holds(ctx, cc::MinDistanceCoarseCondition) =
    isnothing(cc.ψ_c) || norm_DFTK(ctx.basis_c, cc.ψ_c - ctx.ψ_c) > cc.ϵ
function update_coarse_condition!(cc::MinDistanceCoarseCondition, ctx)
    cc.ψ_c = ctx.ψ_c
    return cc
end

# at least dist (fine) steps since the last coarse correction; dist = 1: no two successive ones
mutable struct MinStepsCoarseCondition <: AbstractCoarseCondition
    dist
    n_iter
end
MinStepsCoarseCondition(dist = 1) = MinStepsCoarseCondition(dist, 0)
coarse_condition_holds(ctx, cc::MinStepsCoarseCondition) = cc.n_iter ≤ ctx.n_iter
function update_coarse_condition!(cc::MinStepsCoarseCondition, ctx)
    cc.n_iter = ctx.n_iter + cc.dist + 1
    return cc
end

# the residual of the (uncorrected) coarse problem at ψ_c is below tol, measured in norm (:L2
# or :H1). Needs the coarse Hamiltonian, which is reused by the coarse solver.
struct ResidualNormCoarseCondition <: AbstractCoarseCondition
    tol
    norm::Symbol
end
ResidualNormCoarseCondition(tol; norm = :L2) = ResidualNormCoarseCondition(tol, norm)
coarse_condition_holds(ctx, cc::ResidualNormCoarseCondition) =
    norm_DFTK(ctx.basis_c, coarse_evaluation!(ctx).res, cc.norm) < cc.tol

# the former combined conditions
ToleranceMinStepCoarseCondition(η, ϵ = nothing; dist = 1) =
    MinStepsCoarseCondition(dist) & MinDistanceCoarseCondition(ϵ) & ResidualRatioCoarseCondition(η)

EveryKCoarseCorr(k) = MinStepsCoarseCondition(k - 1)

# classical (non-adaptive) cycles: every cycle consists of ν_pre smoothing (gradient) steps,
# one coarse correction and ν_post smoothing steps. Used as coarse condition on every level;
# the inner levels perform γ cycles per coarse correction of the next finer level
# (γ = 1: V-cycle, γ = 2: W-cycle), see multilevel_riemannian_optimization.
struct CycleCoarseCondition <: AbstractCoarseCondition
    ν_pre::Int
    ν_post::Int
    γ::Int
end
CycleCoarseCondition(ν_pre, ν_post) = CycleCoarseCondition(ν_pre, ν_post, 1)
VCycle(ν_pre, ν_post = ν_pre) = CycleCoarseCondition(ν_pre, ν_post, 1)
WCycle(ν_pre, ν_post = ν_pre) = CycleCoarseCondition(ν_pre, ν_post, 2)
SawtoothCycle(ν_post) = CycleCoarseCondition(0, ν_post, 1)

cycle_length(cc::CycleCoarseCondition) = cc.ν_pre + 1 + cc.ν_post

# n_iter is the number of steps already performed, i.e. the direction for step n_iter + 1 is chosen
coarse_condition_holds(ctx, cc::CycleCoarseCondition) = mod(ctx.n_iter, cycle_length(cc)) == cc.ν_pre

abstract type AbstractCoarseDensity end

struct RecalculateDensity <: AbstractCoarseDensity end

function compute_density(basis, ψ)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    # number of kpoints and occupation
    Nk = length(basis.kpoints)

    occupation = [filled_occ * ones(eltype(basis), n_bands) for ik in 1:Nk]
   
    return DFTK.compute_density(basis, ψ , occupation)
end

function calculate_coarse_density(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ρ_f, ψ_c, ψ_f, ::RecalculateDensity) where {T}
    # check that there are no virtual orbitals
    model = basis_c.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    # number of kpoints and occupation
    Nk = length(basis_c.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik in 1:Nk]
   
    return DFTK.compute_density(basis_c, ψ_c , occupation)
end

struct InterpolateDensity <: AbstractCoarseDensity end

function calculate_coarse_density(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ρ_f, ψ_c, ψ_f, ::InterpolateDensity) where {T}
    return DFTK.interpolate_density(ρ_f, basis_f, basis_c)
end

abstract type AbstractCoarseTolerance end

# get_coarse_tol(info, coarse_tol) returns the tolerance of the coarse solve. info is the
# (callback) info of the calling method at the current iterate; it contains in particular the
# norm of the restricted residual norm_Rres and the tolerance tol of the calling method.

# relative reduction of the restricted residual, but not below the tolerance of the calling method
struct RelativeResTolerance <: AbstractCoarseTolerance
    μ
end

function get_coarse_tol(info, rrt::RelativeResTolerance)
    return max(info.norm_Rres * rrt.μ, info.tol)
end