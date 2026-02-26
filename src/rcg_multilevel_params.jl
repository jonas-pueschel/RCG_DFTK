mutable struct ApproxCoarseGridCostResidual <: AbstractCostResidual
    ψk
    wk
    function ApproxCoarseGridCostResidual(ψ_c, Rres)
        return new(ψ_c, -Rres)
    end
end

function initialize_cost_residual(H, ψ, e_tot, basis, cgcr::ApproxCoarseGridCostResidual)
    Nk = size(ψ)[1]
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik in 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik in 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik in 1:Nk]

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

function initialize_cost_residual(H, ψ, e_tot, basis, cgcr::CoarseGridCostResidual)
    Nk = size(ψ)[1]
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik in 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik in 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik in 1:Nk]

    Rres = - cgcr.wk
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

mutable struct PseudoInverse_1_2_Map <: AbstractVectorMultilevelMap
    pr::ProjectiveRestriction
end

function restrict_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, v, pim::PseudoInverse_1_2_Map) where {T}
    Nk = length(v)
    w = interpolate_f2c(basis_c, basis_f, v)
    w = [w[ik] * pim.pr.Sf[ik] for ik = 1:Nk]
    return proj_TSt(ψ_c, w)
end

function prolongate_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, w, pim::PseudoInverse_1_2_Map) where {T}
    Nk = length(w)
    v = interpolate_c2f(basis_c, basis_f, w)
    return [v[ik] * pim.pr.Sf[ik] for ik = 1:Nk]
end

mutable struct MoorePenroseMap <: AbstractVectorMultilevelMap
    pr::ProjectiveRestriction
end

function restrict_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, v, ::MoorePenroseMap) where {T}
    # TODO
end

function prolongate_vector(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ_c, ψ_f, w, ::MoorePenroseMap) where {T}
    # TODO
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

abstract type AbstractCoarseCondition end

mutable struct ToleranceMinStepCoarseCondition <: AbstractCoarseCondition
    η
    ϵ
    ψ_c
    dist
    n_iter
    function ToleranceMinStepCoarseCondition(η, ϵ; dist = 1)
        # we enforce that the very first step is always a gradient step
        return new(η, ϵ, nothing, dist, max(1,dist))
    end
end

function EveryKCoarseCorr(k)
    return ToleranceMinStepCoarseCondition(0.0, 0.0; dist = k)
end

function check_coarse_condition(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, ψ, res, Rres, cc::ToleranceMinStepCoarseCondition) where {T}
    if (cc.n_iter > 0)
        cc.n_iter -= 1
        return false
    end
    c1 = norm_DFTK(basis_c, Rres) ≥ cc.η * norm_DFTK(basis_f, res) 
    c2 = isnothing(cc.ψ_c) ? true : norm_DFTK(basis_f, cc.ψ_c - ψ) > cc.ϵ
    (c1 && c2) && (cc.ψ_c = ψ)
    (c1 && c2) && (cc.n_iter = cc.dist)
    return c1 && c2
end

abstract type AbstractCoarseDensity end

struct RecalculateDensity <: AbstractCoarseDensity end

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

struct RelativeResTolerance <: AbstractCoarseTolerance
    μ
    tol
end

function get_coarse_tol(basis_c::PlaneWaveBasis{T}, basis_f::PlaneWaveBasis{T}, Rres, res, rrt::RelativeResTolerance) where {T}
    return max(norm_DFTK(basis_c, Rres) * rrt.μ, rrt.tol)
end

