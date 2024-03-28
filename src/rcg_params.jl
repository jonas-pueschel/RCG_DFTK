include("./lyap_solvers.jl")
using Krylov
using DFTK
using Parameters

abstract type AbstractGradient end 

struct RiemannianGradient <: AbstractGradient 
    Pks_Metric
end

function get_H1_gradient(basis::PlaneWaveBasis{T}) where {T}
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    return RiemannianGradient(Pks)
end

function calculate_gradient(ψ, Hψ, H, Λ, res, riem_grad::RiemannianGradient)
    Nk = size(ψ0)[1]
    Pks = riem_grad.Pks_Metric
    P_ψ = [ Pks[ik] \ ψ[ik] for ik = 1:Nk]
    P_Hψ = [ Pks[ik] \ Hψ[ik] for ik = 1:Nk]
    Mtx_lhs = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
    Mtx_rhs = [ψ[ik]'P_Hψ[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    X = [lyap(Mtx_lhs[ik], -Mtx_rhs[ik]) for ik = 1:Nk]
    return [P_Hψ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
end



struct L2Gradient <: AbstractGradient end
function calculate_gradient(ψ, Hψ, H, Λ, res, ::L2Gradient)
    return res
end

@with_kw mutable struct EAGradient <: AbstractGradient 
    Hinv_ψ = nothing
    const itmax::Int64
    const atol::Float64
    const rtol::Float64
    const Pks
    const lin_solver #solver for linear systems
    const h_solver   #type of solver for H 
end

function get_default_EA_gradient(basis::PlaneWaveBasis{T}) where {T}
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    for ik = 1:length(Pks)
        DFTK.precondprep!(Pks[ik], ψ[ik])
    end
    return EAGradient(
        5,
        1e-16,
        1e-16,
        Pks,
        Krylov.minres,
        AbstractHSolver()
    )
end

abstract type AbstractHSolver end

struct ParallelHSolver <: AbstractHSolver end
function solve_H(lin_solver, ψ0, H, b, itmax, atol, rtol, Pks, ::ParallelHSolver)
    Nk = size(ψ0)[1]
    T = Base.Float64
    pack(ψ) = copy(DFTK.reinterpret_real(DFTK.pack_ψ(ψ)))
    unpack(x) = DFTK.unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))
    unsafe_unpack(x) = DFTK.unsafe_unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))

    x0 = pack(ψ0)
    rhs = pack(b)



    function apply_H(x)
        #TODO shift H? 
        ξ = unsafe_unpack(x)
        return pack(H * ξ)
    end

    J = LinearMap{T}(apply_H, size(x0, 1))

    function apply_Prec(x)
        ξ = unsafe_unpack(x)
        Pξ = [Pks[ik] \ ξ[ik] for ik = 1:Nk]
        return pack(Pξ)
    end

    M = LinearMap{T}(apply_Prec, size(x0, 1))
    
    res = lin_solver(J, rhs, x0; M, itmax, atol, rtol)
    return unpack(res[1])
end

struct SequentialHSolver <: AbstractHSolver end
function solve_H(lin_solver, ψ0, H, b, itmax, atol, rtol, Pks, ::SequentialHSolver)
    Nk = size(ψ0)[1]
    res = []
    for ik = 1:Nk
        x0 = ψ0[ik]
        rhs = b[ik]

        function apply_H(x)
            #TODO shift H? 
            return pack(H[ik] * x)
        end
    
        J = LinearMap{T}(apply_H, size(x0, 1))

        function apply_Prec(x)
            return Pks[ik] \ x
        end
        M = LinearMap{T}(apply_Prec, size(x0, 1))
        push!(res, [lin_solver(J, rhs, x0; M, itmax, atol, rtol)])
    end
    return res 
end

function calculate_gradient(ψ, Hψ, H, Λ, res, ea_grad::EAGradient)
    Nk = size(ψ)[1]
    ψ0 = [ψ[ik] / Λ[ik] for ik = 1:Nk]
    # a bit overblown, maybe pass ea_grad?
    ea_grad.Hinv_ψ = solve_H(ea_grad.lin_solver, ψ0, H, ψ, ea_grad.itmax, ea_grad.atol, ea_grad.rtol, ea_grad.Pks, ea_grad.h_solver)
    Ga = [ψ[ik]'ea_grad.Hinv_ψ[ik] for ik = 1:Nk]
    return [ ψ[ik] - Hinv_ψ[ik] / Ga[ik] for ik = 1:Nk]
end


struct PrecGradient <: AbstractGradient 
    apply_prec
end

function calculate_gradient(ψ, Hψ, H, Λ, res, prec_grad::PrecGradient)
    Nk = size(ψ)[1]
    P_res = prec_grad.apply_prec(ψ, Hψ, H, Λ, res) 
    G = [ψ[ik]'P_res[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [P_res[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
end

abstract type AbstractRetraction end

function default_retraction(basis::PlaneWaveBasis{T}) where {T}
    Nk = length(basis.kpoints)
    if (Nk == 1)
        return RetractionQR()
    else
        return RetractionPolar()
    end
end
@with_kw mutable struct RetractionQR <: AbstractRetraction
    R = nothing
end

function calculate_retraction(ψ, η, τ, ret_qr::RetractionQR)
    Nk = size(ψ)[1]
    Q = []
    R = []
    for ik = 1:Nk
        φk = ψ[ik] + τ[ik] * η[ik]
        Nn, Nm = size(φk)
        temp = qr(φk)
        Rk = convert(ArrayType, temp.R)
        Qk = convert(ArrayType, temp.Q)
        push!(Q, [Qk[1:Nn, 1:Nm]])
        push!(R, [Rk[1:Nm, 1:Nm]])
    end
    #save this value for further calculations
    ret_qr.R = R
    return Q
end

@with_kw mutable struct RetractionPolar <: AbstractRetraction
    P = nothing
    Pinv = nothing
end
function calculate_retraction(ψ, η, τ, ret_pol::RetractionPolar)
    Nk = size(ψ)[1]
    Q = []
    P = []
    Pinv = []
    for ik = 1:Nk
        φk = ψ[ik] + τ[ik] * η[ik]
        S = φk'φk
        s, U = eigen(S)
        σ = broadcast(x -> sqrt(x), s)
        σ_inv = broadcast(x -> 1.0/x, σ)
        Pinvk = U * Diagonal(σ_inv) * U'
        Qk = φk * Pinvk
        push!(Q, [Qk])
        push!(P, [U * Diagonal(σ) * U'])
        push!(Pinv, [Pinvk])
    end
    #save values for further calculations
    ret_pol.P = P
    ret_pol.Pinv = Pinv
    return Q
end

abstract type AbstractTransport end

default_transport() = DifferentiatedRetractionTransport()

struct DifferentiatedRetractionTransport <: AbstractTransport end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_qr::RetractionQR, ::AbstractGradient ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Yrf = [ξ[ik] / R[ik] for ik = 1:Nk]
    Ge = [ψ[ik]'Yrf[ik] for ik = 1:Nk]
    skew = [LowerTriangular(deepcopy(Ge[ik])) for ik = 1:Nk]
    skew = [skew[ik] - skew[ik]' for ik = 1:Nk]
    for ik = 1:Nk
        for j = 1:size(skew[ik])[1]
            skew[ik][j,j] = 0.5 * skew[ik][j,j] 
        end
    end
    return [ ψ[ik] * (skew[ik] - Ge[ik])  + Yrf[ik] for ik = 1:Nk]
end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_pol::RetractionPolar, ::AbstractGradient ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    if (is_prev_dir)
        ξPinv = [ξ[ik] * ret_pol.Pinv[ik] for ik = 1:Nk]
        return [ξPinv[ik] - τ[ik] * ψ[ik] * (ξPinv[ik]'ξPinv[ik]) for ik = 1:Nk]
    else
        Yrf = [ξ[ik] * ret_pol.P[ik] for ik = 1:Nk]
        Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
        Mtx_rhs = [Mtx_rhs[ik]' - Mtx_rhs[ik] for ik = 1:Nk]
        X = [lyap(ret_pol.Pinv[ik], Mtx_rhs[ik]) for ik = 1:Nk]
        return [ ψ[ik] * X[ik] + Yrf[ik] + ψ[ik] * (ψ[ik]'Yrf[ik]) for ik = 1:Nk]
    end

end

struct L2ProjectionTransport <: AbstractTransport end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::L2ProjectionTransport,  
                            ::AbstractRetraction, ::AbstractGradient ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    G = [ψ[ik]'ξ[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [ξ[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
end

struct RiemannianProjectionTransport <: AbstractTransport 
    Pks_Metric
end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, riem_proj::RiemannianProjectionTransport,  
                            ::AbstractRetraction, ::AbstractGradient ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Pks = riem_proj.Pks_Metric
    P_ψ = [ Pks[ik] \ ψ[ik] for ik = 1:Nk]
    Mtx_lhs = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
    Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    X = [lyap(Mtx_lhs[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
    return [ξ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
end

struct EAProjectionTransport <: AbstractTransport end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::EAProjectionTransport,  
                            ::AbstractRetraction, ea_grad::EAGradient ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Mtx_lhs = [ψ[ik]'ea_grad.Hinv_ψ[ik] for ik = 1:Nk]
    Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    X = [lyap(Mtx_lhs[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
    return [ξ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
end

struct InverseRetractionTransport <: AbstractTransport end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionQR, ::AbstractGradient ; 
                            is_prev_dir = true)
    if (!is_prev_dir)
        throw(DomainError("inverse Retraction not implemented for non-η directions"))
    end
    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
    R_ir = [tsylv2solve(Mtx_lhs[ik]) for ik = 1:Nk]
    return [- ψ_old[ik] * R_ir[ik] + ψ[ik] for ik = 1:Nk]
end
function calculate_tranpsort(ψ, ξ, τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionPolar, ::AbstractGradient ; 
                            is_prev_dir = true)
    if (!is_prev_dir)
        throw(DomainError("inverse Retraction not implemented for non-η directions"))
    end
    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
    S_ir = [lyap(Mtx_lhs[ik], - 2.0 * Matrix{ComplexF64}(I, size(ψ)[1], size(ψ)[1])) for ik = 1:Nk]
    return [- ψ_old[ik] * S_ir[ik] + ψ[ik] for ik = 1:Nk]
end


abstract type AbstractCGParam end

struct ParamZero <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamZero)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamZero)
    Nk = size(gamma[1])
    return [0.0 for ik = 1:Nk]   
end



@with_kw mutable struct ParamHS <: AbstractCGParam 
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHS)
    param.grad_old = grad
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHS)
    Nk = size(gamma[1])
    T_grad_old = transport(param.grad_old)
    β = [real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/(tr(T_η_old[ik]'res[ik]) - desc_old[ik])) for ik = 1:Nk]

    param.grad_old = grad

    return β
end

struct ParamDY <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamDY)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamDY)
    Nk = size(gamma[1])
    return [real(gamma[ik]/(tr(T_η_old[ik]'res[ik]) - desc_old[ik])) for ik = 1:Nk]
end

@with_kw mutable struct ParamPRP <: AbstractCGParam
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamPRP)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamPRP)
    Nk = size(gamma[1])
    β = [real((gamma[ik] - tr(T_η_old[ik]'res[ik]))/param.gamma_old[ik]) for ik = 1:Nk]
    
    param.gamma_old = gamma

    return  β
end

@with_kw mutable struct ParamFR <: AbstractCGParam 
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamFR)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamFR)
    Nk = size(gamma[1])
    β = [real(gamma[ik]/param.gamma_old[ik]) for ik = 1:Nk]   
    
    param.gamma_old = gamma

    return  β
end


@with_kw mutable struct ParamFR_PRP <: AbstractCGParam
    gamma_old = nothing
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamPRP)
    Nk = size(gamma[1])
    β = [max(min(gamma[ik], (gamma[ik] - real(tr(T_η_old[ik]'res[ik]))))/param.gamma_old[ik], 0) for ik = 1:Nk]

    param.gamma_old = gamma

    return  β
end

@with_kw mutable struct ParamHS_DY <: AbstractCGParam 
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHS_DY)
    param.grad_old = grad
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHS_DY)
    Nk = size(gamma[1])
    T_grad_old = transport(param.grad_old)
    temp = [real(tr(T_η_old[ik]'res[ik]) - desc_old[ik]) for ik = 1:Nk ]
    β = [max(min(real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/temp[ik] - desc_old[ik]), real(gamma[ik]/temp[ik])), 0) for ik = 1: Nk]

    param.grad_old = grad

    return β
end

@with_kw mutable struct ParamHZ <: AbstractCGParam 
    const μ
    gamma_old = nothing
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHZ)
    param.grad_old = grad
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHZ)
    Nk = size(gamma[1])
    T_grad_old = transport(param.grad_old)
    temp1 = [real(tr(T_η_old[ik]'res[ik]) - desc_old[ik]) for ik = 1:Nk]
    temp2 = [real(gamma[ik] - 2 * tr(T_grad_old[ik]'res[ik]) + param.gamma_old[ik]/temp1[ik]) for ik = 1:Nk]
    β = [(gamma[ik] - real(tr(T_grad_old[ik]'res[ik]) - tr(T_η_old[ik]'res[ik]) * param.μ * temp2[ik]))/temp1[ik] for ik = 1:Nk]     
    
    param.grad_old = grad
    param.gamma_old = gamma

    return β
end


abstract type AbstractStepSize end

#default_stepsize() = ExactHessianStep(2.0)

struct ExactHessianStep <: AbstractStepSize 
    τ_max::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, τ_old, desc, Λ, H, ρ, occupation, basis, stepsize::ExactHessianStep)
    Nk = size(ψ[1])
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    K_η = DFTK.apply_K(basis, η, ψ, ρ, occupation)
    η_K_η = [tr(η[ik]'K_η[ik]) for ik in 1:Nk]
    #return [real(η_Ω_η[ik] + η_K_η[ik]) > 0 ? real(- desc[ik]/(η_Ω_η[ik] + η_K_η[ik])) : stepsize.τ_max for ik in 1:Nk]
    return [min(real(- desc[ik]/abs(η_Ω_η[ik] + η_K_η[ik])) , stepsize.τ_max) for ik in 1:Nk]
end


struct ApproxHessianStep <: AbstractStepSize
    τ_max::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, τ_old, desc, Λ, H, ρ, occupation, basis, ::ApproxHessianStep)
    Nk = size(ψ[1])
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    return [min(real(abs(desc[ik]/(η_Ω_η[ik]))), stepsize.τ_max) for ik in 1:Nk]
end
struct ConstantStep <: AbstractStepSize
    τ_const
end
function calculate_τ(ψ, η, grad, res, T_η_old, τ_old, desc, Λ, H, ρ, occupation, basis, ::ConstantStep)
    return options.τ_const
end


#Note: This may only show intended behaviour for β ≡ 0, i.e. η = -grad
@with_kw mutable struct BarzilaiBorweinStep <: AbstractStepSize
    const τ_min::Float64
    const τ_max::Float64
    const τ_0
    desc_old = nothing
    is_odd = false
end
function calculate_τ(ψ, η, grad, res, T_η_old, τ_old, desc, Λ, H, ρ, occupation, basis, stepsize::BarzilaiBorweinStep)
    Nk = size(ψ[1])
    if (η_old === nothing)
        #first step
        τ = τ_0
    else
        temp = [real(tr(T_η_old[ik]'res[ik])) for ik = 1:Nk]
        desc_old = stepsize.desc_old
        if (is_odd)
            #short step size
            τ = [τ_old[ik] * abs((- desc_old[ik] + temp[ik]) / (-desc[ik] - desc_old[ik] + 2 * temp[ik])) for ik = 1:Nk]
        else
            #long step size
            τ = [τ_old[ik] * abs((- desc_old[ik])/(- desc_old[ik] + temp[ik])) for ik = 1:Nk]
        end
    end
    #apply min, max
    τ = [min(max(τ[ik], stepsize.τ_min), stepsize.τ_max) for ik = 1:Nk]

    #update params
    stepsize.desc_old = desc
    stepsize.is_odd = !stepsize.is_odd

    return τ
end

abstract type AbstractBacktrackingRule end

default_rule() = AvgNonmonotoneRule(0.95, 0.0001, 0.5)

@with_kw mutable struct AvgNonmonotoneRule <: AbstractBacktrackingRule
    const α::Float64
    const β::Float64
    const δ::Float64
    q = 1 
    c = nothing
end

function check_rule(E_current, E_next, τ, desc, rule::AvgNonmonotoneRule)
    return E_next <= rule.c + rule.β * sum(τ .* desc) 
end
function backtrack!(τ, rule::AvgNonmonotoneRule)
    τ .= rule.δ * τ
end
function update_rule!(E, rule::AvgNonmonotoneRule)
    rule.q = rule.α * rule.q + 1
    rule.c = (1-1/rule.q)*rule.c + 1/rule.q * E;
end

struct AvgArmijoRule <: AbstractBacktrackingRule
    β::Float64
    δ::Float64
end
function check_rule(E_current, E_next, τ, desc, rule::AvgArmijoRule)
    return E_next <= E_current + rule.β * sum(τ .* desc) 
end
function backtrack!(τ, rule::AvgArmijoRule)
    τ .= rule.δ * τ
end
function update_rule!(E, ::AvgArmijoRule) end



abstract type AbstractBacktracking end

@with_kw mutable struct StandardBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const max_iter::Int64
    const greedy::Bool
    τ_old = nothing
end


function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, stepsize::AbstractStepSize, backtracking::StandardBacktracking)


    if (backtracking.greedy && backtracking.τ_old !== nothing)
        #try to reuse old step size
        τ = backtracking.τ_old
        ψ_next = calculate_retraction(ψ, η, τ, retraction)
        ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
        energies_next, H_next = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)
        
        if check_rule(energies.total, energies_next.total, τ, desc, backtracking.rule)
            update_rule!(energies_next, backtracking.rule)
            return (;ψ_next, ρ_next, energies_next, H_next)
        end 
    end

    τ = calculate_τ(ψ, η, grad, res, T_η_old, backtracking.τ_old, desc, Λ, H, ρ, occupation, basis, stepsize)

    for k in 0:backtracking.max_iter
        if (k != 0)
            backtrack!(τ, rule)
        end
        
        ψ_next = calculate_retraction(ψ, η, τ, ret)
        ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
        energies_next, H_next = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)
        

        if check_rule(energies.total, energies_next.total, τ, desc, backtracking.rule)
            break
        end 
    end

    backtracking.τ_old = τ
    update_rule!(energies_next, backtracking.rule)
    return (;ψ_next, ρ_next, energies_next, H_next)
end


