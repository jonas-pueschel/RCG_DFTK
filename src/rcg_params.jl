using Krylov
using DFTK

include("./lyap_solvers.jl")
include("./local_optimal_solvers.jl")

abstract type AbstractGradient end 

struct RiemannianGradient <: AbstractGradient 
    Pks_Metric
end

function H1Gradient(basis::PlaneWaveBasis{T}) where {T}
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    return RiemannianGradient(Pks)
end

function calculate_gradient(ψ, Hψ, H, Λ, res, riem_grad::RiemannianGradient)
    Nk = size(ψ)[1]
    Pks = riem_grad.Pks_Metric
    #for ik = 1:length(Pks)
    #    DFTK.precondprep!(Pks[ik], ψ[ik])
    #end
    P_ψ = [ Pks[ik] \ ψ[ik] for ik = 1:Nk]
    P_Hψ = [ Pks[ik] \ Hψ[ik] for ik = 1:Nk]
    Mtx_lhs = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
    Mtx_rhs = [ψ[ik]'P_Hψ[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    X = [lyap(Mtx_lhs[ik], -Mtx_rhs[ik]) for ik = 1:Nk]
    g = [P_Hψ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
    return g
end



struct L2Gradient <: AbstractGradient end
function calculate_gradient(ψ, Hψ, H, Λ, res, ::L2Gradient)
    return res
end


abstract type AbstractShiftStrategy end


mutable struct EAGradient <: AbstractGradient
    const basis
    const itmax::Int64
    const rtol::Float64
    const atol::Float64
    const Pks
    const shift::AbstractShiftStrategy
    const krylov_solver #solver for linear systems
    const h_solver::AbstractHSolver   #type of solver for H 
    Hinv_ψ

    function EAGradient(basis::PlaneWaveBasis{T}, shift; itmax = 100, rtol = 1e-2, atol = 1e-16, krylov_solver = Krylov.minres, h_solver = NaiveHSolver(), Pks = nothing) where {T}
        if isnothing(Pks)
            Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        end
        return new(basis, itmax, rtol, atol, Pks, shift, krylov_solver, h_solver, nothing)
    end
end

function default_EA_gradient(basis)
    shift = CorrectedRelativeΛShift(; μ = 0.01)
    return EAGradient(basis, shift; 
        rtol = 2.5e-2,
        itmax = 10,
        h_solver = LocalOptimalHSolver(basis, ProjectedSystemLOIS),
        krylov_solver = Krylov.minres,
        #Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        Pks = [PreconditionerLOIS(basis, kpt, 1) for kpt in basis.kpoints]
        ) 
end


struct ConstantShift <: AbstractShiftStrategy 
    Σ
end
function get_constant_shift(Σ::Number, Nk)
    return [Σ for ik = 1:Nk]
end
function get_constant_shift(Σ::AbstractArray, Nk)
    return [Σ[ik] for ik = 1:Nk]
end
function calculate_shift(ψ, Hψ, H, Λ, res, shift::ConstantShift)
    Nk = size(ψ)[1]
    return get_constant_shift(shift.Σ, Nk)
end

struct RelativeEigsShift <: AbstractShiftStrategy
    μ::Float64
end
function calculate_shift(ψ, Hψ, H, Λ, res, shift::RelativeEigsShift)
    Nk = size(ψ)[1] 
    λ_min = [min(real(eigen(Λ[ik]).values)...) for ik = 1:Nk]
    return [- λ_min[ik] * shift.μ for ik = 1:Nk]
end

#TODO: Save SVD for prec?
mutable struct CorrectedRelativeΛShift <: AbstractShiftStrategy 
    μ
    recalculate_μ::Bool
    function CorrectedRelativeΛShift(;μ = nothing)
        recalculate_μ = isnothing(μ)
        return new(μ, recalculate_μ)
    end
end

function calculate_shift(ψ, Hψ, H, Λ, res, shift::CorrectedRelativeΛShift)
    #correcting
    Nk = size(ψ)[1] 
    Σ = []
    if (shift.recalculate_μ)
        shift.μ = min(norm(res), 1.0)
    end

    for ik = 1:Nk
        s, U = eigen(Λ[ik]) #TODO cast to real may change eigvals?
        σ = broadcast(x -> real(x) < 0 ? -( 1 + shift.μ) * real(x) :  - (1 - shift.μ) * real(x) , s)
        Σk = U * Diagonal(σ) * U'
        push!(Σ, Σk)
    end
    #return = [ λ_max[ik] < 0 ? (- shift.μ * Λ[ik]) : 0 * Λ[ik] for ik = 1:Nk]
    return Σ
end

function calculate_gradient(ψ, Hψ, H, Λ, res, ea_grad::EAGradient)
    Nk = size(ψ)[1];
    Σ = calculate_shift(ψ, Hψ, H, Λ, res, ea_grad.shift);

    for ik = 1:Nk
       DFTK.precondprep!(ea_grad.Pks[ik], ψ[ik]); 
    end

    inner_tols = [min(max(ea_grad.rtol * norm(res[ik]), ea_grad.atol), 1.0) for ik = 1:Nk]


    X = solve_H(ea_grad.krylov_solver, H, res, Σ, ψ, Hψ, ea_grad.itmax, inner_tols, ea_grad.Pks, ea_grad.h_solver)

    #G1 = [ψ[ik] - (ψ[ik] -X[ik]) /(I - ψ[ik]'X[ik]) for ik = 1:Nk]
    
    # Approximate but numerically stable formula
    G2 = [X[ik] /(I + ψ[ik]'X[ik]) - ψ[ik] * ψ[ik]'X[ik] for ik = 1:Nk]
    return G2
end

abstract type AbstractRetraction end 

@kwdef mutable struct RetractionQR <: AbstractRetraction
    R = nothing
end

function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    Nn, Nk = size(φk)
    temp = qr(φk)
    Q = convert(ArrayType, temp.Q)[1:Nn, 1:Nk]
    R = convert(ArrayType, temp.R)[1:Nk, 1:Nk]
    return (Q,R)
end


#DFTK.@timing 
function calculate_retraction(ψ, η, τ, ret_qr::RetractionQR)
    Nk = size(ψ)[1]
    Q = []
    R = []
    φ = ψ + η .* τ
    for ik = 1:Nk
        φk = φ[ik]
        Qk, Rk = ortho_qr(φk)
        push!(Q, Qk)
        push!(R, Rk)
    end
    #save this value for further calculations
    ret_qr.R = R
    return Q
end

@kwdef mutable struct RetractionPolar <: AbstractRetraction
    Sf = nothing
    Sfinv = nothing
end
#DFTK.@timing 
function calculate_retraction(ψ, η, τ, ret_pol::RetractionPolar)
    Nk = size(ψ)[1]
    Q = []
    Sf = []
    Sfinv = []
    φ = ψ + η .* τ
    for ik = 1:Nk
        φk = φ[ik]
        S = φk'φk
        s, U = eigen(S)
        Σ = broadcast(x -> sqrt(x), s)
        Σ_inv = broadcast(x -> 1.0/x, Σ)
        Sfinv_k = U * Diagonal(Σ_inv) * U'
        Qk = φk * Sfinv_k
        push!(Q, Qk)
        push!(Sf, U * Diagonal(Σ) * U')
        push!(Sfinv, Sfinv_k)
    end
    #save values for further calculations
    ret_pol.Sf = Sf
    ret_pol.Sfinv = Sfinv
    return Q
end

abstract type AbstractTransport end

default_transport() = DifferentiatedRetractionTransport()

struct DifferentiatedRetractionTransport <: AbstractTransport end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_qr::RetractionQR; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Yrf = [ξ[ik] / ret_qr.R[ik] for ik = 1:Nk]
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
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_pol::RetractionPolar ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Mtx_rhs = [ψ[ik]'ξ[ik]  for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    DSf = [lyap(ret_pol.Sf[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
    return [(ξ[ik] - ψ[ik] * DSf[ik]) * ret_pol.Sfinv[ik] for ik = 1:Nk]
end


struct L2ProjectionTransport <: AbstractTransport end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::L2ProjectionTransport,  
                            ::AbstractRetraction ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    G = [ψ[ik]'ξ[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [ξ[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
end

struct RiemannianProjectionTransport <: AbstractTransport 
    Pks_Metric
end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, riem_proj::RiemannianProjectionTransport,  
                            ::AbstractRetraction ; 
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
function H1ProjectionTransport(basis::PlaneWaveBasis{T}) where {T}
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    return RiemannianProjectionTransport(Pks)
end

struct EAProjectionTransport <: AbstractTransport
    ea_grad::EAGradient
end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ea_proj::EAProjectionTransport,  
                            ::AbstractRetraction ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    Mtx_lhs = [ψ[ik]'ea_grad.Hinv_ψ[ik] for ik = 1:Nk]
    Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
    Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
    X = [lyap(Mtx_lhs[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
    return [ξ[ik] - ea_proj.ea_grad.Hinv_ψ[ik] * X[ik] for ik = 1:Nk]
end

struct InverseRetractionTransport <: AbstractTransport end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionQR ; 
                            is_prev_dir = true)
    if (!is_prev_dir)
        throw(DomainError("inverse Retraction not implemented for non-η directions"))
    end
    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
    R_ir = [tsylv2solve(Mtx_lhs[ik]) for ik = 1:Nk]
    return [- ψ_old[ik] * R_ir[ik] + ψ[ik] for ik = 1:Nk]
end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionPolar ; 
                            is_prev_dir = true)
    if (!is_prev_dir)
        throw(DomainError("inverse Retraction not implemented for non-η directions"))
    end
    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
    S_ir = [lyap(Mtx_lhs[ik], - 2.0 * Matrix{ComplexF64}(I, size(ψ)[1], size(ψ)[1])) for ik = 1:Nk]
    return [- ψ_old[ik] * S_ir[ik] + ψ[ik] for ik = 1:Nk]
end

struct NoTransport <: AbstractTransport end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::NoTransport,  
                            ::AbstractRetraction ; 
                            is_prev_dir = true)
    return ξ;
end

abstract type AbstractCGParam end

struct ParamZero <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamZero)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamZero)
    return 0.0
end


@kwdef mutable struct ParamHS <: AbstractCGParam 
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHS)
    param.grad_old = grad
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHS)
    Nk = size(gamma)[1]
    T_grad_old = transport(param.grad_old)
    β = real(sum((gamma[ik] - dot(T_grad_old[ik],res[ik])) for ik = 1:Nk)/sum(dot(T_η_old[ik],res[ik]) - desc_old[ik] for ik = 1:Nk))

    param.grad_old = grad

    return β
end

struct ParamDY <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamDY)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamDY)
    Nk = size(gamma)[1]
    β =  real(sum(gamma[ik] for ik = 1:Nk)/sum(dot(T_η_old[ik],res[ik]) - desc_old[ik] for ik = 1:Nk))

    return β
end

@kwdef mutable struct ParamPRP <: AbstractCGParam
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamPRP)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamPRP)
    Nk = size(gamma)[1]
    β = real(sum(gamma[ik] - dot(T_η_old[ik],res[ik]) for ik = 1:Nk)/sum(param.gamma_old[ik] for ik = 1:Nk))
    
    param.gamma_old = gamma

    return β
end

@kwdef mutable struct ParamFR <: AbstractCGParam 
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamFR)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamFR)
    Nk = size(gamma)[1]
    β = real(sum(gamma[ik] for ik = 1:Nk)/sum(param.gamma_old[ik] for ik = 1:Nk))   
    
    param.gamma_old = gamma

    return β
end


@kwdef mutable struct ParamFR_PRP <: AbstractCGParam
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamFR_PRP)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamFR_PRP)
    Nk = size(gamma)[1]
    β = max(min(sum(gamma[ik]  for ik = 1:Nk), sum(gamma[ik] - real(dot(T_η_old[ik],res[ik]))  for ik = 1:Nk))/sum(param.gamma_old[ik] for ik = 1:Nk), 0)

    param.gamma_old = gamma

    return β
end

@kwdef mutable struct ParamHS_DY <: AbstractCGParam 
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHS_DY)
    param.grad_old = grad
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHS_DY)
    Nk = size(gamma)[1]
    T_grad_old = transport(param.grad_old)
    β_DY =  real(sum(gamma[ik] for ik = 1:Nk)/sum(dot(T_η_old[ik],res[ik]) - desc_old[ik] for ik = 1:Nk))
    β_HS = real(sum((gamma[ik] - dot(T_grad_old[ik],res[ik])) for ik = 1:Nk)/sum(dot(T_η_old[ik],res[ik]) - desc_old[ik] for ik = 1:Nk))

    param.grad_old = grad

    return max(min(β_HS, β_DY), 0)
end


#@kwdef and mutable structs do not work as expected
mutable struct ParamHZ <: AbstractCGParam 
    const μ
    gamma_old
    grad_old
    function ParamHZ(μ)
        return new(μ, nothing, nothing)
    end
end
    

function init_β(gamma, res, grad, param::ParamHZ)
    param.grad_old = grad
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHZ)
    Nk = size(gamma)[1]
    T_grad_old = transport(param.grad_old)
    temp1 = real(sum(dot(T_η_old[ik],res[ik]) - desc_old[ik]) for ik = 1:Nk)
    temp2 = real(sum(gamma[ik] - 2 * dot(T_grad_old[ik],res[ik]) + param.gamma_old[ik] for ik = 1:Nk)/temp1[ik] for ik = 1:Nk)
    β = real((sum(gamma[ik] - real(dot(T_grad_old[ik],res[ik])) for ik = 1:Nk) - sum(dot(T_η_old[ik],res[ik]) for ik = 1:Nk) * param.μ * temp2)/temp1)     
    
    param.grad_old = grad
    param.gamma_old = gamma

    return β
end


abstract type AbstractStepSize end

struct ExactHessianStep <: AbstractStepSize end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ExactHessianStep; next = nothing)
    Nk = size(ψ)[1]

    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [dot(η[ik],Ω_η[ik]) for ik in 1:Nk]
    K_η = DFTK.apply_K(basis, η, ψ, ρ, occupation)
    η_K_η = [dot(η[ik],K_η[ik]) for ik in 1:Nk]

    τ = min(real(- sum(desc)/sum((abs(η_Ω_η[ik] + η_K_η[ik])) for ik in 1:Nk))) 
    return τ
end


struct ApproxHessianStep <: AbstractStepSize end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ApproxHessianStep; next = nothing)
    Nk = size(ψ)[1]
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [dot(η[ik],Ω_η[ik]) for ik in 1:Nk]

    τ = real(- sum(desc)/sum(abs(η_Ω_η[ik]) for ik in 1:Nk))
    return τ
end

struct ConstantStep <: AbstractStepSize
    τ_const
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ConstantStep; next = nothing)
    return stepsize.τ_const
end

mutable struct AlternatingStep <: AbstractStepSize
    const τ_list
    iter
    function AlternatingStep(τ_list)
        return new(τ_list, -1)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::AlternatingStep; next = nothing)
    Nk = size(ψ)[1]
    τ = stepsize.τ_list[(stepsize.iter += 1)%length(stepsize.τ_list) + 1]

    return τ
end

#Note: This may only show intended behaviour for β ≡ 0, i.e. η = -grad
mutable struct BarzilaiBorweinStep <: AbstractStepSize
    const τ_min::Float64
    const τ_max::Float64
    const τ_0
    τ_old
    desc_old
    is_odd
    function BarzilaiBorweinStep(τ_min, τ_max, τ_0)
        return new(τ_min, τ_max, τ_0, nothing, nothing, false)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::BarzilaiBorweinStep; next = nothing)
    Nk = size(ψ)[1]
    if (stepsize.τ_old === nothing)
        #first step
        if (stepsize.τ_0 isa AbstractStepSize)
            τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.τ_0)
        else    
            τ = stepsize.τ_0;
        end
    else
        temp = [real(dot(T_η_old[ik],res[ik])) for ik = 1:Nk];
        if (stepsize.is_odd)
            #short step size
            p = sum(temp) - sum(stepsize.desc_old)
            q = 2 * sum(temp) - sum(desc) - sum(stepsize.desc_old)
            τ = stepsize.τ_old .* abs(p/q)
        else
            p = sum(stepsize.desc_old)
            q = sum(temp) - p
            τ = stepsize.τ_old .* abs(p/q)
        end
    end

    τ = min(max(τ, stepsize.τ_min), stepsize.τ_max);

    #update params
    stepsize.desc_old = desc;
    stepsize.is_odd = !stepsize.is_odd;
    stepsize.τ_old = τ;
    
    return τ;
end

abstract type AbstractBacktrackingRule end

default_rule() = NonmonotoneRule(0.95, 0.0001, 0.5, nothing, nothing)

@kwdef mutable struct NonmonotoneRule <: AbstractBacktrackingRule
    const α::Float64
    const β::Float64
    const δ::Float64
    q 
    c 
    function NonmonotoneRule(α::Float64, β::Float64, δ::Float64)
        return new(α, β, δ, 1, nothing)
    end
end

function check_rule(E_current, desc_current, next, rule::NonmonotoneRule)
    if (isnothing(rule.c))
        #initialize c on first step
        rule.c = E_current;
    end
    E_next = next.energies_next.total
    return E_next <= rule.c + rule.β * real(sum(next.τ .* desc_current));
end
function backtrack(τ, rule::NonmonotoneRule)
    return rule.δ * τ
end
function update_rule!(next, rule::NonmonotoneRule)
    E = next.energies_next.total
    rule.q = rule.α * rule.q + 1
    rule.c = (1-1/rule.q) * rule.c + 1/rule.q * E;
end

struct ArmijoRule <: AbstractBacktrackingRule
    β::Float64
    δ::Float64
end
function check_rule(E_current, desc_current, next, rule::ArmijoRule)
    #small correction if change in energy is small TODO: better workaround.
    E_next = next.energies_next.total
    return E_next <= E_current + (rule.β * (real(sum(next.τ .* desc_current))) + 32 * eps(Float64) * abs(E_current))
end
function backtrack(τ, rule::ArmijoRule)
    return rule.δ * τ
end
function update_rule!(next, ::ArmijoRule) end


mutable struct WolfeHZRule <: AbstractBacktrackingRule
    const c_1::Float64
    const c_2::Float64
    const δ::Float64
    τ_l
    τ_r
    τ_new
    val_l
    val_r
    function WolfeHZRule(c_1, c_2, δ)
        return new(c_1, c_2, δ, nothing, nothing, nothing, nothing, nothing)
    end
end
function check_rule(E_current, desc_current, next, rule::WolfeHZRule)
    #small correction if change in energy is small TODO: better workaround.
    E_next = next.energies_next.total
    desc_next = [dot(res_next_k, Tη_next_k) for (res_next_k, Tη_next_k) = zip(next.res_next, next.Tη_next)]
    desc_next_all = real(sum(desc_next))
    desc_curr_all = real(sum(desc_current))

    if (E_next > E_current + (rule.c_1 * desc_curr_all + 100 * eps(Float64) * abs(E_current)) && rule.c_1 != 0)
        #Armijo condition not satisfied --> reset HZ
        rule.τ_new = rule.δ * next.τ
        return false
    elseif (abs(desc_next_all) <= rule.c_2 * abs(desc_curr_all))
        #Curvature condition satisfied --> prepare for next iteration
        return true
    end



    if (isnothing(rule.τ_l))
        rule.τ_l = 0.0 * next.τ
        rule.val_l = desc_curr_all
    end
    if (isnothing(rule.τ_r) || (rule.τ_new != next.τ))
        if (desc_next_all < desc_curr_all)
            #bandaid edge case #1
            update_rule!(nothing, rule)
            rule.τ_new = (1/rule.δ) * next.τ
            return false
        end
        #todo check if this is actually positive
        rule.τ_new = (- desc_curr_all / (desc_next_all - desc_curr_all)) * next.τ

        rule.τ_r = next.τ
        rule.val_r = desc_next_all
        return false
    end

    if (rule.val_l * rule.val_r <= 0)
        #make interval smaller
        if (rule.val_l * desc_next_all > 0 && rule.τ_new > rule.τ_l )
            rule.τ_l = rule.τ_new
            rule.val_l = desc_next_all
        elseif (rule.val_l * desc_next_all <= 0 && rule.τ_new < rule.τ_r) 
            rule.τ_r = rule.τ_new
            rule.val_r = desc_next_all
        end
    else
        #extend interval into respective direction
        if (rule.τ_new > rule.τ_r)
            rule.τ_l = rule.τ_r
            rule.val_l = rule.val_r 
            rule.τ_r = rule.τ_new
            rule.val_r = desc_next_all
        elseif (rule.τ_new < rule.τ_l)
            rule.τ_r = rule.τ_l
            rule.val_r = rule.val_l 
            rule.τ_l = rule.τ_new
            rule.val_l = desc_next_all
        else
            #else: we are screwed lol
            rule.τ_l = rule.τ_new
            rule.val_l = desc_next_all
        end
    end

    if (rule.τ_r < rule.τ_l)
        rule.τ_r, rule.τ_l, rule.val_r, rule.val_l = rule.τ_l, rule.τ_r, rule.val_l, rule.val_r 
    end
    
    ω = (- rule.val_l / (rule.val_r - rule.val_l)) 
    rule.τ_new = ω * rule.τ_r + (1 - ω) * rule.τ_l

    # print("val l: "); println(rule.val_l)
    # print("val r: "); println(rule.val_r)
    # print("tau l: "); println(rule.τ_l)
    # print("tau r: "); println(rule.τ_r)
    # print("tau new: "); println(rule.τ_new)
    return false
end

function backtrack(τ, rule::WolfeHZRule)
    if (isnothing(rule.τ_new))
        return τ
    end
    return rule.τ_new
end

function update_rule!(next, rule::WolfeHZRule) 
    rule.τ_new = nothing
    rule.τ_r = nothing
    rule.val_r = nothing
    rule.τ_l = nothing
    rule.val_l = nothing
end



abstract type AbstractBacktracking end


#DFTK.@timing 
#TODO integrate the transport here
function do_step(basis, occupation, ψ, η, τ, retraction, transport)
    Nk = size(ψ)[1]

    ψ_next = calculate_retraction(ψ, η, τ, retraction)

    ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
    energies_next, H_next = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)

    Hψ_next = [H_next.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_next)]
    Λ_next = [ψ_next[ik]'Hψ_next[ik] for ik = 1:Nk]
    Λ_next = 0.5 * [(Λ_next[ik] + Λ_next[ik]') for ik = 1:Nk]
    res_next = [Hψ_next[ik] - ψ_next[ik] * Λ_next[ik] for ik = 1:Nk]

    Tη_next = calculate_transport(ψ_next, η, τ, ψ, transport, retraction; is_prev_dir = true)

    (; ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next, Tη_next, τ)
end


mutable struct StandardBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const stepsize::AbstractStepSize
    const maxiter::Int64

    function  StandardBacktracking(rule::AbstractBacktrackingRule, stepsize::AbstractStepSize, maxiter::Int64)
        return new(rule, stepsize, maxiter)
    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, transport::AbstractTransport, backtracking::StandardBacktracking)

    τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.stepsize)

    next = nothing

    for k = 0:backtracking.maxiter
        if (k != 0)
            τ = backtrack(τ, backtracking.rule)
        end
        
        next = do_step(basis, occupation, ψ, η, τ, retraction, transport)
        

        if check_rule(energies.total, desc, next, backtracking.rule)
            break
        end 
    end

    update_rule!(next, backtracking.rule);
    return next;
end


mutable struct AdaptiveBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const τ_0::AbstractStepSize
    const maxiter::Int64
    τ_old
    function AdaptiveBacktracking(rule::AbstractBacktrackingRule, stepsize::AbstractStepSize, maxiter::Int64)
        return new(rule, stepsize, maxiter, nothing)
    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, transport::AbstractTransport, backtracking::AdaptiveBacktracking)
    
    if isnothing(backtracking.τ_old)
        τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.τ_0)
    else
        τ = backtracking.τ_old
    end

    next = do_step(basis, occupation, ψ, η, τ, retraction, transport)

    for k = 1:backtracking.maxiter
        if check_rule(energies.total, desc, next, backtracking.rule)
            break
        end 
        τ = backtrack(τ, backtracking.rule)
        next = do_step(basis, occupation, ψ, η, τ, retraction, transport)
    end

    backtracking.τ_old = τ
    update_rule!(next, backtracking.rule)
    return next
end

