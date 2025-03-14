using Krylov
using DFTK

include("./lyap_solvers.jl")
include("./inner_solvers.jl")

abstract type AbstractGradient end 

struct RiemannianGradient <: AbstractGradient 
    Pks_Metric
    horizontal
end

function H1Gradient(basis::PlaneWaveBasis{T}; horizontal = false) where {T}
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    return RiemannianGradient(Pks, horizontal)
end

function calculate_gradient(ψ, Hψ, H, Λ, res, riem_grad::RiemannianGradient)
    Nk = size(ψ)[1]
    Pks = riem_grad.Pks_Metric
    # for ik = 1:length(Pks)
    #    DFTK.precondprep!(Pks[ik], ψ[ik])
    # end

    P_ψ = [ Pks[ik] \ ψ[ik] for ik = 1:Nk]
    P_r = [ Pks[ik] \ res[ik] for ik = 1:Nk]
    G1 = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
    G2 = [ψ[ik]'P_r[ik] for ik = 1:Nk]
    if (riem_grad.horizontal)
        X = [G1[ik]\G2[ik] for ik = 1:Nk]
    else
        G2 = [G2[ik] + G2[ik]' for ik = 1:Nk]
        X = [lyap(G1[ik], -G2[ik]) for ik = 1:Nk]
    end
    g = [P_r[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
    return g
end

struct L2Gradient <: AbstractGradient end
function calculate_gradient(ψ, Hψ, H, Λ, res, ::L2Gradient)
    return res
end

struct HessianGradient <: AbstractGradient 
    basis
    tol::Float64
end
function calculate_gradient(ψ, Hψ, H, Λ, res, hg::HessianGradient)
    #TODO check the DFTK implementation

    norm_res = norm(res)

    #this already has been done somewhere else but is not passed
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = DFTK.div(model.n_electrons, n_spin * filled_occ, RoundUp)
    @assert n_bands == size(ψ[1], 2)

    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ = 1:Nk]

    return norm_res * DFTK.solve_ΩplusK(hg.basis, ψ, res/norm_res, occupation; tol=hg.tol).δψ
end

abstract type AbstractShiftStrategy end


mutable struct EAGradient <: AbstractGradient
    const basis
    const itmax::Int64
    const tol::Float64
    const Pks
    const shift::AbstractShiftStrategy
    const h_solver::AbstractHSolver   #type of solver for H 
    const naive_formula::Bool
    Hinv_ψ

    function EAGradient(basis::PlaneWaveBasis{T}, shift; itmax = 100, tol = 1e-2, h_solver = GlobalOptimalHSolver(), Pks = nothing, naive_formula = false) where {T}
        if isnothing(Pks)
            Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        end
        return new(basis, itmax, tol, Pks, shift, h_solver, naive_formula, nothing)
    end
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
DFTK.@timing function calculate_shift(ψ, Hψ, H, Λ, res, shift::ConstantShift)
    Nk = size(ψ)[1]
    return get_constant_shift(shift.Σ, Nk)
end

struct RelativeEigsShift <: AbstractShiftStrategy
    μ::Float64
end
DFTK.@timing function calculate_shift(ψ, Hψ, H, Λ, res, shift::RelativeEigsShift)
    Nk = size(ψ)[1] 
    λ_min = [min(real(eigen(Λ[ik]).values)...) for ik = 1:Nk]
    return [λ_min[ik] * shift.μ * I for ik = 1:Nk]
end

struct RelativeEigsShift2 <: AbstractShiftStrategy
    μ::Float64
end
DFTK.@timing function calculate_shift(ψ, Hψ, H, Λ, res, shift::RelativeEigsShift2)
    Nk = size(ψ)[1] 
    λ_max = [max(real(eigen(Λ[ik]).values)...) for ik = 1:Nk]
    return [λ_max[ik] * shift.μ * I for ik = 1:Nk]
end

mutable struct CorrectedRelativeΛShift <: AbstractShiftStrategy   
    μ
    recalculate_μ::Bool
    function CorrectedRelativeΛShift(;μ = nothing)
        recalculate_μ = isnothing(μ)
        return new(μ, recalculate_μ)
    end
end

DFTK.@timing function calculate_shift(ψ, Hψ, H, Λ, res, shift::CorrectedRelativeΛShift)
    #correcting
    Nk = size(ψ)[1] 
    Σ = []
    if (shift.recalculate_μ)
        shift.μ = min(norm(res), 1.0)
    end

    for ik = 1:Nk
        push!(Σ, Λ[ik] - shift.μ * I)
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

    if ea_grad.naive_formula
        # Exact formula prone to numerical errors
        ξ0 = [ψ[ik]/(Λ[ik] - Σ[ik]) for ik = 1:Nk]
        Hξ0 = [Hψ[ik]/(Λ[ik] - Σ[ik]) for ik = 1:Nk]
        X = solve_H(H, ψ, Σ, ψ, Hψ, ea_grad.itmax, ea_grad.tol, ea_grad.Pks, ea_grad.h_solver; ξ0, Hξ0)
        return [ψ[ik] -  X[ik] /(ψ[ik]'X[ik]) for ik = 1:Nk]
    else
        # Approximate but numerically stable formula
        X = solve_H(H, res, Σ, ψ, Hψ, ea_grad.itmax, ea_grad.tol, ea_grad.Pks, ea_grad.h_solver)

        Mtx = [ψ[ik]'X[ik] for ik = 1:Nk]
        return [X[ik] /(I - Mtx[ik]) - ψ[ik] * Mtx[ik] for ik = 1:Nk]
    end
end

abstract type AbstractRetraction end 

@kwdef mutable struct RetractionQR <: AbstractRetraction
    R = nothing
end


DFTK.@timing function ortho_qr(φk::ArrayType) where {ArrayType <: AbstractArray}
    Nn, Nk = size(φk)
    temp = qr(φk)
    Q = convert(ArrayType, temp.Q)[1:Nn, 1:Nk]
    R = convert(ArrayType, temp.R)[1:Nk, 1:Nk]
    return (Q,R)
end


#DFTK.@timing 
DFTK.@timing function calculate_retraction(ψ, η, τ, ret_qr::RetractionQR)
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
DFTK.@timing function calculate_retraction(ψ, η, τ, ret_pol::RetractionPolar)
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

DFTK.@timing function calculate_transport(ψ, ξ, η, τ, ψ_old, ::DifferentiatedRetractionTransport,  
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
DFTK.@timing function calculate_transport(ψ, ξ, η, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_pol::RetractionPolar ; 
                            is_prev_dir = true)
    Nk = size(ψ)[1]
    if is_prev_dir
        DSf = [τ * ret_pol.Sfinv[ik] * (ξ[ik]'ξ[ik]) for ik = 1:Nk]
    else
        Mtx_rhs = [η[ik]'ξ[ik]  for ik = 1:Nk]
        Mtx_rhs = τ * [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
        DSf = [lyap(ret_pol.Sf[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
    end
    return [(ξ[ik] - ψ[ik] * DSf[ik]) * ret_pol.Sfinv[ik] for ik = 1:Nk]
end


struct L2ProjectionTransport <: AbstractTransport end

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, ::L2ProjectionTransport,  
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

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, riem_proj::RiemannianProjectionTransport,  
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

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, ea_proj::EAProjectionTransport,  
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

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionQR ; 
                            is_prev_dir = true)
    if (!is_prev_dir)
        throw(DomainError("inverse Retraction not implemented for non-η directions"))
    end
    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
    R_ir = [tsylv2solve(Mtx_lhs[ik]) for ik = 1:Nk]
    return [- ψ_old[ik] * R_ir[ik] + ψ[ik] for ik = 1:Nk]
end

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, ::InverseRetractionTransport,  
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

DFTK.@timing function calculate_transport(ψ, ξ, η,  τ, ψ_old, ::NoTransport,  
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

mutable struct ParamRestart <: AbstractCGParam 
    cg_param
    n_restart
    n_iter 
    function ParamRestart(cg_param, n_restart)
        return new(cg_param, n_restart, 0)
    end
end

function init_β(gamma, res, grad, param::ParamRestart)
    init_β(gamma, res, grad, param.cg_param)
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamRestart)
    param.n_iter += 1
    if (param.n_iter % param.n_restart != 0)
        return calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param.cg_param)
    else
        init_β(gamma, res, grad, param.cg_param)
        return 0.0
    end
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

struct ExactHessianStep <: AbstractStepSize
    basis
    occupation
    function ExactHessianStep(basis::PlaneWaveBasis{T}) where {T}
        model = basis.model
        filled_occ = DFTK.filled_occupation(model)
        n_spin = model.n_spin_components
        n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
        occupation = [filled_occ * ones(T, n_bands) for kpt = basis.kpoints]
        return new(basis, occupation)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize::ExactHessianStep)
    Nk = size(ψ)[1]

    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [dot(η[ik],Ω_η[ik]) for ik in 1:Nk]
    K_η = DFTK.apply_K(stepsize.basis, η, ψ, ρ, stepsize.occupation)
    η_K_η = [dot(η[ik],K_η[ik]) for ik in 1:Nk]

    τ = min(real(- sum(desc)/sum((abs(η_Ω_η[ik] + η_K_η[ik])) for ik in 1:Nk))) 
    return get_next(τ)
end


struct ApproxHessianStep <: AbstractStepSize end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize::ApproxHessianStep)
    Nk = size(ψ)[1]
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [dot(η[ik],Ω_η[ik]) for ik in 1:Nk]

    τ = real(- sum(desc)/sum(abs(η_Ω_η[ik]) for ik in 1:Nk))
    return get_next(τ)
end

struct ConstantStep <: AbstractStepSize
    τ_const
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize::ConstantStep)
    τ = stepsize.τ_const
    return get_next(τ)
end

mutable struct AlternatingStep <: AbstractStepSize
    const τ_list
    iter
    function AlternatingStep(τ_list)
        return new(τ_list, -1)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize::AlternatingStep)
    τ = stepsize.τ_list[(stepsize.iter += 1)%length(stepsize.τ_list) + 1]
    return get_next(τ)
end

#Note: This may only show intended behaviour for β ≡ 0, i.e. η = -grad
mutable struct BarzilaiBorweinStep <: AbstractStepSize
    const τ_min::Float64
    const τ_max::Float64
    const τ_0::AbstractStepSize
    τ_old
    desc_old
    is_odd
    function BarzilaiBorweinStep(τ_min, τ_max, τ_0)
        return new(τ_min, τ_max, τ_0, nothing, nothing, false)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize::BarzilaiBorweinStep)
    Nk = size(ψ)[1]
    if (stepsize.τ_old === nothing)
        #first step
        next = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, stepsize.τ_0)
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
        τ = min(max(τ, stepsize.τ_min), stepsize.τ_max);
        next = get_next(τ)
    end

    #update params
    stepsize.desc_old = desc;
    stepsize.is_odd = !stepsize.is_odd;
    stepsize.τ_old = next.τ;
    
    return next;
end

abstract type AbstractBacktrackingRule end

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
function backtrack(next, rule::NonmonotoneRule)
    return rule.δ * next.τ
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
function backtrack(next, rule::ArmijoRule)
    return rule.δ * next.τ
end
function update_rule!(next, ::ArmijoRule) end

mutable struct ModifiedSecantRule <: AbstractBacktrackingRule
    const δ::Float64
    const σ::Float64
    const ϵ::Float64
    const γ::Float64
    a
    b
    slope_a
    slope_b
    val_zero
    τ_next
    function ModifiedSecantRule(δ, σ, ϵ, γ)
        return new(δ, σ, ϵ, γ, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end
function check_rule(E_current, desc_current, next, rule::ModifiedSecantRule)
    #small correction if change in energy is small TODO: better workaround.
    E_next = next.energies_next.total
    slope_next = real(sum([dot(res_next_k, Tη_next_k) for (res_next_k, Tη_next_k) = zip(next.res_next, next.Tη_next)]))
    slope_zero = real(sum(desc_current))

    check_armijo = E_next - E_current ≤ rule.δ * next.τ * slope_zero 
    check_curvature = abs(slope_next) ≤ rule.σ * abs(slope_zero)
    check_wolfe_mod = abs(slope_next) ≤ min(1 - 2 * rule.δ, rule.σ) * abs(slope_zero) 
    check_reduce = E_next ≤ (1 + rule.ϵ) * abs(E_current)

    if (check_armijo && check_curvature) || (check_wolfe_mod && check_reduce)
        return true
    end

    if (isnothing(rule.a))
        # initialize search interval
        rule.a = 0.0
        rule.slope_a = slope_zero
        rule.b = next.τ
        rule.slope_b = slope_next
        rule.val_zero = E_current
        return false
    end

    if (E_next > (1 + rule.ϵ) * rule.val_zero && slope_next < 0)
        rule.τ_next = rule.γ * rule.b
        reset_rule!(rule)
        return false
    end

    if (next.τ < rule.a)
        rule.τ_next = rule.b/rule.γ
        reset_rule!(rule)
        return false
    end

    # update interval
    c = next.τ
    slope_c = slope_next
    if (c > rule.b) #automatically rule.slope_b < 0 holds here
        # shift interval to the right
        rule.a = rule.b
        rule.slope_a = rule.slope_b
        rule.b = c
        rule.slope_b = slope_c
    elseif (rule.a ≤ c && c ≤ rule.b)
        if (slope_c < 0)
            # take right interval
            rule.a = c
            rule.slope_a = slope_c
        else
            # take left interval
            rule.b = c
            rule.slope_b = slope_c
        end
    end
    return false
end

function backtrack(next, rule::ModifiedSecantRule)
    
    if (isnothing(rule.a))
        return rule.τ_next
    end
    #secant step
    return (rule.a * rule.slope_b - rule.b * rule.slope_a)/(rule.slope_b - rule.slope_a)
end

function reset_rule!(rule::ModifiedSecantRule) 
    rule.a = nothing
    rule.b = nothing
    rule.slope_a = nothing
    rule.slope_b = nothing
    rule.val_zero = nothing
end

function update_rule!(next, rule::ModifiedSecantRule) 
    reset_rule!(rule::ModifiedSecantRule)
end

abstract type IterationStrategy end

#DFTK.@timing 
function get_next_rcg(basis, occupation, ψ, η, τ, retraction::AbstractRetraction, transport::AbstractTransport)
    Nk = size(ψ)[1]

    ψ_next = calculate_retraction(ψ, η, τ, retraction)

    ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
    energies_next, H_next = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)

    Hψ_next = [H_next.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_next)]
    Λ_next = [ψ_next[ik]'Hψ_next[ik] for ik = 1:Nk]
    Λ_next = 0.5 * [(Λ_next[ik] + Λ_next[ik]') for ik = 1:Nk]
    res_next = [Hψ_next[ik] - ψ_next[ik] * Λ_next[ik] for ik = 1:Nk]

    Tη_next = calculate_transport(ψ_next, η, η, τ, ψ, transport, retraction; is_prev_dir = true)

    (; ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next, Tη_next, τ)
end

# standard backtracking: every step, an initial stepsize 
mutable struct StandardBacktracking <: IterationStrategy 
    const rule::AbstractBacktrackingRule
    const stepsize::AbstractStepSize
    const maxiter::Int64

    function  StandardBacktracking(rule::AbstractBacktrackingRule, stepsize::AbstractStepSize, maxiter::Int64)
        return new(rule, stepsize, maxiter)
    end
end

#DFTK.@timing 
function do_step(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking::StandardBacktracking)

    next = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking.stepsize)

    for k = 0:backtracking.maxiter
        if check_rule(energies.total, desc, next, backtracking.rule)
            break
        end 
        τ = backtrack(next, backtracking.rule)
        next = get_next(τ)
        #TODO: this should maybe be solved differently, i.e. by using a pointer 
        if (isa(backtracking.stepsize, BarzilaiBorweinStep))
            #τ_old is set when calculated, thus we need to update it when backtracking
            backtracking.stepsize.τ_old = τ
        end
    end

    update_rule!(next, backtracking.rule);
    return next;
end

# do backtracking, but re-use previous step size instead of re-calculating
mutable struct AdaptiveBacktracking <: IterationStrategy 
    const rule::AbstractBacktrackingRule
    const τ_0::AbstractStepSize
    const maxiter::Int64
    τ_old
    function AdaptiveBacktracking(rule::AbstractBacktrackingRule, stepsize::AbstractStepSize, maxiter::Int64)
        return new(rule, stepsize, maxiter, nothing)
    end
end
#DFTK.@timing 
function do_step(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking::AdaptiveBacktracking)
    
    if isnothing(backtracking.τ_old)
        next = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking.τ_0)
        τ = next.τ
    else
        τ = backtracking.τ_old
        next = get_next(τ)
    end

    for k = 1:backtracking.maxiter
        if check_rule(energies.total, desc, next, backtracking.rule)
            break
        end 
        τ = backtrack(next, backtracking.rule)
        next = get_next(τ)
    end

    backtracking.τ_old = τ
    update_rule!(next, backtracking.rule)
    return next
end

struct NoBacktracking <:IterationStrategy
    stepsize::AbstractStepSize
end

DFTK.@timing function do_step(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking::NoBacktracking)
    return calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, backtracking.stepsize)
end
