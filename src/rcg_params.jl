include("./lyap_solvers.jl")
include("./shifted_preconditioner.jl")

using Krylov
using DFTK

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
    return [P_Hψ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
end



struct L2Gradient <: AbstractGradient end
function calculate_gradient(ψ, Hψ, H, Λ, res, ::L2Gradient)
    return res
end

abstract type AbstractHSolver end
abstract type AbstractShiftStrategy end

mutable struct EAGradient <: AbstractGradient 
    const itmax::Int64
    const rtol::Float64
    const atol::Float64
    const Pks
    const shift::AbstractShiftStrategy
    const krylov_solver #solver for linear systems
    const h_solver::AbstractHSolver   #type of solver for H 
    Hinv_ψ

    function EAGradient(basis::PlaneWaveBasis{T}, shift; itmax = 100, rtol = 1e-1, atol = 1e-16, krylov_solver = Krylov.minres, h_solver = KpointHSolver(), Pks = nothing) where {T}
        if isnothing(Pks)
            Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        end
        return new(itmax, rtol, atol, Pks, shift, krylov_solver, h_solver, nothing)
    end
end



struct ConstantShift <: AbstractShiftStrategy 
    σ
end
function get_constant_shift(σ::Float64, Nk)
    return [σ for ik = 1:Nk]
end
function get_constant_shift(σ::AbstractArray, Nk)
    return [σ[ik] for ik = 1:Nk]
end
function calculate_shift(ψ, Hψ, H, Λ, res, shift::ConstantShift)
    Nk = size(ψ)[1]
    return get_constant_shift(shift.σ, Nk)
end

struct RelativeEigsShift <: AbstractShiftStrategy
    μ::Float64
end
function calculate_shift(ψ, Hψ, H, Λ, res, shift::RelativeEigsShift)
    Nk = size(ψ)[1] 
    λ_min = [min(real(eigen(Λ[ik]).values)...) for ik = 1:Nk]
    #println(λ_min)
    return [- λ_min[ik] * shift.μ for ik = 1:Nk]
end

struct RelativeΛShift <: AbstractShiftStrategy 
    μ::Float64
end

function calculate_shift(ψ, Hψ, H, Λ, res, shift::RelativeΛShift)
    return - shift.μ * Λ
end

#TODO: Save SVD for prec?
struct CorrectedRelativeΛShift <: AbstractShiftStrategy 
    μ::Float64
end

function calculate_shift(ψ, Hψ, H, Λ, res, shift::CorrectedRelativeΛShift)
    #correcting
    Nk = size(ψ)[1] 
    Σ = []
    for ik = 1:Nk
        s, U = eigen(Λ[ik]) #TODO cast to real may change eigvals?
        σ = broadcast(x -> real(x) < 0 ? - shift.μ * real(x) :  (shift.μ - 2) * real(x) , s)
        Σk = U * Diagonal(σ) * U'
        push!(Σ, Σk)
    end
    #return = [ λ_max[ik] < 0 ? (- shift.μ * Λ[ik]) : 0 * Λ[ik] for ik = 1:Nk]
    return Σ
end

struct TotalHSolver <: AbstractHSolver end
function solve_H(krylov_solver, ψ0, H, b, σ, itmax, inner_tols, Pks, ::TotalHSolver)
    Nk = size(ψ0)[1]
    T = Base.Float64

    pack(ψ) = copy(DFTK.reinterpret_real(DFTK.pack_ψ(ψ)))
    unpack(x) = DFTK.unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))
    unsafe_unpack(x) = DFTK.unsafe_unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ0))

    x0 = pack(ψ0)
    rhs = pack(b)

    function apply_H(x)
        ξ = unsafe_unpack(x)
        return pack([H[ik] * ξ[ik] + ξ[ik] * σ[ik]])
    end

    J = LinearMap{T}(apply_H, size(x0, 1))

    function apply_Prec(x)
        ξ = unsafe_unpack(x)
        Pξ = [ Pks[ik] \ ξ[ik] for ik = 1:Nk]
        return pack(Pξ)
    end

    M = LinearMap{T}(apply_Prec, size(x0, 1))
    
    res = krylov_solver(J, rhs, x0; M, itmax, atol, rtol)
    return unpack(res[1])
end

struct KpointHSolver <: AbstractHSolver end
function solve_H(krylov_solver, ψ0, H, b, σ, itmax, inner_tols, Pks, ::KpointHSolver)
    Nk = size(ψ0)[1]
    T = Base.Float64

    function pack(Y::AbstractMatrix)
        n_rows, n_cols = size(Y)
        x = zeros(ComplexF64, n_rows * n_cols)
        for j = 1:n_cols
            x[(j-1) * n_rows + 1: j * n_rows] .= Y[:,j]
        end
        return copy(DFTK.reinterpret_real(x))
    end

    function unpack_general(x::AbstractVector, n_rows, n_cols)
        y = DFTK.reinterpret_complex(x)
        Y = zeros(ComplexF64, (n_rows, n_cols))
        for j = 1:n_cols
           Y[:,j] .=  y[(j-1) * n_rows + 1: j * n_rows]
        end
        return Y
    end
    

    res = []
    for ik = 1:Nk

        n_rows, n_cols = size(ψ0[ik])
        unpack(x) = unpack_general(x, n_rows, n_cols)

        x0 = pack(ψ0[ik])
        rhs = pack(b[ik])

        function apply_H(x)
            Y = unpack(x)
            return pack(H[ik] * Y +  Y * σ[ik])
        end
    
        J = LinearMap{T}(apply_H, size(x0, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pks[ik] \ Y)
        end
        M = LinearMap{T}(apply_Prec, size(x0, 1))

        sol, stats = krylov_solver(J, rhs, x0; M, itmax, atol = inner_tols[ik], rtol = 1e-16)

        #println(stats.niter)

        push!(res, unpack(sol))
    end

    return res 
end


function calculate_gradient(ψ, Hψ, H, Λ, res, ea_grad::EAGradient)
    Nk = size(ψ)[1];
    σ = calculate_shift(ψ, Hψ, H, Λ, res, ea_grad.shift);

    #pass shift to the shifted preconditioner
    for ik = 1:Nk
        update_shiftedTPA!(ea_grad.Pks[ik], σ[ik]);
        #DFTK.precondprep!(ea_grad.Pks[ik], ψ[ik]); 
    end

    #println([ea_grad.Pks[ik].mean_kin for ik = 1:Nk]);


    #calculate good initial guess
    Pres = [ea_grad.Pks[ik] \ res[ik] for ik = 1:Nk]
    ψ0 = [(ψ[ik] - Pres[ik]) / (Λ[ik] + σ[ik] * I)  for ik = 1:Nk]
    #ψ0 = [(ψ[ik]) / (Λ[ik] + σ[ik] * I)  for ik = 1:Nk]

    inner_tols = [max(ea_grad.rtol * norm(res[ik]), ea_grad.atol) for ik = 1:Nk]

    ea_grad.Hinv_ψ = solve_H(ea_grad.krylov_solver, ψ0, H, ψ, σ, ea_grad.itmax, inner_tols, ea_grad.Pks, ea_grad.h_solver)
    Ga = [ψ[ik]'ea_grad.Hinv_ψ[ik] for ik = 1:Nk]
    return [ ψ[ik] - ea_grad.Hinv_ψ[ik] / Ga[ik] for ik = 1:Nk]
end

mutable struct EAPrecGradient <: AbstractGradient 
    const itmax::Int64
    const atol::Float64
    const rtol::Float64
    const Pks
    const shift::AbstractShiftStrategy
    const krylov_solver #solver for linear systems
    const h_solver::AbstractHSolver   #type of solver for H 
    #TODO Initial value calculation?
    function EAPrecGradient(itmax::Int64, shift; rtol = 1e-2, atol = 1e-16, krylov_solver = Krylov.minres, h_solver = KpointHSolver())
        Pks = [PreconditionerShiftedTPA(basis, kpt) for kpt in basis.kpoints]
        return new(itmax, rtol, atol, Pks, shift, krylov_solver, h_solver)
    end
end

function calculate_gradient(ψ, Hψ, H, Λ, res, eap_grad::EAPrecGradient)
    Nk = size(ψ)[1]
    Pks = eap_grad.Pks;
    σ = calculate_shift(ψ, Hψ, H, Λ, res, eap_grad.shift)

    #this is done twice
    #for ik = 1:length(Pks)
    #    precondprep!(Pks[ik], ψ[ik]; shift = σ[ik])
    #end

    #calculate good initial guess
    P_res = [ Pks[ik] \ res[ik] for ik = 1:Nk]
    G = [ψ[ik]'P_res[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    ψ0 = [P_res[ik] - ψ[ik] * G[ik] for ik = 1:Nk]


    Hinv_res = solve_H(eap_grad.krylov_solver, ψ0, H, res, σ, eap_grad.itmax, atol, Pks, eap_grad.h_solver)
    
    #project to tangent space?
    G = [ψ[ik]'Hinv_res[ik] for ik = 1:Nk]
    G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
    return [Hinv_res[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
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
    for ik = 1:Nk
        φk = ψ[ik] + τ[ik] * η[ik]
        Qk, Rk = ortho_qr(φk)
        push!(Q, Qk)
        push!(R, Rk)
    end
    #save this value for further calculations
    ret_qr.R = R
    return Q
end

@kwdef mutable struct RetractionPolar <: AbstractRetraction
    P = nothing
    Pinv = nothing
end
#DFTK.@timing 
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
        push!(Q, Qk)
        push!(P, U * Diagonal(σ) * U')
        push!(Pinv, Pinvk)
    end
    #save values for further calculations
    ret_pol.P = P
    ret_pol.Pinv = Pinv
    return Q
end

abstract type AbstractTransport end

default_transport() = DifferentiatedRetractionTransport()

struct DifferentiatedRetractionTransport <: AbstractTransport end
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
                            ret_qr::RetractionQR, ::AbstractGradient ; 
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
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::L2ProjectionTransport,  
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
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, riem_proj::RiemannianProjectionTransport,  
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
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::EAProjectionTransport,  
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
#DFTK.@timing 
function calculate_transport(ψ, ξ, τ, ψ_old, ::InverseRetractionTransport,  
                            ::RetractionQR, ::AbstractGradient ; 
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

mutable struct ParamRestart <: AbstractCGParam 
    param::AbstractCGParam
    restart::Int64
    iter::Int64
    function ParamRestart(param::AbstractCGParam, restart::Int64)
        return new(param, restart, 0)
    end
end
function init_β(gamma, res, grad, param::ParamRestart)
    init_β(gamma, res, grad, param.param)
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamRestart)
    if ((param.iter += 1)%param.restart == 0)
        println("restart!")
        #restart every param.restart-th step
        init_β(gamma, res, grad, param.param)
        Nk = size(gamma)[1]
        return [0.0 for ik = 1:Nk]   
    else
        return calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param.param)
    end
end

struct ParamZero <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamZero)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamZero)
    Nk = size(gamma)[1]
    return [0.0 for ik = 1:Nk]   
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
    β = [real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/(tr(T_η_old[ik]'res[ik]) - desc_old[ik])) for ik = 1:Nk]

    param.grad_old = grad

    return β
end

struct ParamDY <: AbstractCGParam end
function init_β(gamma, res, grad, ::ParamDY)
    #empty
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , ::ParamDY)
    Nk = size(gamma)[1]
    return [real(gamma[ik]/(tr(T_η_old[ik]'res[ik]) - desc_old[ik])) for ik = 1:Nk]
end

@kwdef mutable struct ParamPRP <: AbstractCGParam
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamPRP)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamPRP)
    Nk = size(gamma)[1]
    β = [real((gamma[ik] - tr(T_η_old[ik]'res[ik]))/param.gamma_old[ik]) for ik = 1:Nk]
    
    param.gamma_old = gamma

    return  β
end

@kwdef mutable struct ParamFR <: AbstractCGParam 
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamFR)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamFR)
    Nk = size(gamma)[1]
    β = [real(gamma[ik]/param.gamma_old[ik]) for ik = 1:Nk]   
    
    param.gamma_old = gamma

    return  β
end


@kwdef mutable struct ParamFR_PRP <: AbstractCGParam
    gamma_old = nothing
end
function init_β(gamma, res, grad, param::ParamFR_PRP)
    param.gamma_old = gamma
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamFR_PRP)
    Nk = size(gamma)[1]
    β = [max(min(gamma[ik], (gamma[ik] - real(tr(T_η_old[ik]'res[ik]))))/param.gamma_old[ik], 0) for ik = 1:Nk]

    param.gamma_old = gamma

    return  β
end

@kwdef mutable struct ParamHS_DY <: AbstractCGParam 
    grad_old = nothing
end
function init_β(gamma, res, grad, param::ParamHS_DY)
    param.grad_old = grad
end
function calculate_β(gamma, desc_old, res, grad, T_η_old, transport , param::ParamHS_DY)
    Nk = size(gamma)
    T_grad_old = transport(param.grad_old)
    temp = [real(tr(T_η_old[ik]'res[ik]) - desc_old[ik]) for ik = 1:Nk ]
    β = [max(min(real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/temp[ik] - desc_old[ik]), real(gamma[ik]/temp[ik])), 0) for ik = 1: Nk]

    param.grad_old = grad

    return β
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
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ExactHessianStep)
    Nk = size(ψ)[1]
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
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ApproxHessianStep)
    Nk = size(ψ)[1]
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    return [min(real(abs(desc[ik]/(η_Ω_η[ik]))), stepsize.τ_max) for ik in 1:Nk]
end
struct ConstantStep <: AbstractStepSize
    τ_const
end
function get_constant_step(τ_const::Float64, Nk)
    return [τ_const for ik = 1:Nk]
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ConstantStep)
    Nk = size(ψ)[1]
    return get_constant_step(stepsize.τ_const, Nk)
end

struct ScaledStepSize <:AbstractStepSize
    stepsize::AbstractStepSize
    s::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ScaledStepSize)
    return stepsize.s * calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.stepsize)
end

mutable struct InitScaledStepSize <:AbstractStepSize
    stepsize::AbstractStepSize
    s::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::InitScaledStepSize)
    temp = stepsize.s * calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.stepsize);
    stepsize.s = 1.0;
    return temp;
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
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::BarzilaiBorweinStep)
    Nk = size(ψ)[1]
    if (stepsize.τ_old === nothing)
        #first step
        τ = [stepsize.τ_0 for ik = 1:Nk];
    else
        temp = [real(tr(T_η_old[ik]'res[ik])) for ik = 1:Nk];
        desc_old = stepsize.desc_old;
        if (stepsize.is_odd)
            #short step size
            τ = [stepsize.τ_old[ik] * abs((- stepsize.desc_old[ik] + temp[ik]) / (-desc[ik] - stepsize.desc_old[ik] + 2 * temp[ik])) for ik = 1:Nk];
        else
            #long step size
            τ = [stepsize.τ_old[ik] * abs((- stepsize.desc_old[ik])/(- stepsize.desc_old[ik] + temp[ik])) for ik = 1:Nk];
        end
    end
    #apply min, max
    τ = [min(max(τ[ik], stepsize.τ_min), stepsize.τ_max) for ik = 1:Nk];

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

function check_rule(E_current, E_next, τ, desc, rule::NonmonotoneRule)
    if (isnothing(rule.c))
        #initialize c on first step
        rule.c = E_current;
    end
    return E_next <= rule.c + rule.β * real(sum(τ .* desc));
end
function backtrack!(τ, rule::NonmonotoneRule)
    τ .= rule.δ * τ
end
function update_rule!(E, rule::NonmonotoneRule)
    rule.q = rule.α * rule.q + 1
    rule.c = (1-1/rule.q) * rule.c + 1/rule.q * E;
end

struct ArmijoRule <: AbstractBacktrackingRule
    β::Float64
    δ::Float64
end
function check_rule(E_current, E_next, τ, desc, rule::ArmijoRule)
    #small correction if change in energy is small TODO: better workaround.
    return E_next <= E_current + (rule.β * (real(sum(τ .* desc))) + 32 * eps(Float64) * abs(E_current))
end
function backtrack!(τ, rule::ArmijoRule)
    τ .= rule.δ * τ
end
function update_rule!(E, ::ArmijoRule) end



abstract type AbstractBacktracking end


#DFTK.@timing 
function do_step(basis, occupation, ψ, η, τ, retraction)
    ψ_next = calculate_retraction(ψ, η, τ, retraction)

    ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
    energies_next, H_next = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)

    (ψ_next, ρ_next, energies_next, H_next)
end

mutable struct StandardBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const maxiter::Int64
    function  StandardBacktracking(rule::AbstractBacktrackingRule, maxiter::Int64)
        return new(rule, maxiter)
    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, stepsize::AbstractStepSize, backtracking::StandardBacktracking)

    τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize)
    
    ψ_next = nothing
    energies_next = nothing
    H_next = nothing
    ρ_next = nothing

    for k = 0:backtracking.maxiter
        if (k != 0)
            backtrack!(τ, backtracking.rule)
        end
        
        ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
        

        if check_rule(energies.total, energies_next.total, τ, desc, backtracking.rule)
            break
        end 
    end

    update_rule!(energies_next.total, backtracking.rule)
    return (;ψ_next, ρ_next, energies_next, H_next, τ)
end


mutable struct GreedyBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const maxiter::Int64
    const recalculate::Int64
    τ_old
    iter::Int64
    function  GreedyBacktracking(rule::AbstractBacktrackingRule, maxiter::Int64, recalculate::Int64)
        return new(rule, maxiter, recalculate, nothing, -1)
    end
end

# this function allows to plot how goof τ is chosen. minimum should be at 1.0
# function plot_τ_quality(f)
#     x = range(0, 2, length=50)
#     y = f.(x)
#     display(plot(x,y))
# end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, stepsize::AbstractStepSize, backtracking::GreedyBacktracking)

    
    # τ = nothing
    # function f(x)
    #     fψ_next, fρ_next, fenergies_next, fH_next = do_step(basis, occupation, ψ, η, x * τ, retraction)
    #     Nk = size(fψ_next)[1]
    #     fHψ = fH_next * fψ_next
    #     fΛ = [fψ_next[ik]'fHψ[ik] for ik = 1:Nk]
    #     fΛ = 0.5 * [(fΛ[ik] + fΛ[ik]') for ik = 1:Nk]
    #     fres = [fHψ[ik] - fψ_next[ik] * fΛ[ik] for ik = 1:Nk]
    #     lst = [tr(fres[ik]'η[ik]) for ik = 1:Nk]
    #     lst = [real(lst[ik])^2 for ik = 1:Nk]
    #     return(real(sum(lst)))
    # end

    if ((backtracking.iter += 1)%backtracking.recalculate != 0 && !isnothing(backtracking.τ_old))
        #try to reuse old step size
        τ = backtracking.τ_old

        ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
        
        if check_rule(energies.total, energies_next.total, τ, desc, backtracking.rule)
            update_rule!(energies_next.total, backtracking.rule)
            # plot_τ_quality(f)
            return (;ψ_next, ρ_next, energies_next, H_next, τ)
        end 
    end

    τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize)
    
    #reset the count if old stepsize couldnt be used
    backtracking.iter = 0;

    ψ_next = nothing
    energies_next = nothing
    H_next = nothing
    ρ_next = nothing

    for k = 0:backtracking.maxiter
        if (k != 0)
            backtrack!(τ, backtracking.rule)
        end
        
        ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
        
        if check_rule(energies.total, energies_next.total, τ, desc, backtracking.rule)
            break
        end 
    end


    # plot_τ_quality(f)

    backtracking.τ_old = τ
    #println(τ)
    update_rule!(energies_next.total, backtracking.rule)
    return (;ψ_next, ρ_next, energies_next, H_next, τ)
end

