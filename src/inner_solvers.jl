using DFTK

abstract type AbstractHSolver end

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


function solve_preconditioned_least_squares(krylov_solver, HΦ, Φ, bk, Pk, Σk, itmax, tol)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    PHΦ = Pk \ HΦ 
    PΦ = Pk \ Φ

    Pbk = Pk \ bk

    rhs = PHΦ'Pbk + (PΦ'Pbk) * Σk
    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    rhs = pack(rhs)

    A1 = PHΦ'PHΦ
    A2 = PΦ'PHΦ
    A2 = A2 + A2'
    P2 = (PΦ'PΦ)
    Σk2 = Σk * Σk
    function apply_A(x)
        Y = unpack(x)
        return pack(A1 * Y + A2 * Y * Σk + P2 * Y * Σk2)
    end

    J = LinearMap{T}(apply_A, size(rhs, 1))

    x, stats = krylov_solver(J, rhs; itmax, atol = tol, rtol = tol)

    # if (!stats.solved)
    #     print(stats)
    # end

    return unpack(x)
end

abstract type NestedInnerSolver end

mutable struct GalerkinInnerSolver <: NestedInnerSolver 
    A
    Prec
    Φ
    HΦ
    shift
    function GalerkinInnerSolver()
        return new(nothing, nothing, nothing, nothing, 0.0)
    end
    function GalerkinInnerSolver(A, Prec, Φ, HΦ, shift)
        return new(A, Prec, Φ, HΦ, shift)
    end
end

function copy_InnerSolver(InnerSolver::GalerkinInnerSolver)
    return GalerkinInnerSolver(InnerSolver.A, InnerSolver.Prec, InnerSolver.Φ, InnerSolver.HΦ, InnerSolver.shift == 0.0 ? nothing : InnerSolver.shift)
end


DFTK.@timing function solve_InnerSolver(bk, ξ0, Σk, InnerSolver::GalerkinInnerSolver)
    T = Base.Float64

    shift = 0.0
    if (!isnothing(InnerSolver.shift))
        #remedy aufbau principle
        shift =  InnerSolver.shift + eigmin(0.5 * (InnerSolver.A + InnerSolver.A'))
        shift = shift < 0 ? - 1.1 * shift : 0.0
    end

    # if (shift != 0)
    #     println(shift)
    # end

    rhs = InnerSolver.Φ'bk
    x0 = InnerSolver.Φ'ξ0


    norm_rhs = norm(rhs)

    rhs0 = rhs/ norm_rhs
    rhs0 = pack(rhs0)
    x1 = x0 / norm_rhs
    x1 = pack(x1)

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    function apply_H(x)
        Y = unpack(x)
        return pack(InnerSolver.A * Y +  Y * Σk + Y * shift)
    end

    J = LinearMap{T}(apply_H, size(rhs0, 1))

    function apply_Prec(x)
        Y = unpack(x)
        return pack(InnerSolver.Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs0, 1))

    x, stats = Krylov.minres(J, rhs0, x1; M, itmax = 100, atol = 1e-8, rtol = 1e-8)
    #if (!stats.solved)
    #    print(stats)
    #end
    x *= norm_rhs

    return InnerSolver.Φ * unpack(x)
end

# DFTK.@timing function solve_InnerSolver(bk, ξ0, Σk, InnerSolver::GalerkinInnerSolver)
#     T = Base.Float64

#     shift = 0.0
#     if (!isnothing(InnerSolver.shift))
#         #remedy aufbau principle
#         shift =  InnerSolver.shift + eigmin(0.5 * (InnerSolver.A + InnerSolver.A'))
#         shift = shift < 0 ? - 1.1 * shift : 0.0
#     end

#     # if (shift != 0)
#     #     println(shift)
#     # end

#     rhs = InnerSolver.Φ'bk
#     x0 = InnerSolver.Φ'ξ0

#     m,p = size(rhs)

#     σs, U = eigen(Σk)

#     R = rhs * U

#     Y = zeros(ComplexF64, m,p)

#     for l = 1:p
#         Y[:,l] = (InnerSolver.A + I(m) * σs[l])\R[:,l]
#     end
#     #println(size(Y))

#     X = Y * U'
#     return InnerSolver.Φ * X
# end

DFTK.@timing function update_InnerSolver(Y, HY, Pk, InnerSolver::GalerkinInnerSolver)
    add = Y - InnerSolver.Φ * (InnerSolver.Φ'Y);
    #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
    idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]
    add = add[:, idxs]

    add, R = ortho_qr(add);

    Hadd = HY[:, idxs] - InnerSolver.HΦ * (InnerSolver.Φ'Y[:, idxs])
    Hadd = Hadd /R

    A12 = InnerSolver.Φ'Hadd
    A21 = A12'
    A22 = add'Hadd
    A22 = 0.5 * (A22 + A22')
    InnerSolver.A = vcat(hcat(InnerSolver.A  , A12),
            hcat(A21, A22))
    Padd = Pk \ add
    P12 = InnerSolver.Φ'Padd
    P21 = P12'
    P22 = add'Padd
    P22 = 0.5 * (P22 + P22')
    InnerSolver.Prec = vcat(hcat(InnerSolver.Prec, P12),
                hcat(P21 , P22))

    InnerSolver.Φ = hcat(InnerSolver.Φ, add)
    InnerSolver.HΦ = hcat(InnerSolver.HΦ, Hadd)
    #InnerSolver.PΦ = hcat(InnerSolver.PΦ, Padd)

    if (!isnothing(InnerSolver.shift))
        InnerSolver.shift = eigmin(InnerSolver.A)
    end
end

DFTK.@timing function init_InnerSolver(ψk, Hψk, Σk, Pk, InnerSolver::GalerkinInnerSolver)
    InnerSolver.Φ = ψk
    InnerSolver.HΦ = Hψk 
    A = ψk'Hψk
    InnerSolver.A = 0.5 * (A' + A) 

    #make A + shift spd in vertical space (does not affect riemannian gradient but improves condition greatly!)
    m = size(ψk)[2]
    evs = real(eigen(Σk * I(m)).values)
    sσ = (max(evs..., 0) + max((-evs)..., 0) )

    #InnerSolver.A = (1.0 + sσ) * I(m)
    InnerSolver.A[1:m, 1:m] += 10 * I(m) * sσ
    PΦ = Pk \ ψk
    Prec = ψk'PΦ
    InnerSolver.Prec = 0.5 * (Prec' + Prec)
end

function is_converged_ΔΦx(Φx, Φx_old, Σk, b, InnerSolver, tol)
    return norm(Φx_old - Φx) < tol
end

function is_converged_HΔΦx(Φx, Φx_old, Σk, b, InnerSolver, tol)
    ΦΔx = Φx_old - Φx
    Δx = (InnerSolver.Φ' * ΦΔx) 
    HΦΔx = InnerSolver.HΦ * Δx + ΦΔx * Σk
    return norm(HΦΔx) < tol
end

function is_converged_res(Φx, Φx_old, Σk, b, InnerSolver, tol)
    HΦx = InnerSolver.HΦ * (InnerSolver.Φ' * Φx) + Φx * Σk
    return norm(HΦx - b) < tol
end

mutable struct NestedHSolver <: AbstractHSolver 
    Pks_inner
    InnerSolver_ks
    is_converged_InnerSolver
    calculate_Φx0::Bool
    it::Int64
    function NestedHSolver(basis::PlaneWaveBasis{T}, InnerSolver_type::DataType; is_converged_InnerSolver = is_converged_ΔΦx, calculate_Φx0 = true) where {T}
        Pks_inner = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        InnerSolver_ks = [InnerSolver_type() for kpt in basis.kpoints]
        new(Pks_inner, InnerSolver_ks, is_converged_InnerSolver, calculate_Φx0, 0)
    end
end

function solve_H(krylov_solver, H, b, Σ, ψ, Hψ, itmax, inner_tols, Pks_outer, sol::NestedHSolver)
    sol.it += 1
    Nk = size(ψ)[1]
    T = Base.Float64

    result = []
    for (ik, bk, ψk, Hψk, Σk, Pk_inner, Pk_outer, InnerSolver) = zip(1:Nk, b, ψ, Hψ, Σ, sol.Pks_inner, Pks_outer, sol.InnerSolver_ks)
        Φx0 = nothing
        if  (sol.it > 1 && sol.calculate_Φx0)
            ξ0 = bk * 0.0
            Φx0 = solve_InnerSolver(bk, ξ0, Σk, InnerSolver)
            Φx0 = norm(Φx0) == 0 ? nothing : Φx0
        end

        n_rows, n_cols = size(bk)
        unpack(x) = unpack_general(x, n_rows, n_cols)

        #initialize
        init_InnerSolver(ψk, Hψk, Σk, Pk_inner, InnerSolver)

        rhs = pack(bk)
        Φx = bk * 0.0
        Φx_old = nothing

        function apply_H(x)
            Y = unpack(x)
            HY = H[ik] * Y
            update_InnerSolver(Y, HY, Pk_inner, InnerSolver)
            return pack(HY + Y * Σk)
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pk_outer \ Y)
        end

        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        function postprocess(solver)

            ξ0 = unpack(solver.x)
            Φx_old = Φx
            Φx = solve_InnerSolver(bk, ξ0, Σk, InnerSolver)

            return sol.is_converged_InnerSolver(Φx, Φx_old, Σk, bk, InnerSolver, inner_tols[ik])     
        end

        if (!isnothing(Φx0))
            krylov_solver(J, rhs, pack(Φx0); M, itmax, atol = inner_tols[ik], rtol = 1e-16, callback = postprocess)
        else
            krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16, callback = postprocess)
        end

        if (Pk_outer isa PreconditionerInnerSolver)
            update_PreconditionerInnerSolver!(Pk_outer, InnerSolver)
            Pk_outer.Σk = Σk
        end

        push!(result, Φx)
    end

    return result
end


struct NaiveHSolver <: AbstractHSolver end
function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, inner_tols, Pks, ::NaiveHSolver)
    Nk = size(ψ)[1]
    T = Base.Float64

    res = []
    for ik = 1:Nk

        n_rows, n_cols = size(ψ[ik])
        unpack(x) = unpack_general(x, n_rows, n_cols)

        rhs = pack(b[ik])

        function apply_H(x)
            Y = unpack(x)
            return pack(H[ik] * Y +  Y * σ[ik])
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pks[ik] \ Y)
        end
        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        sol, stats = krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16)


        push!(res, unpack(sol))
    end

    return res 
end

mutable struct PreconditionerInnerSolver{T <: Real}
    InnerSolver_history
    memory::Int64
    Σk
    Pk
end

function PreconditionerInnerSolver(basis::PlaneWaveBasis{T}, kpt::Kpoint, memory::Int64) where {T}
    Pk = DFTK.PreconditionerTPA(basis, kpt) 
    PreconditionerInnerSolver{T}([], memory, nothing, Pk)
end


@views function ldiv!(Y, P::PreconditionerInnerSolver, R)
    Y0 = Y
    Y1 = copy(Y)
    Y = zeros(size(Y1))
    for InnerSolver = P.InnerSolver_history
        Y2 = InnerSolver.Φ * (InnerSolver.Φ'Y1)
        Y1 -= Y2
        ξ0 = 0.0 * Y
        Y += solve_InnerSolver(Y2, ξ0, P.Σk, InnerSolver)
    end 
    Y += P.Pk \ Y1
    
    if (real(dot(Y0, Y)) < 0)
        #Failsafe
        P.InnerSolver_history = []
        Y = P.Pk \ Y0
    end
    Y
end
ldiv!(P::PreconditionerInnerSolver, R) = ldiv!(R, P, R)
(Base.:\)(P::PreconditionerInnerSolver, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::PreconditionerInnerSolver, R)
    #TODO
    Y = P.Pk * Y

    Y
end
(Base.:*)(P::PreconditionerInnerSolver, R) = mul!(copy(R), P, R)

function update_PreconditionerInnerSolver!(P::PreconditionerInnerSolver{T}, InnerSolver) where {T}
    InnerSolver_copy = copy_InnerSolver(InnerSolver)
    pushfirst!(P.InnerSolver_history, InnerSolver_copy)
    if (length(P.InnerSolver_history) > P.memory)
        pop!(P.InnerSolver_history)
    end
end

function DFTK.precondprep!(P::PreconditionerInnerSolver, X::AbstractArray)
    DFTK.precondprep!(P.Pk, X)
end
DFTK.precondprep!(P::PreconditionerInnerSolver, ::Nothing) = 1
