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

abstract type LocalOptimalInnerSolver end

mutable struct GalerkinLOIS <: LocalOptimalInnerSolver 
    A
    Prec
    Φ
    HΦ
    shift
    function GalerkinLOIS()
        return new(nothing, nothing, nothing, nothing, 0.0)
    end
    function GalerkinLOIS(A, Prec, Φ, HΦ, shift)
        return new(A, Prec, Φ, HΦ, shift)
    end
end

function copy_LOIS(lois::GalerkinLOIS)
    return GalerkinLOIS(lois.A, lois.Prec, lois.Φ, lois.HΦ, lois.shift == 0.0 ? nothing : lois.shift)
end


DFTK.@timing function solve_LOIS(bk, ξ0, Σk, lois::GalerkinLOIS)
    T = Base.Float64

    shift = 0.0
    if (!isnothing(lois.shift))
        #remedy aufbau principle
        shift =  lois.shift + eigmin(0.5 * (lois.A + lois.A'))
        shift = shift < 0 ? - 1.1 * shift : 0.0
    end

    # if (shift != 0)
    #     println(shift)
    # end

    rhs = lois.Φ'bk
    x0 = lois.Φ'ξ0


    norm_rhs = norm(rhs)

    rhs0 = rhs/ norm_rhs
    rhs0 = pack(rhs0)
    x1 = x0 / norm_rhs
    x1 = pack(x1)

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    function apply_H(x)
        Y = unpack(x)
        return pack(lois.A * Y +  Y * Σk + Y * shift)
    end

    J = LinearMap{T}(apply_H, size(rhs0, 1))

    function apply_Prec(x)
        Y = unpack(x)
        return pack(lois.Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs0, 1))

    x, stats = Krylov.minres(J, rhs0, x1; M, itmax = 100, atol = 1e-8, rtol = 1e-8)
    #if (!stats.solved)
    #    print(stats)
    #end
    x *= norm_rhs

    return lois.Φ * unpack(x)
end

# DFTK.@timing function solve_LOIS(bk, ξ0, Σk, lois::GalerkinLOIS)
#     T = Base.Float64

#     shift = 0.0
#     if (!isnothing(lois.shift))
#         #remedy aufbau principle
#         shift =  lois.shift + eigmin(0.5 * (lois.A + lois.A'))
#         shift = shift < 0 ? - 1.1 * shift : 0.0
#     end

#     # if (shift != 0)
#     #     println(shift)
#     # end

#     rhs = lois.Φ'bk
#     x0 = lois.Φ'ξ0

#     m,p = size(rhs)

#     σs, U = eigen(Σk)

#     R = rhs * U

#     Y = zeros(ComplexF64, m,p)

#     for l = 1:p
#         Y[:,l] = (lois.A + I(m) * σs[l])\R[:,l]
#     end
#     #println(size(Y))

#     X = Y * U'
#     return lois.Φ * X
# end

DFTK.@timing function update_LOIS(Y, HY, Pk, lois::GalerkinLOIS)
    add = Y - lois.Φ * (lois.Φ'Y);
    #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
    idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]
    add = add[:, idxs]

    add, R = ortho_qr(add);

    Hadd = HY[:, idxs] - lois.HΦ * (lois.Φ'Y[:, idxs])
    Hadd = Hadd /R

    A12 = lois.Φ'Hadd
    A21 = A12'
    A22 = add'Hadd
    A22 = 0.5 * (A22 + A22')
    lois.A = vcat(hcat(lois.A  , A12),
            hcat(A21, A22))
    Padd = Pk \ add
    P12 = lois.Φ'Padd
    P21 = P12'
    P22 = add'Padd
    P22 = 0.5 * (P22 + P22')
    lois.Prec = vcat(hcat(lois.Prec, P12),
                hcat(P21 , P22))

    lois.Φ = hcat(lois.Φ, add)
    lois.HΦ = hcat(lois.HΦ, Hadd)
    #lois.PΦ = hcat(lois.PΦ, Padd)

    if (!isnothing(lois.shift))
        lois.shift = eigmin(lois.A)
    end
end

DFTK.@timing function init_LOIS(ψk, Hψk, Σk, Pk, lois::GalerkinLOIS)
    lois.Φ = ψk
    lois.HΦ = Hψk 
    A = ψk'Hψk
    lois.A = 0.5 * (A' + A) 

    #make A + shift spd in vertical space (does not affect riemannian gradient but improves condition greatly!)
    m = size(ψk)[2]
    evs = real(eigen(Σk * I(m)).values)
    sσ = (max(evs..., 0) + max((-evs)..., 0) )

    #lois.A = (1.0 + sσ) * I(m)
    lois.A[1:m, 1:m] += 10 * I(m) * sσ
    PΦ = Pk \ ψk
    Prec = ψk'PΦ
    lois.Prec = 0.5 * (Prec' + Prec)
end

function is_converged_ΔΦx(Φx, Φx_old, Σk, b, lois, tol)
    return norm(Φx_old - Φx) < tol
end

function is_converged_HΔΦx(Φx, Φx_old, Σk, b, lois, tol)
    ΦΔx = Φx_old - Φx
    Δx = (lois.Φ' * ΦΔx) 
    HΦΔx = lois.HΦ * Δx + ΦΔx * Σk
    return norm(HΦΔx) < tol
end

function is_converged_res(Φx, Φx_old, Σk, b, lois, tol)
    HΦx = lois.HΦ * (lois.Φ' * Φx) + Φx * Σk
    return norm(HΦx - b) < tol
end

mutable struct LocalOptimalHSolver <: AbstractHSolver 
    Pks_inner
    lois_ks
    is_converged_lois
    calculate_Φx0::Bool
    it::Int64
    function LocalOptimalHSolver(basis::PlaneWaveBasis{T}, lois_type::DataType; is_converged_lois = is_converged_ΔΦx, calculate_Φx0 = true) where {T}
        Pks_inner = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        lois_ks = [lois_type() for kpt in basis.kpoints]
        new(Pks_inner, lois_ks, is_converged_lois, calculate_Φx0, 0)
    end
end

function solve_H(krylov_solver, H, b, Σ, ψ, Hψ, itmax, inner_tols, Pks_outer, sol::LocalOptimalHSolver)
    sol.it += 1
    Nk = size(ψ)[1]
    T = Base.Float64

    result = []
    for (ik, bk, ψk, Hψk, Σk, Pk_inner, Pk_outer, lois) = zip(1:Nk, b, ψ, Hψ, Σ, sol.Pks_inner, Pks_outer, sol.lois_ks)
        Φx0 = nothing
        if  (sol.it > 1 && sol.calculate_Φx0)
            ξ0 = bk * 0.0
            Φx0 = solve_LOIS(bk, ξ0, Σk, lois)
            Φx0 = norm(Φx0) == 0 ? nothing : Φx0
        end

        n_rows, n_cols = size(bk)
        unpack(x) = unpack_general(x, n_rows, n_cols)

        #initialize
        init_LOIS(ψk, Hψk, Σk, Pk_inner, lois)

        rhs = pack(bk)
        Φx = bk * 0.0
        Φx_old = nothing

        function apply_H(x)
            Y = unpack(x)
            HY = H[ik] * Y
            update_LOIS(Y, HY, Pk_inner, lois)
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
            Φx = solve_LOIS(bk, ξ0, Σk, lois)

            return sol.is_converged_lois(Φx, Φx_old, Σk, bk, lois, inner_tols[ik])     
        end

        if (!isnothing(Φx0))
            krylov_solver(J, rhs, pack(Φx0); M, itmax, atol = inner_tols[ik], rtol = 1e-16, callback = postprocess)
        else
            krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16, callback = postprocess)
        end

        if (Pk_outer isa PreconditionerLOIS)
            update_PreconditionerLOIS!(Pk_outer, lois)
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

mutable struct PreconditionerLOIS{T <: Real}
    lois_history
    memory::Int64
    Σk
    Pk
end

function PreconditionerLOIS(basis::PlaneWaveBasis{T}, kpt::Kpoint, memory::Int64) where {T}
    Pk = DFTK.PreconditionerTPA(basis, kpt) 
    PreconditionerLOIS{T}([], memory, nothing, Pk)
end


@views function ldiv!(Y, P::PreconditionerLOIS, R)
    Y0 = Y
    Y1 = copy(Y)
    Y = zeros(size(Y1))
    for lois = P.lois_history
        Y2 = lois.Φ * (lois.Φ'Y1)
        Y1 -= Y2
        ξ0 = 0.0 * Y
        Y += solve_LOIS(Y2, ξ0, P.Σk, lois)
    end 
    Y += P.Pk \ Y1
    
    if (real(dot(Y0, Y)) < 0)
        #Failsafe
        P.lois_history = []
        Y = P.Pk \ Y0
    end
    Y
end
ldiv!(P::PreconditionerLOIS, R) = ldiv!(R, P, R)
(Base.:\)(P::PreconditionerLOIS, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::PreconditionerLOIS, R)
    #TODO
    Y = P.Pk * Y

    Y
end
(Base.:*)(P::PreconditionerLOIS, R) = mul!(copy(R), P, R)

function update_PreconditionerLOIS!(P::PreconditionerLOIS{T}, lois) where {T}
    lois_copy = copy_LOIS(lois)
    pushfirst!(P.lois_history, lois_copy)
    if (length(P.lois_history) > P.memory)
        pop!(P.lois_history)
    end
end

function DFTK.precondprep!(P::PreconditionerLOIS, X::AbstractArray)
    DFTK.precondprep!(P.Pk, X)
end
DFTK.precondprep!(P::PreconditionerLOIS, ::Nothing) = 1
