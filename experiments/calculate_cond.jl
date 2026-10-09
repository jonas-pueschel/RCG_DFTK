using LinearAlgebra

pack(ψ) = DFTK.reinterpret_real(DFTK.pack_ψ(ψ))
unpack(x) = DFTK.unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ))
unsafe_unpack(x) = DFTK.unsafe_unpack_ψ(DFTK.reinterpret_complex(x), size.(ψ))

function calculate_cond(inv_metric, basis, ψ, ρ, H, Λ, occupation; maxiter = 500, rtol = 1e-6, κtol = 1e-3, check = 10)
    function A(x)
        δψ = unsafe_unpack(x)
        Kδψ = DFTK.apply_K(basis, δψ, ψ, ρ, occupation)
        Ωδψ = DFTK.apply_Ω(δψ, ψ, H, Λ)
        temp = Ωδψ + Kδψ
        temp = [temp[ik] - ψ[ik] *(ψ[ik]'temp[ik]) for ik = 1:length(basis.kpoints)]
        return pack(temp)
    end

    function Pinv(x)
        δψ = unsafe_unpack(x)
        return pack(inv_metric(δψ))
    end
    n_bands = size(ψ[1], 2)
    B0 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
    B = pack([B0[ik] - ψ[ik] *(ψ[ik]'B0[ik]) for ik = 1:length(basis.kpoints)])

    return pcg_cond(A, Pinv, B; maxiter, rtol, κtol, check)
end

function pcg_cond(A, Pinv, B; maxiter = 500, rtol = 1e-6, κtol = 1e-3, check = 10)
    R = copy(B); Z = Pinv(R); D = copy(Z)
    rz = real(dot(R, Z)); rz0 = rz
    α = Float64[]; β = Float64[]
    κold = NaN
    lastchange = NaN
    converged = false
    breakdown = false
    tridiag_eigs() = begin
        k = length(α)
        d = [1/α[1]; [1/α[j] + β[j-1]/α[j-1] for j in 2:k]]
        e = [sqrt(β[j]) / α[j] for j in 1:k-1]
        eigvals(SymTridiagonal(d, e))
    end
    for k in 1:maxiter
        AD = A(D)
        a = rz / real(dot(D, AD))
        R .-= a .* AD
        Z = Pinv(R)
        rz_new = real(dot(R, Z))
        if rz_new <= 0
            @warn "pcg_cond: breakdown at iteration $k (inexact Pinv?); returning current estimate"
            breakdown = true
            break
        end
        b = rz_new / rz
        push!(α, a); push!(β, b)
        rz = rz_new
        if k % check == 0
            λ = tridiag_eigs()
            κ = λ[end] / λ[1]
            lastchange = abs(κ - κold) / κ
            if lastchange < κtol
                converged = true
                break
            end
            κold = κ
        end
        if sqrt(rz / rz0) < rtol
            converged = true
            break
        end
        D = Z .+ b .* D
    end
    if !converged && !breakdown
        @warn "pcg_cond: maxiter = $maxiter reached without convergence; " *
              "last relative change in κ was $(round(lastchange, sigdigits = 2)). " *
              "The estimate is a lower bound and may be too small."
    end
    λ = tridiag_eigs()
    return λ[end] / λ[1], λ[1], λ[end]
end

function inv_h1_metric(basis, ψ)
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    Nk = size(ψ)[1]
    function inv_metric(η)
        P_ψ = [ Pks[ik] \ ψ[ik] for ik in 1:Nk]
        P_η = [ Pks[ik] \ η[ik] for ik in 1:Nk]
        G1 = [ψ[ik]'P_ψ[ik] for ik in 1:Nk]
        G2 = [ψ[ik]'P_η[ik] for ik in 1:Nk]
        X = [G1[ik] \ G2[ik] for ik in 1:Nk]
        return [P_η[ik] - P_ψ[ik] * X[ik] for ik in 1:Nk]
    end
    return inv_metric
end

function inv_ea_metric(basis, ψ, Hψ, H, Λ; tol = 1e-6, itmax = 30)
    Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
    Nk = size(ψ)[1]
    h_solver = GlobalOptimalHSolver(;solve_horizontal = true)
    
    function inv_metric(η)
        return RCG_DFTK.solve_H(H, η, Λ, ψ, Hψ, itmax, tol, Pks, h_solver)
    end
    return inv_metric
end

function inv_l2_metric()
    return (x) -> x
end