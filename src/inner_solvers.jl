using DFTK

#TODO: implement initial guess 

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

struct NaiveHSolver <: AbstractHSolver 
    #TODO: put krylov solver here
end
DFTK.@timing function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, tol, Pks, nhs::NaiveHSolver; ξ0 = nothing)
    Nk = size(ψ)[1]
    T = Base.Float64

    res = []
    for ik = 1:Nk

        n_rows, n_cols = size(ψ[ik])
        unpack(x) = unpack_general(x, n_rows, n_cols)

        rhs = pack(b[ik])
        norm_rhs = norm(rhs)
        rhs = rhs / norm_rhs

        function apply_H(x)
            Y = unpack(x)
            return pack(H[ik] * Y - Y * σ[ik])
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pks[ik] \ Y)
        end
        M = LinearMap{T}(apply_Prec, size(rhs, 1))


        if (isnothing(ξ0))
            sol, stats = krylov_solver(J, rhs; M, itmax, atol = tol, verbose = 0)
        else
            ξ0k = ξ0[ik] / norm_rhs
            sol, stats = krylov_solver(J, rhs, pack(ξ0k); M, itmax, atol = tol, verbose = 0)
        end

        if (!stats.solved)
            @warn "inner solver failed to converge, lower tol or increase inner iter"
        end

        push!(res, norm_rhs * unpack(sol))
    end

    return res 
end

struct GlobalOptimalHSolver <: AbstractHSolver 
    solve_horizontal
    function GlobalOptimalHSolver(; solve_horizontal = true)
        return new(solve_horizontal)
    end
end

DFTK.@timing function solve_H(krylov_solver, H, b, Σ, ψ, Hψ, itmax, tol, Pks_outer, gos::GlobalOptimalHSolver)
    Nk = size(ψ)[1]
    
    results = Array{Matrix{ComplexF64}}(undef, Nk)
    
    #Threads.@threads 
    for (ik, Hk, bk, ψk, Σk, Pk_outer) = collect(zip(1:Nk, H.blocks, b, ψ, Σ, Pks_outer))
        result = global_optimal_solve(Hk, bk, Σk, ψk, Pk_outer, itmax, tol, gos.solve_horizontal)
        results[ik] = result
    end

    return results
end


function global_optimal_solve(Hk, bk, Σk, ψk, Pk, itmax, tol, solve_horizontal)

    norm_bk = norm(bk)
    C = bk/norm_bk

    apply_Pk = function (ξk)
        Pξk = Pk\ξk
        return solve_horizontal ? Pξk - ψk * (ψk'Pξk) : Pξk
    end

    apply_Hk = function (ξk)
        Hξk = Hk * ξk
        return solve_horizontal ? Hξk - ψk * (ψk'Hξk) : Hξk
    end

    R = C
    PR = apply_Pk(R)
    V,~ = ortho_qr(PR)
    W = apply_Hk(V)
    A = V'W
    C_loc = V'C 
    Y_0 = V'PR
    for iter = 1:itmax
        Y = solve_local_system(A, Σk, C_loc, Y_0)
        X = V*Y
        R = C - W*Y + X * Σk
        if (norm(R) < tol)
            return X * norm_bk
        end
        temp = apply_Pk(R)
        PR = temp - V * (V'temp)
        tV,~ = ortho_qr(PR) #TODO: Rank revealing QR
        tW = apply_Hk(tV)

        H12 = V'tW
        H22 = tV'tW 

        V = hcat(V, tV)
        W = hcat(W, tW)

        A = vcat(hcat(A, H12),
                 hcat(H12', H22))

        C_add = tV'C
        C_loc = vcat(C_loc, C_add)
        #TODO: make this more readable, C_add is only here for the shape
        Y_0 = vcat(Y, 0.0 * C_add)
    end
    Y = solve_local_system(A, Σk, C_loc, Y_0)
    X = V*Y
    R = C - W*Y + X * Σk
    if norm(R) >= tol
        @warn "inner solver failed to converge, lower tol or increase inner iter"
    end
    return X * norm_bk
end

struct LocalOptimalHSolver <: AbstractHSolver end

DFTK.@timing function solve_H(krylov_solver, H, b, Σ, ψ, Hψ, itmax, tol, Pks_outer, los::LocalOptimalHSolver)
    Nk = size(ψ)[1]
    
    results = Array{Matrix{ComplexF64}}(undef, Nk)
    
    #Threads.@threads 
    for (ik, Hk, bk, ψk, Σk, Pk_outer) = collect(zip(1:Nk, H.blocks, b, ψ, Σ, Pks_outer))
        result = local_optimal_solve(Hk, bk, Σk, ψk, Pk_outer, itmax, tol)
        results[ik] = result
    end

    return results
end

function build_local_space!(Ξ, HΞ)
    Q,R = ortho_qr(Ξ)
    Ξ .= Q
    HΞ .= HΞ/R
    H_loc = Ξ'HΞ
    return H_loc
end

function solve_local_system(H_loc, Σk, rhs_loc, Y_0)
    n_rows, n_cols = size(rhs_loc)

    rhs_loc = pack(rhs_loc)
    unpack(x_unpack) = unpack_general(x_unpack, n_rows, n_cols)

    apply_H = function (x)
        Y = unpack(x)
        return pack(H_loc * Y - Y * Σk)
    end

    J = LinearMap{Base.Float64}(apply_H, size(rhs_loc, 1))


    x_sol, stats = Krylov.minres(J, rhs_loc, pack(Y_0); itmax = 100, atol = 1e-8, rtol = 1e-8)

    return unpack(x_sol)
end

#solves on horizontal space
#TODO: change nomenclature so X,Y are similar to global_optimal_solve
function local_optimal_solve(Hk, bk, Σk, ψk, Pk, itmax, tol)

    norm_bk = norm(bk)
    rhs = bk/norm_bk

    apply_Pk = function (ξk)
        Pξk = Pk\ξk
        return Pξk - ψk * (ψk'Pξk)
    end

    apply_Hk = function (ξk)
        Hξk = Hk * ξk
        return Hξk - ψk * (ψk'Hξk)
    end


    ξ = apply_Pk(rhs)
    Hξ = apply_Hk(ξ)
    
    w = apply_Pk(Hξ - ξ * Σk - rhs)
    Hw = apply_Hk(w)

    Ξ = hcat(ξ, w) 
    HΞ = hcat(Hξ, Hw)
    H_loc = build_local_space!(Ξ, HΞ)

    #TODO: this can be calculated efficiently in build_local_space
    Y_0 = Ξ'ξ  

    for iter = 1:itmax
        X = solve_local_system(H_loc, Σk, Ξ'rhs, Y_0)
        ξ_old = ξ
        Hξ_old = Hξ
        ξ = Ξ * X
        Hξ = HΞ * X
        #TODO: difference?
        #ξ_diff = ξ_new - ξ
        #Hξ_diff = Hξ_new - Hξ
        r = Hξ - ξ * Σk - rhs
        if norm(r) < tol
            return ξ * norm_bk
        end
        w = apply_Pk(r)
        Hw = apply_Hk(w)

        Ξ = hcat(ξ, w, ξ_old) 
        HΞ = hcat(Hξ, Hw, Hξ_old)
        H_loc = build_local_space!(Ξ, HΞ)
        #TODO: this can be calculated efficiently in build_local_space
        Y_0 = Ξ'ξ  
    end
    X = solve_local_system(H_loc, Σk, Ξ'rhs, Y_0)
    ξ_old = ξ
    Hξ_old = Hξ
    ξ = Ξ * X
    Hξ = HΞ * X
    #TODO: difference?
    #ξ_diff = ξ_new - ξ
    #Hξ_diff = Hξ_new - Hξ
    r = Hξ - ξ * Σk - rhs
    if norm(r) >= tol
        @warn "inner solver failed to converge, lower tol or increase inner iter"
    end
    
    return ξ * norm_bk
end

