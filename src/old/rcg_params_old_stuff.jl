
struct LocalOptimalHSolver <: AbstractHSolver 
    levels::Int
end
function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, inner_tols, Pks, lohs::LocalOptimalHSolver)
    Nk = size(ψ)[1]
    T = Base.Float64
    
    result =  []
    for (ik, ψk, Hψk, bk, σk, Pk) = zip(1:Nk, ψ, Hψ, b, σ, Pks)

        m = size(ψk)[2]
        
        #build preconditioned krylov space
        Hk = H[ik]
        Pbk = Pk \ bk
        Hbk = Hk * bk
        HPbk = Hk * Pbk
        Hkrylov_orb = HPbk

        # Pψk = Pk \ ψk
        # HPψk = Hk * Pψk
        # Φ,R = ortho_qr(hcat(ψk, bk, Pbk, Pψk))
        # HΦ = hcat(Hψk, Hbk, HPbk, HPψk)/R

        Φ,R = ortho_qr(hcat(ψk, bk, Pbk))
        HΦ = hcat(Hψk, Hbk, HPbk)/R

        Φx = bk * 0.0

        PΦ = Pk \ Φ

        A = Φ'HΦ
        A = 0.5 * (A' + A)
        #make A spd in normal space (does not affect riemannian grad but improves condition greatly!)
        A[1:m,1:m] +=  max(real(eigen(σk * I(m)).values)..., 0.0) * I(m)
        
        rhs = Φ'bk
        x0 = rhs * 0.0;

        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)

        lvl = 1
        while true
            x = solveLocalOptimal(krylov_solver,  A, x0, rhs, Prec, σk, 100, inner_tols[ik] * 1e-4)

            Φx = Φ * x
            AΦx = HΦ * x + Φx * σk
            
            residual = AΦx - bk
            normres = norm(residual)



            #println(normres)
            
            if (normres < inner_tols[ik] || lvl >= lohs.levels || size(Hkrylov_orb)[2] == 0)    
                if (lvl == lohs.levels || size(Hkrylov_orb)[2] == 0)
                    #preemptive stop
                end
                push!(result, Φx)
                break
            end
            #add dimension to krylov space

            lvl += 1
            #extend krylov space
            krylov_orb = Pk \ Hkrylov_orb
            #krylov_orb = Pk  \ HΦ

            #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
            idxs = [i for i = 1:size(krylov_orb)[2] if norm(krylov_orb[:,i] - Φ * (Φ'krylov_orb[:,i]))/norm(krylov_orb[:,i]) > 1e-4]
            krylov_orb = krylov_orb[:, idxs]

            add = krylov_orb - Φ * (Φ'krylov_orb);
            add, R = ortho_qr(add);
            #
            #Φ,R = ortho_qr(hcat(Φ, krylov_orb))
            

            Hkrylov_orb = Hk * krylov_orb
            Hadd = Hkrylov_orb - HΦ * (Φ'krylov_orb)
            Hadd = Hadd /R

            #TODO: make these more efficient!
            m_add = length(idxs) 
            A12 = Φ'Hadd
            A21 = A12'
            A22 = add'Hadd
            A22 = 0.5 * (A22 + A22')
            A = vcat(hcat(A  , A12),
                     hcat(A21, A22))
            x0 = vcat(x, zeros((m_add, m)))
            rhs = vcat(rhs, zeros((m_add, m)))

            Padd = Pk \ add
            P12 = Φ'Padd
            P21 = P12'
            P22 = add'Padd
            P22 = 0.5 * (P22 + P22')
            Prec = vcat(hcat(Prec, P12),
                        hcat(P21 , P22))

            Φ = hcat(Φ, add)
            HΦ =  hcat(HΦ, Hadd)
            PΦ = hcat(PΦ, Padd)


            #Prec = Φ'PΦ
        end
    end

    return result 
end


#enrichment strategys (not working)

#1. enrich with res, Hres

#2. enrich with sol of prev.
Φy = nothing
#enrich space (Actualy worsens performance)
if (!isnothing(sol.A_old) && !isnothing(sol.Prec_old) && !isnothing(sol.Φ_old))
    Φ_old = sol.Φ_old[ik]
    Prec_old = sol.Prec_old[ik]
    A_old = sol.A_old[ik]
    
    rhs_enrich = Φ_old'bk
    x0_enrich = rhs_enrich .* 0.0

    #println(norm(bk - Φ_old * (rhs_enrich))/norm(bk))

    normalization = 1/norm(rhs_enrich)
    rhs_enrich *= normalization

    #TODO: Check, if rhs is even covered by the space?
    #Note: no noramlizatioion for the result necessary, as it only spans space
    y = solveLocalOptimal(krylov_solver, A_old, x0_enrich, rhs_enrich, Prec_old, σk, 500, 1e-8)
    Φy = Φ_old * y
    
    #HΦy = H[ik] * Φy

    #push!(Ys, Φy)
    #push!(HYs, HΦy)
end



mutable struct LocalOptimalSolver2 <: AbstractHSolver 
    Yks_old
    HYks_old
    memory::Int64
    function LocalOptimalSolver2(basis, memory)
        Yks_old = []
        HYks_old = []
        for kpt = basis.kpoints
            push!(Yks_old, []) 
            push!(HYks_old, []) 
        end

        return new(Yks_old, HYks_old, memory)
    end
end
function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, inner_tols, Pks, sol::LocalOptimalSolver2)
    Nk = size(ψ)[1]
    T = Base.Float64

    result = []
    for (ik, bk, ψk, Hψk, σk, Pk) = zip(1:Nk, b, ψ, Hψ, σ, Pks)
        n_rows, n_cols = size(bk)
        unpack(x) = unpack_general(x, n_rows, n_cols)
        rhs = pack(bk)

        Ys = []
        HYs = []
        
        function apply_H(x)
            Y = unpack(x)
            HY = H[ik] * Y
            push!(Ys, Y)
            push!(HYs, HY)
            return pack(HY + Y * σk)
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pk \ Y)
        end

        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        Φx0, stats = krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16)
        Φx0 = unpack(Φx0)

        Φ = ψk
        HΦ = Hψk #+ max(real(eigen(σk).values)..., 0.0) * ψk

        pushfirst!(sol.Yks_old[ik], Ys)
        pushfirst!(sol.HYks_old[ik], HYs)

        Yks_old = sol.Yks_old[ik]
        HYks_old = sol.HYks_old[ik]
        for (Ys,HYs) = zip(Yks_old, HYks_old)
            for (Y,HY) = zip(Ys,HYs)
                #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
                add = Y - Φ * (Φ'Y);
                idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]

                add = add[:, idxs]
                Y = Y[:,idxs]
                HY = HY[:,idxs]


                add, R = ortho_qr(add);

                Hadd = HY - HΦ * (Φ'Y)
                Hadd = Hadd /R

                Φ = hcat(Φ, add)
                HΦ = hcat(HΦ, Hadd)
            end
        end


        #Φ, R = ortho_qr(hcat(ψk, Ys..., Ys_old...))
        #HΦ = hcat(Hψk, HYs..., HYs_old...) / R
        #HΦ = H[ik] * Φ
        A = Φ'HΦ
        A = 0.5 * (A' + A)
        m = size(ψk)[2]
        #make A + shift spd in vertical space (does not affect riemannian gradient but improves condition greatly!)
        A[1:m, 1:m] += + max(real(eigen(σk * I(m)).values)..., 0.0) * I(m)
        PΦ = Pk \ Φ
        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)

        #get rhs and normalize
        rhs_small = Φ'bk
        normalization = 1/norm(rhs_small)
        rhs_norm = rhs_small * normalization
        #println(norm(bk - Φ * (rhs_small))/norm(bk))

        x0_norm = rhs_norm * 0.0

        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)

        x1 = solve_projected_system(krylov_solver,  A, x0_norm, rhs_norm, Prec, σk, 500, 1e-8) 
        x2 = solve_least_squares(krylov_solver, HΦ, Φ, bk * normalization, Pk, σk, 500, 1e-8)
        x3 = solve_preconditioned_least_squares(krylov_solver, HΦ, Φ, bk * normalization, Pk, σk, 500, 1e-8)
        x1 /= normalization
        x2 /= normalization
        x3 /= normalization
        Φx1 = Φ * x1
        Φx2 = Φ * x2
        Φx3 = Φ * x3
        # ΦxExact, stats = krylov_solver(J, rhs; M, itmax = 300, atol = 1e-16, rtol = 1e-16)
        # ΦxExact = unpack(ΦxExact)
        # #ΦxExact = Φ * (Φ'ΦxExact)
        # println(norm(Φx0 - ΦxExact))
        # println(norm(Φx1 - ΦxExact))
        # println(norm(Φx2 - ΦxExact))
        # println(norm(Φx3 - ΦxExact))
        # println(norm(Φ * (Φ'ΦxExact) - ΦxExact))
        # println("#######")
        # How good the small system got solved
        #residual_x = A * x + x * σk - Φ'bk

        # #How good the solution is for the big system

        # residual0 = (HΦ * Φ'Φx0 + Φx0 * σk - bk)
        # residual1 = (HΦ * x1 + Φx1 * σk - bk)
        # residual2 = (HΦ * x2 + Φx2 * σk - bk)
        # residual3 = (HΦ * x3 + Φx3 * σk - bk)


        # Aresidual0 = H[ik] * residual0 + residual0 * σk
        # Aresidual1 = H[ik] * residual1 + residual1 * σk
        # Aresidual2 = H[ik] * residual2 + residual2 * σk

        # normres0 = real(dot(Aresidual0, Aresidual0))
        # normres1 = real(dot(Aresidual1, Aresidual1))
        # normres2 = real(dot(Aresidual2, Aresidual2))

        # #println(normres1/normres2)
        # println(normres0/norm(bk))
        # println(normres1/norm(bk))
        # println(normres2/norm(bk))
        # println("####")

        if (length(sol.Yks_old[ik]) > sol.memory)
            pop!(sol.Yks_old[ik])
            pop!(sol.HYks_old[ik])
        end
        
        push!(result, Φx1)
    end

    return result

end



#stepsize = BarzilaiBorweinStep(0.1, 2.5, 1.0);
#backtracking = GreedyBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), 10, 1)
#cg_param = ParamZero();



mutable struct SimpleLocalOptimalHSolver <: AbstractHSolver 
    Pks_inner
    function SimpleLocalOptimalHSolver(basis::PlaneWaveBasis{T}) where {T}
        Pks_inner = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        new(Pks_inner)
    end
end
function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, inner_tols, Pks_outer, slohs::SimpleLocalOptimalHSolver)
    Nk = size(ψ)[1]
    T = Base.Float64

    result = []
    for (ik, bk, ψk, Hψk, σk, Pk_inner, Pk_outer) = zip(1:Nk, b, ψ, Hψ, σ, slohs.Pks_inner, Pks_outer)
        n_rows, n_cols = size(bk)
        unpack(x) = unpack_general(x, n_rows, n_cols)
        rhs = pack(bk)

        Ys = []
        HYs = []
        
        function apply_H(x)
            Y = unpack(x)
            HY = H[ik] * Y
            push!(Ys, Y)
            push!(HYs, HY)
            return pack(HY + Y * σk)
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pk_outer \ Y)
        end
        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        Φx0, stats = krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16)
        Φx0 = unpack(Φx0)


        Φ = ψk
        HΦ = Hψk

        for (Y, HY) = zip(Ys, HYs)
            #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
            add = Y - Φ * (Φ'Y);
            idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]

            add = add[:, idxs]
            Y = Y[:,idxs]
            HY = HY[:,idxs]


            add, R = ortho_qr(add);

            Hadd = HY - HΦ * (Φ'Y)
            Hadd = Hadd /R

            Φ = hcat(Φ, add)
            HΦ = hcat(HΦ, Hadd)
        end

        HY = HYs[end]
        add = HY - Φ * (Φ'HY);
        add, R = ortho_qr(add);

        println(norm(add'HΦ))

        A = Φ'HΦ
        A = 0.5 * (A' + A)
        m = size(ψk)[2]
        #make A + shift spd in vertical space (does not affect riemannian gradient but improves condition greatly!)
        A[1:m, 1:m] += + max(real(eigen(σk * I(m)).values)..., 0.0) * I(m)
        PΦ = Pk_inner \ Φ
        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)

        #get rhs and normalize
        rhs_small = Φ'bk
        normalization = 1/norm(rhs_small)
        rhs_norm = rhs_small * normalization
        #println(norm(bk - Φ * (rhs_small))/norm(bk))

        x0_norm = rhs_norm * 0.0

        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)

        x1 = solve_projected_system(krylov_solver,  A, x0_norm, rhs_norm, Prec, σk, 500, 1e-8) 
        x2 = solve_least_squares(krylov_solver, HΦ, Φ, bk * normalization, Pk_inner, σk, 500, 1e-8)
        x3 = solve_preconditioned_least_squares(krylov_solver, HΦ, Φ, bk * normalization, Pk_inner, σk, 500, 1e-8)
        x1 /= normalization
        x2 /= normalization
        x3 /= normalization

        ΦxExact, stats = krylov_solver(J, rhs; M, itmax = 300, atol = 1e-16, rtol = 1e-16)
        ΦxExact = unpack(ΦxExact)
       

        #println(norm(Φ * (Φ'ΦxExact) - Φx0) / norm(Φ * (Φ'ΦxExact)))
        println(norm(ΦxExact - Φ * x1)/ norm((ΦxExact)))
        #println(norm(Φ * (Φ'ΦxExact) - Φ * x2)/ norm(Φ * (Φ'ΦxExact)))
        #println(norm(Φ * (Φ'ΦxExact) - Φ * x3)/ norm(Φ * (Φ'ΦxExact)))
        println("#####")

        push!(result, Φ * x2)
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

        #println(stats.niter)

        push!(res, unpack(sol))
    end

    return res 
end



function solve_projected_system(krylov_solver, A, x0, rhs, Prec, σk, itmax, tol)
    T = Base.Float64

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    x0 = pack(x0)
    rhs = pack(rhs)

    function apply_H(x)
        Y = unpack(x)
        return pack(A * Y +  Y * σk)
    end

    J = LinearMap{T}(apply_H, size(rhs, 1))

    function apply_Prec(x)
        Y = unpack(x)
        return pack(Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs, 1))

    x, stats = krylov_solver(J, rhs, x0; M, itmax, atol = tol, rtol = tol)
    #if (!stats.solved)
    #    print(stats)
    #end

    return unpack(x)
end

function solve_preconditioned_least_squares(krylov_solver, HΦ, Φ, bk, Pk, σk, itmax, tol)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    PHΦ = Pk \ HΦ 
    PΦ = Pk \ Φ

    Pbk = Pk \ bk

    rhs = PHΦ'Pbk + (PΦ'Pbk) * σk
    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    rhs = pack(rhs)

    A1 = PHΦ'PHΦ
    A2 = PΦ'PHΦ
    A2 = A2 + A2'
    P2 = (PΦ'PΦ)
    σk2 = σk * σk
    function apply_A(x)
        Y = unpack(x)
        return pack(A1 * Y + A2 * Y * σk + P2 * Y * σk2)
    end

    J = LinearMap{T}(apply_A, size(rhs, 1))

    x, stats = krylov_solver(J, rhs; itmax, atol = tol, rtol = tol)

    # if (!stats.solved)
    #     print(stats)
    # end

    return unpack(x)
end

function solve_least_squares(krylov_solver, HΦ, Φ, bk, Pk, σk, itmax, tol)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    rhs = HΦ'bk + (Φ'bk) * σk
    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    rhs = pack(rhs)

    A1 = HΦ'HΦ
    A2 = Φ'HΦ
    A2 = A2 + A2'
    σk2 = σk * σk'
    function apply_A(x)
        Y = unpack(x)
        return pack(A1 * Y + A2 * Y * σk + Y * σk2)
    end

    J = LinearMap{T}(apply_A, size(rhs, 1))

    PΦ = Pk \ Φ
    Prec = PΦ'PΦ
    function apply_Prec(x)
        Y = unpack(x)
        return pack(Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs, 1))

    x, stats = krylov_solver(J, rhs; M, itmax, atol = tol, rtol = tol)
    # if (!stats.solved)
    #     print(stats)
    # end

    return unpack(x)
end

abstract type LocalOptimalInnerSolver end

struct ProjectedSystemSolver <: LocalOptimalInnerSolver end

function solve_H(krylov_solver, H, b, σ, ψ, Hψ, itmax, inner_tols, Pks_outer, sol::AdaptiveLocalOptimalHSolver)
    Nk = size(ψ)[1]
    T = Base.Float64


    result = []
    for (ik, bk, ψk, Hψk, σk, Pk_inner, Pk_outer) = zip(1:Nk, b, ψ, Hψ, σ, sol.Pks_inner, Pks_outer)

        if (Pk_outer isa LMTPA)
            prepare_LMTPA!(Pk_outer, ψk, Hψk, σk, σk)
        end

        n_rows, n_cols = size(bk)
        unpack(x) = unpack_general(x, n_rows, n_cols)

        rhs = pack(bk)

        #initialize
        Φx = 0.0 * ψk
        Φx_old = nothing
        Φ = ψk
        HΦ = Hψk 
        A = Φ'HΦ
        A = 0.5 * (A' + A) 
        m = size(ψk)[2]
        #make A + shift spd in vertical space (does not affect riemannian gradient but improves condition greatly!)
        A[1:m, 1:m] += + max(real(eigen(σk * I(m)).values)..., 0.0) * I(m)
        PΦ = Pk_inner \ Φ
        Prec = Φ'PΦ
        Prec = 0.5 * (Prec' + Prec)


        Ys = [] 
        HYs = []

        
        function apply_H(x)
            Y = unpack(x)
            HY = H[ik] * Y
            push!(Ys, Y)
            push!(HYs, HY)
            return pack(HY + Y * σk)
        end
    
        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            Y = unpack(x)
            return pack(Pk_outer \ Y)
        end

        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        function postprocess(solver)
            #update Φ,A,...

            while (length(HYs) > 0)
                HY = popfirst!(HYs);
                Y = popfirst!(Ys);

                #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
                idxs = [i for i = 1:size(Y)[2] if norm(Y[:,i] - Φ * (Φ'Y[:,i]))/norm(Y[:,i]) > 1e-4]
                Y = Y[:, idxs]
                HY = HY[:, idxs]


                add = Y - Φ * (Φ'Y);
                add, R = ortho_qr(add);

                Hadd = HY - HΦ * (Φ'Y)
                Hadd = Hadd /R

                A12 = Φ'Hadd
                A21 = A12'
                A22 = add'Hadd
                A22 = 0.5 * (A22 + A22')
                A = vcat(hcat(A  , A12),
                        hcat(A21, A22))
                Padd = Pk_inner \ add
                P12 = Φ'Padd
                P21 = P12'
                P22 = add'Padd
                P22 = 0.5 * (P22 + P22')
                Prec = vcat(hcat(Prec, P12),
                            hcat(P21 , P22))

                Φ = hcat(Φ, add)
                HΦ = hcat(HΦ, Hadd)
                PΦ = hcat(PΦ, Padd)
            end

            #get rhs and normalize
            rhs_small = Φ'bk
            normalization = 1/norm(rhs_small)
            rhs_norm = rhs_small * normalization
            #println(norm(bk - Φ * (rhs_small))/norm(bk))

            x0_small = Φ'unpack(solver.x)

            x0_norm = x0_small * normalization

            Prec = Φ'PΦ
            Prec = 0.5 * (Prec' + Prec)

            x = solve_projected_system(krylov_solver, A, x0_norm, rhs_norm, Prec, σk, 500, 1e-8) 
            x /= normalization

            #How good the small system got solved
            #residual_x = A * x + x * σk - Φ'bk

            #How good the solution is for the big system
            Φx_old = Φx
            Φx = Φ * x
            residual = HΦ * x + Φx * σk - bk

            #normres = norm(residual)

            #println(normres/norm(bk))

            if (norm(Φx_old - Φx) < inner_tols[ik])    
                return true
            end

            return false
        end

        krylov_solver(J, rhs; M, itmax, atol = inner_tols[ik], rtol = 1e-16, callback = postprocess)

        if (Pk_outer isa LMTPA)
            update_LMTPA!(Pk_outer, Ys, HYs)
        end

        push!(result, Φx)
    end



    return result
end



mutable struct LMTPA{T <: Real} <: AbstractShiftedTPA
    basis::PlaneWaveBasis
    kpt::Kpoint
    memory::Int64
    Φ
    ψ
    A
    P
    Λ
    Σ
    Ys_arr
    HYs_arr
    Pk
end

function LMTPA(basis::PlaneWaveBasis{T}, kpt::Kpoint, memory::Int64) where {T}
    Pk = DFTK.PreconditionerTPA(basis, kpt) 
    LMTPA{T}(basis, kpt, memory, nothing, nothing, nothing, nothing, nothing, nothing, [], [], Pk)
end


@views function ldiv!(Y, P::LMTPA, R)
    Y0 = Y
    if (isnothing(P.Φ))
        Y = P.Pk \ Y
    else
        Y1 = (P.Φ'Y)
        Y2 = Y - P.Φ * Y1

        #println(norm(Y1)/norm(Y))


        T = Base.Float64
        n_rows, n_cols = size(Y1)
        unpack(x) = unpack_general(x, n_rows, n_cols)


        rhs = pack(Y1)

        function apply_H(x)
            X = unpack(x)
            return pack(P.A * X +  X * P.Σ)
        end

        J = LinearMap{T}(apply_H, size(rhs, 1))

        function apply_Prec(x)
            X = unpack(x)
            return pack(P.P * X)
        end
        M = LinearMap{T}(apply_Prec, size(rhs, 1))

        x, stats = Krylov.minres(J, rhs; M, itmax = 100, rtol = 1e-8, atol = 1e-16)
        #if (!stats.solved)
        #    print(stats)
        #end
        PY1 = P.Φ * unpack(x)
        PY2 = P.Pk \ Y2

        Y =  PY1 + PY2
        if (real(dot(Y0,Y)) < 0)
            
        end
    end

    Y
end
ldiv!(P::LMTPA, R) = ldiv!(R, P, R)
(Base.:\)(P::LMTPA, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::LMTPA, R)
    if (isnothing(P.Φ))
        Y = P.Pk * Y
    else
        Y1 = P.Φ'Y
        Y2 = Y - P.Φ * Y1

        m = size(Y,2)

        PY1 = P.Φ * (P.A * Y1 + Y1 * P.Σ)
        PY2 = P.Pk * Y2

        Y = PY1 + PY2
    end
    Y
end
(Base.:*)(P::LMTPA, R) = mul!(copy(R), P, R)

function prepare_LMTPA!(P::LMTPA{T}, ψ, Hψ, Λ, Σ) where {T}
    m = size(ψ, 2)
    P.ψ = ψ
    P.Λ = Λ


    shift = -1.1 * min(eigen(Λ).values..., 0)
    P.Σ = shift * I(m)
    Φ = ψ
    HΦ = Hψ


    for (Ys, HYs) = zip(P.Ys_arr, P.HYs_arr)
        Φ= hcat(Φ, Ys...)
        HΦ = hcat(HΦ, HYs...)
    end

    #println(eigen(Λ).values)

    P.Φ = Φ
    P.A = Φ'*HΦ
    P.A = 0.5 * (P.A + P.A')

    #P.A[1:m, 1:m] += shift * I(m)
    PΦ = P.Pk \ P.Φ
    P.P = Φ'PΦ
end

function update_LMTPA!(P::LMTPA{T}, Ys, HYs) where {T}
    pushfirst!(P.Ys_arr, Ys)
    pushfirst!(P.HYs_arr, HYs)

    if (length(P.Ys_arr) > P.memory)
        pop!(P.Ys_arr)
        pop!(P.HYs_arr)
    end
end

function DFTK.precondprep!(P::LMTPA, X::AbstractArray)

end
DFTK.precondprep!(P::LMTPA, ::Nothing) = 1


mutable struct LMTPA2{T <: Real} <: AbstractShiftedTPA
    basis::PlaneWaveBasis
    kpt::Kpoint
    Φ
    HΦ
    A
    Σ
    Ys
    HYs
    Pk
end

function LMTPA2(basis::PlaneWaveBasis{T}, kpt::Kpoint) where {T}
    Pk = DFTK.PreconditionerTPA(basis, kpt) 
    LMTPA2{T}(basis, kpt, nothing, nothing, nothing, nothing, [], [], Pk)
end


@views function ldiv!(Y, P::LMTPA2, R)
    if (isnothing(P.Φ))
        Y = P.Pk \ Y
    else
        Y1 = (P.Φ'Y)
        Y2 = Y - P.Φ * Y1

        #println(norm(Y1)/norm(Y))


        T = Base.Float64
        n_rows, n_cols = size(Y1)
        unpack(x) = unpack_general(x, n_rows, n_cols)


        rhs = pack(Y1)

        function apply_H(x)
            X = unpack(x)
            return pack(P.A * X +  X * P.Σ)
        end

        J = LinearMap{T}(apply_H, size(rhs, 1))

        # function apply_Prec(x)
        #     X = unpack(x)
        #     return pack(P.P * X)
        # end
        # M = LinearMap{T}(apply_Prec, size(rhs, 1))

        x, stats = Krylov.minres(J, rhs; itmax = 1000, rtol = 1e-8, atol = 1e-16)
        if (!stats.solved)
           print(stats)
        end
        PY1 = P.Φ * unpack(x)
        PY2 = P.Pk \ Y2

        Y =  PY1 + PY2
    end
    Y
end
ldiv!(P::LMTPA2, R) = ldiv!(R, P, R)
(Base.:\)(P::LMTPA2, R) = ldiv!(P, copy(R))

# These are needed by eg direct minimization with CG
@views function mul!(Y, P::LMTPA2, R)
    if (isnothing(P.Φ))
        Y = P.Pk * Y
    else
        Y1 = P.Φ'Y
        Y2 = Y - P.Φ * Y1

        PY1 = P.Φ * (P.σ .* (Y1))
        PY2 = P.Pk * Y2

        Y = PY1 + PY2
    end
    Y
end
(Base.:*)(P::LMTPA2, R) = mul!(copy(R), P, R)

function prepare_LMTPA2!(P::LMTPA2{T}, ψ, H, Λ, Σ) where {T}
    P.Σ = Σ

    shift = max(real(eigen(Σ).values)...) - min(real(eigen(Σ).values)...)

    Φ = zeros(size(ψ,1),0)
    HΦ = zeros(size(ψ,1),0)

    for (Y, HY) = zip(P.Ys, P.HYs)
        #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
        add = Y - Φ * (Φ'Y);
        idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-8]

        add = add[:, idxs]
        Y = Y[:,idxs]
        HY = HY[:,idxs]


        add, R = ortho_qr(add);

        Hadd = HY - HΦ * (Φ'Y)
        Hadd = Hadd /R

        Φ = hcat(Φ, add)
        HΦ = hcat(HΦ, Hadd)
    end


    P.Φ = Φ
    P.HΦ = HΦ


    A = P.Φ'P.HΦ

    P.A = 0.5 * (A' + A)

    P.A += (Φ'ψ) * (ψ'Φ) * shift
end

function update_LMTPA2!(P::LMTPA2{T}, Ys, HYs) where {T}
    P.Ys = Ys
    P.HYs = HYs
end

function DFTK.precondprep!(P::LMTPA2, X::AbstractArray)
    DFTK.precondprep(P.Pk, X)
end
DFTK.precondprep!(P::LMTPA2, ::Nothing) = 1


function solve_projected_system(krylov_solver, A, x0, rhs, Prec, Σk, itmax, tol)
    T = Base.Float64

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    x0 = pack(x0)
    rhs = pack(rhs)

    function apply_H(x)
        Y = unpack(x)
        return pack(A * Y +  Y * Σk)
    end

    J = LinearMap{T}(apply_H, size(rhs, 1))

    function apply_Prec(x)
        Y = unpack(x)
        return pack(Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs, 1))

    x, stats = krylov_solver(J, rhs, x0; M, itmax, atol = tol, rtol = tol)
    #if (!stats.solved)
    #    print(stats)
    #end

    return unpack(x)
end


function solve_least_squares(krylov_solver, HΦ, Φ, bk, PΦ, Σk, itmax, tol)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    rhs = HΦ'bk + (Φ'bk) * Σk
    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    rhs = pack(rhs)

    A1 = HΦ'HΦ
    A2 = Φ'HΦ
    A2 = A2 + A2'
    Σk2 = Σk * Σk'
    function apply_A(x)
        Y = unpack(x)
        return pack(A1 * Y + A2 * Y * Σk + Y * Σk2)
    end

    J = LinearMap{T}(apply_A, size(rhs, 1))

    #PΦ = Pk \ Φ
    Prec = PΦ'PΦ
    function apply_Prec(x)
        Y = unpack(x)
        return pack(Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs, 1))

    x, stats = krylov_solver(J, rhs; M, itmax, atol = tol, rtol = tol)
    # if (!stats.solved)
    #     print(stats)
    # end

    return unpack(x)
end

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

mutable struct QuadraticInterpolationBacktracking <: AbstractBacktracking 
    const c_1::Float64
    const c_2::Float64
    const maxiter::Int64
    const τ_0
    τ_old

    function  QuadraticInterpolationBacktracking(c_1::Float64, c_2::Float64, maxiter::Int64, τ_0)
        return new(c_1, c_2, maxiter, τ_0, nothing)
    end
end

# this function allows to plot how goof τ is chosen. minimum should be at 1.0
# function plot_τ_quality(f)
#     x = range(0, 2, length=50)
#     y = f.(x)
#     display(plot(x,y))
# end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, backtracking::QuadraticInterpolationBacktracking)
    Nk = size(ψ)[1]

    if (backtracking.τ_old === nothing)
        #first step
        if (backtracking.τ_0 isa AbstractStepSize)
            τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.τ_0)
        else    
            τ = [backtracking.τ_0 for ik = 1:Nk];
        end
    else
        τ = backtracking.τ_old
    end

    ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)

    for k = 1:backtracking.maxiter


        Hψ_next = [H_next.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_next)]
        Λ_next = [ψ_next[ik]'Hψ_next[ik] for ik = 1:Nk]
        Λ_next = 0.5 * [(Λ_next[ik] + Λ_next[ik]') for ik = 1:Nk]
        res_next = [Hψ_next[ik] - ψ_next[ik] * Λ_next[ik] for ik = 1:Nk]

        #TODO transport η
        desc_next = [dot(res_next[ik], η[ik]) for ik = 1:Nk]

        #check generalized curvature
        desc_all = real(sum(τ .* desc))
        desc_next_all = real(sum(τ .* desc_next))

        if (desc_next_all <= desc_all)
            # parabola opened in the wrong direction --> step size is far too big or energy functional is concave 
            # i.e. we are far away from a local minimum

            #TODO check concave in order to catch infinite loop?

            #TODO shrinking parameter?
            τ[ik] /= 2 
            ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
            continue
        elseif (abs(desc_next_all) > abs(desc_all) * backtracking.c_2)
            b = real(desc_all);
            two_a = real(desc_next_all) - b;

            #replace τ[ik] by the location of the extremum of the quadratic interpolation
            τ *= (-b/two_a);

            # for ik = 1:Nk
            #     if (abs(real(desc_next[ik])) > abs(real(desc[ik])) * backtracking.c_2)
            #         # b = real(desc[ik]);
            #         # two_a = real(desc_next[ik]) - b;

            #         #replace τ[ik] by the location of the extremum of the quadratic interpolation
            #         τ[ik] *= (-b/two_a);
            #     end
            # end
            
            ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
            continue
        end


        # continue_flag = false
        # desc_all = real(sum(τ .* desc))
        # desc_next_all = real(sum(τ .* desc_next))

        # if (abs(desc_next_all) > abs(desc_all) * (backtracking.c_2 + 32 * eps(Float64) ) ||
        #     energies_next.total > energies.total + (backtracking.c_1 * desc_all + 32 * eps(Float64) * abs(energies.total)))
        #     #println("yes")
        #     for ik = 1:Nk
        #         if (real(desc_next[ik]) <= real(desc[ik]))
        #             # parabola opened in the wrong direction --> step size is far too big or energy functional is concave 
        #             # i.e. we are far away from a local minimum

        #             #TODO check concave in order to catch infinite loop?

        #             #TODO shrinking parameter?
        #             τ[ik] /= 2 
        #             continue_flag = true
        #         else #if (abs(real(desc_next[ik])) > abs(real(desc[ik])) * backtracking.c_2)
        #             b = real(desc[ik]);
        #             two_a = real(desc_next[ik]) - b;

        #             #replace τ[ik] by the location of the extremum of the quadratic interpolation
        #             τ[ik] *= (-b/two_a);

        #             continue_flag = true
        #         end
        #     end
        #     if (continue_flag)
        #         ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
        #         continue
        #     end
        # else 
        #     #println("no")
        # end

        break;
    end

    backtracking.τ_old = τ
    return (;ψ_next, ρ_next, energies_next, H_next, τ)
end


mutable struct QuadraticInterpolationBacktracking <: AbstractBacktracking 
    const c_1::Float64
    const c_2::Float64
    const maxiter::Int64
    const τ_0
    τ_old

    function  QuadraticInterpolationBacktracking(c_1::Float64, c_2::Float64, maxiter::Int64, τ_0)
        return new(c_1, c_2, maxiter, τ_0, nothing)
    end
end

# this function allows to plot how goof τ is chosen. minimum should be at 1.0
# function plot_τ_quality(f)
#     x = range(0, 2, length=50)
#     y = f.(x)
#     display(plot(x,y))
# end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, backtracking::QuadraticInterpolationBacktracking)
    Nk = size(ψ)[1]

    if (backtracking.τ_old === nothing)
        #first step
        if (backtracking.τ_0 isa AbstractStepSize)
            τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.τ_0)
        else    
            τ = [backtracking.τ_0 for ik = 1:Nk];
        end
    else
        τ = backtracking.τ_old
    end

    ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)

    for k = 1:backtracking.maxiter


        Hψ_next = [H_next.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_next)]
        Λ_next = [ψ_next[ik]'Hψ_next[ik] for ik = 1:Nk]
        Λ_next = 0.5 * [(Λ_next[ik] + Λ_next[ik]') for ik = 1:Nk]
        res_next = [Hψ_next[ik] - ψ_next[ik] * Λ_next[ik] for ik = 1:Nk]

        #TODO transport η
        desc_next = [dot(res_next[ik], η[ik]) for ik = 1:Nk]

        #check generalized curvature
        desc_all = real(sum(τ .* desc))
        desc_next_all = real(sum(τ .* desc_next))

        if (desc_next_all <= desc_all)
            # parabola opened in the wrong direction --> step size is far too big or energy functional is concave 
            # i.e. we are far away from a local minimum

            #TODO check concave in order to catch infinite loop?

            #TODO shrinking parameter?
            τ[ik] /= 2 
            ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
            continue
        elseif (abs(desc_next_all) > abs(desc_all) * backtracking.c_2)
            b = real(desc_all);
            two_a = real(desc_next_all) - b;

            # #replace τ by the location of the extremum of the quadratic interpolation
            τ *= (-b/two_a);

            ψ_next, ρ_next, energies_next, H_next = do_step(basis, occupation, ψ, η, τ, retraction)
            continue
        end

        break;
    end

    backtracking.τ_old = τ
    return (;ψ_next, ρ_next, energies_next, H_next, τ)
end

# this function allows to plot how good τ is chosen. minimum should be at 1.0
# function plot_τ_quality(f)
#     x = range(0, 2, length=50)
#     y = f.(x)
#     display(plot(x,y))
# end

# τ = nothing
# function f(x)
#     fψ_next, fρ_next, fenergies_next, fH_next, Hψ_next, Λ_next, res_next = do_step(basis, occupation, ψ, η, x * τ, retraction)
#     Nk = size(fψ_next)[1]
#     fHψ = fH_next * fψ_next
#     fΛ = [fψ_next[ik]'fHψ[ik] for ik = 1:Nk]
#     fΛ = 0.5 * [(fΛ[ik] + fΛ[ik]') for ik = 1:Nk]
#     fres = [fHψ[ik] - fψ_next[ik] * fΛ[ik] for ik = 1:Nk]
#     lst = [tr(fres[ik]'η[ik]) for ik = 1:Nk]
#     lst = [real(lst[ik])^2 for ik = 1:Nk]
#     return(real(sum(lst)))
# end



mutable struct HagerZhangBacktracking <: AbstractBacktracking 

    function  HagerZhangBacktracking(rule::AbstractBacktrackingRule, maxiter::Int64, recalculate::Int64)

    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, backtracking::HagerZhangBacktracking)

        return (;ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next, τ)
end



mutable struct QuadraticInterpolationStepping <: AbstractBacktracking 
    const τ_0
    const recalculate::Int64
    τ_old
    iter::Int64

    function  QuadraticInterpolationStepping(τ_0, recalculate)
        return new(τ_0, recalculate, nothing, 0)
    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, backtracking::QuadraticInterpolationStepping)
    Nk = size(ψ)[1]

    if (backtracking.τ_old === nothing || (backtracking.iter+=1)%backtracking.recalculate == 0)
        #first step
        if (backtracking.τ_0 isa AbstractStepSize)
            τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.τ_0)
        else    
            τ = [backtracking.τ_0 for ik = 1:Nk];
        end
    else
        τ = backtracking.τ_old
    end

    ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next = do_step(basis, occupation, ψ, η, τ, retraction)


    Hψ_next = [H_next.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ_next)]
    Λ_next = [ψ_next[ik]'Hψ_next[ik] for ik = 1:Nk]
    Λ_next = 0.5 * [(Λ_next[ik] + Λ_next[ik]') for ik = 1:Nk]
    res_next = [Hψ_next[ik] - ψ_next[ik] * Λ_next[ik] for ik = 1:Nk]

    #TODO transport η
    desc_next = [dot(res_next[ik], η[ik]) for ik = 1:Nk]

    #check generalized curvature
    desc_all = real(sum(τ .* desc))
    desc_next_all = real(sum(τ .* desc_next))

    if (abs(desc_next_all) > abs(desc_all) * 0.00)
        b = desc_all;
        two_a = desc_next_all - b;

        # replace τ by the location of the extremum of the quadratic interpolation
        τ *= (-b/two_a);

        ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next = do_step(basis, occupation, ψ, η, τ, retraction)
    end
    backtracking.τ_old = τ
    return (;ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next, τ)
end

struct ScaledStepSize <:AbstractStepSize
    stepsize::AbstractStepSize
    s::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ScaledStepSize; next = nothing)
    return stepsize.s * calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.stepsize)
end

mutable struct InitScaledStepSize <:AbstractStepSize
    stepsize::AbstractStepSize
    s::Float64
end
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::InitScaledStepSize; next = nothing)
    temp = stepsize.s * calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.stepsize);
    stepsize.s = 1.0;
    return temp;
end

# backtracking = GreedyBacktracking(
#         GeneralizedWolfeRule(0.1, 0.2),
#         #SecantStep(ApproxHessianStep(2.5; componentwise = true)), 
#         ExactHessianStep(2.5; componentwise = true),
#         0, 5);


struct ExactHessianStep <: AbstractStepSize 
    τ_max::Float64
    componentwise::Bool
    function ExactHessianStep(τ_max; componentwise = true)
        return new(τ_max, componentwise)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ExactHessianStep; next = nothing)
    Nk = size(ψ)[1]
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    K_η = DFTK.apply_K(basis, η, ψ, ρ, occupation)
    η_K_η = [tr(η[ik]'K_η[ik]) for ik in 1:Nk]
    #return [real(η_Ω_η[ik] + η_K_η[ik]) > 0 ? real(- desc[ik]/(η_Ω_η[ik] + η_K_η[ik])) : stepsize.τ_max for ik in 1:Nk]
    if stepsize.componentwise
        return [min(real(- desc[ik]/abs(η_Ω_η[ik] + η_K_η[ik])) , stepsize.τ_max) for ik in 1:Nk]
    else
        τ = min(real(- sum(desc)/sum(abs(η_Ω_η[ik] + η_K_η[ik]) for ik in 1:Nk)) , stepsize.τ_max) 
        return [τ for ik in 1:Nk]
    end
end


# function backtrack!(τ, rule::GeneralizedWolfeRule)
#     if (isnothing(rule.desc_curr))
#         return
#     end
#     Nk = length(τ)

#     println(τ)

#     ω = [- τ[ik] * real(rule.desc_curr[ik]) / real(rule.desc_next[ik] - rule.desc_curr[ik]) for ik = 1:Nk]

#     a = max(ω...)
#     b = min(ω...)

#     α = rule.α

#     if (a <= α * b)
#         τ .= ω
#     end 
    
#     if (α <= 1.0 )
#         ω .= 1.0
#     else
#         #τ0 ./= norm(τ0)
#         c = (a - α * b)/(α - 1)
#         ω .+= c
#     end


#     desc_next_all = real(sum(τ .* rule.desc_next))
#     desc_curr_all = real(sum(τ .* rule.desc_curr))

#     q = [abs((rule.desc_next[ik] - rule.desc_curr[ik])/τ[ik]) for ik = 1:Nk]


#     τ0 = min(real(- sum(ω .* rule.desc_curr)/sum(ω[ik] * ω[ik] * q[ik] for ik = 1:Nk))) 

#     τ .= τ0 * ω
#     println(τ)
# end

# function backtrack!(τ, rule::GeneralizedWolfeRule)
#     if (isnothing(rule.desc_curr))
#         return
#     end

#     if (rule.α != 1.0)
#         for ik = 1:length(τ)
#             α = - real(rule.desc_curr[ik]) / real(rule.desc_next[ik] - rule.desc_curr[ik])
#             τ[ik] *= α
#         end
#     else
#         desc_next_all = real(sum(τ .* rule.desc_next))
#         desc_curr_all = real(sum(τ .* rule.desc_curr))

#         α = - desc_curr_all / (desc_next_all - desc_curr_all)

#         τ .*= α
#     end
# end

mutable struct NewtonDirection <: AbstractGradient
    const basis
    const itmax::Int64
    const rtol::Float64
    const atol::Float64
    const Pks
    const krylov_solver #solver for linear systems
    const h_solver::AbstractHSolver   #type of solver for H 

    function NewtonDirection(basis::PlaneWaveBasis{T}; itmax = 10, rtol = 1e-2, atol = 1e-16, krylov_solver = Krylov.minres, h_solver = AdaptiveLocalOptimalHSolver(basis, ProjectedSystemSolver), Pks = nothing) where {T}
        if isnothing(Pks)
            Pks = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        end
        return new(basis, itmax, rtol, atol, Pks, shift, krylov_solver, h_solver, nothing)
    end
end

function calculate_gradient(ψ, Hψ, H, Λ, res, newton_dir::NewtonDirection)
    Nk = size(ψ)[1];
    Σ = - Λ

    for ik = 1:Nk
       DFTK.precondprep!(newton_dir.Pks[ik], ψ[ik]); 
    end

    inner_tols = [max(newton_dir.rtol * norm(res[ik]), newton_dir.atol) for ik = 1:Nk]


    #TODO no k point split possible!?
    H_plus_K = nothing


    X = solve_H(newton_dir.krylov_solver, H_plus_K, res, Σ, ψ, Hψ, newton_dir.itmax, inner_tols, newton_dir.Pks, newton_dir.h_solver)

    return X
end


function find_τ(τ_avg, τ_cmp, α)
    τ_new(t) = (1 - t) * τ_avg + t * τ_cmp
    exc(t) = (max(τ_new(t)...)/min(τ_new(t)...)) - α

    if (exc(1) < 0 || α == 0)
        return τ_new(1)
    end
    if (exc(0) > 0 || α == 1)
        #bisection will not work here
        return exc(0) < exc(1) ? τ_new(0) : τ_new(1)
    end
    t0 = 0
    t1 = 1
    #find convex combination by bisection
    for iter = 1:20
        t_new = (t0 + t1) / 2
        if (exc(t_new) > 0)
            t1 = t_new
        else
            t0 = t_new
        end
    end
    return τ_new(t0)
end

struct ExactHessianStep <: AbstractStepSize 
    α::Float64
    use_weights::Bool
    function ExactHessianStep(α; use_weights = true)
        return new(α, use_weights)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ExactHessianStep; next = nothing)
    Nk = size(ψ)[1]

    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    K_η = DFTK.apply_K(basis, η, ψ, ρ, occupation)
    η_K_η = [tr(η[ik]'K_η[ik]) for ik in 1:Nk]

    ω = [real(- desc[ik])/abs(η_Ω_η[ik] + η_K_η[ik]) for ik in 1:Nk]
    #correct negative values in ω
    if (min(ω...) < 0)
        println("ALARM")
        ω = [abs(ωk) for ωk in ω]
    end
    #τ0 = Nk * τ0 / sum(τ0) #avg
    if (stepsize.use_weights)
        a = max(ω...)
        b = min(ω...)

        if (a <= stepsize.α * b || stepsize.α == 0 )
            return ω
        end 
        
        if (stepsize.α <= 1.0 )
            ω .= 1.0
        else
            #τ0 ./= norm(τ0)
            c = (a - stepsize.α  * b)/(stepsize.α - 1)
            ω .+= c
        end
        τ = min(real(- sum(ω .* desc)/sum(ω[ik] * ω[ik] * (abs(η_Ω_η[ik] + η_K_η[ik])) for ik in 1:Nk))) 
        return τ * ω
    else
        avg = min(real(- sum(desc)/sum( (abs(η_Ω_η[ik] + η_K_η[ik])) for ik in 1:Nk))) 
        τ_avg = [avg for ik = 1:Nk]
        τ_cmp = ω

        return find_τ(τ_avg, τ_cmp, stepsize.α)
    end
end


struct SuperExactHessianStep <: AbstractStepSize 

end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::SuperExactHessianStep; next = nothing)
    Nk = size(ψ)[1]

    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    Hess = [0.0 for ik = 1:Nk]
    for ik = 1:Nk
        ηc = [nk == ik ? ηk * 1.0 : ηk * 0.0 for (ηk, nk) = zip(η, 1:length(η))]
        K_ηc = DFTK.apply_K(basis, ηc, ψ, ρ, occupation)
        ηc_K_ηc = sum([tr(ηc[ik]'K_ηc[ik]) for ik in 1:Nk])
        Hess[ik] = real(ηc_K_ηc) 
    end
    ω = [(real(- desc[ik])/real(η_Ω_η[ik] + Hess[ik])) for ik in 1:Nk]
    println(ω)
    #ω = [1.0 for ik = 1:Nk]
    ωη = ω .* η
    K_ωη  = DFTK.apply_K(basis, ωη, ψ, ρ, occupation)
    ωη_K_ωη = [tr(ωη[ik]'K_ωη[ik]) for ik in 1:Nk]
    τ = real(sum(- ω[ik] * desc[ik] for ik = 1:Nk))/real(sum(ω[ik] * ω[ik] * η_Ω_η[ik] + ωη_K_ωη[ik] for ik in 1:Nk))
    return ω .* τ 
end


struct ApproxHessianStep <: AbstractStepSize
    α::Float64
    function ApproxHessianStep(α)
        return new(α)
    end
end
#DFTK.@timing 
function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::ApproxHessianStep; next = nothing)
    Nk = size(ψ)[1]
    Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
    η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
    ω = [real(- desc[ik]/abs(η_Ω_η[ik])) for ik in 1:Nk]
    #τ0 = Nk * τ0 / sum(τ0) #avg
    a = max(ω...)
    b = min(ω...)

    if (a <= stepsize.α * b || stepsize.α  == 0 )
        return ω
    end 
    
    if (stepsize.α  <= 1.0 )
        ω .= 1.0
    else
        #τ0 ./= norm(τ0)
        c = (a - stepsize.α * b)/(stepsize.α - 1)
        ω .+= c
    end
    τ = min(real(- sum(ω .* desc)/sum(ω[ik] * ω[ik] * (abs(η_Ω_η[ik])) for ik in 1:Nk))) 
    return τ * ω
end


#Note: This may only show intended behaviour for β ≡ 0, i.e. η = -grad
mutable struct BarzilaiBorweinStep <: AbstractStepSize
    const τ_min::Float64
    const τ_max::Float64
    const τ_0
    const α
    τ_old
    desc_old
    is_odd
    function BarzilaiBorweinStep(τ_min, τ_max, τ_0; α = 1.2)
        return new(τ_min, τ_max, τ_0, α, nothing, nothing, false)
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
            τ = [stepsize.τ_0 for ik = 1:Nk];
        end
    else
        temp = [real(tr(T_η_old[ik]'res[ik])) for ik = 1:Nk];
        if (stepsize.is_odd)
            #short step size
            τ_cmp = [stepsize.τ_old[ik] * abs((- stepsize.desc_old[ik] + temp[ik]) / (-desc[ik] - stepsize.desc_old[ik] + 2 * temp[ik])) for ik = 1:Nk];
            p = sum(temp) - sum(stepsize.desc_old)
            q = 2 * sum(temp) - sum(desc) - sum(stepsize.desc_old)
            τ_avg = stepsize.τ_old .* abs(p/q)
        else
            τ_cmp = [stepsize.τ_old[ik] * abs((- stepsize.desc_old[ik])/(- stepsize.desc_old[ik] + temp[ik])) for ik = 1:Nk];
            p = sum(stepsize.desc_old)
            q = sum(temp) - p
            τ_avg = stepsize.τ_old .* abs(p/q)
        end
        τ = find_τ(τ_avg, τ_cmp, stepsize.α);
    end
    #apply min, max
    #TODO do berfore?
    τ = [min(max(τ[ik], stepsize.τ_min), stepsize.τ_max) for ik = 1:Nk];

    #update params
    stepsize.desc_old = desc;
    stepsize.is_odd = !stepsize.is_odd;
    stepsize.τ_old = τ;
    
    return τ;
end

mutable struct SecantStep <:AbstractStepSize
        const τ_0
        const recalculate::Int64
        const retraction::AbstractRetraction
        const transport::AbstractTransport
        iter
        τ_old
    function  SecantStep(τ_0; recalculate = 0, retraction = RetractionPolar(), transport = DifferentiatedRetractionTransport())
        return new(τ_0, recalculate, retraction, transport, 0, nothing)
    end
end

function calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize::SecantStep; next = nothing)
    Nk = size(ψ)[1]
    if (isnothing(next))
        if (stepsize.τ_old === nothing || (stepsize.recalculate != 0 && (stepsize.iter+=1)%stepsize.recalculate == 0))
            #first step
            
            if (stepsize.τ_0 isa AbstractStepSize)
                τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, stepsize.τ_0)
            else    
                τ = [stepsize.τ_0 for ik = 1:Nk];
            end
        else
            τ = stepsize.τ_old;
        end
        next = do_step(basis, occupation, ψ, η, τ, stepsize.retraction, stepsize.transport);
    end

    ψ_next, ρ_next, energies_next, H_next, Hψ_next, Λ_next, res_next, Tη_next, τ = next

    desc_next = [dot(res_next[ik], Tη_next[ik]) for ik = 1:Nk]
    #caclulate generalized curvature
    desc_all = real(sum(τ .* desc))
    desc_next_all = real(sum(τ .* desc_next))

    #TODO check a condition?
    b = desc_all;
    two_a = desc_next_all - b;

    # replace τ by the location of the extremum of the quadratic interpolation
    τ *= (-b/two_a);
    stepsize.τ_old = τ
    
    return τ;
end

# η = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
# A = rand(4,4)
# Q, R = qr(A)
# ηc = [nk == 1 ? ηk * 1.0 : ηk * 0.0 for (ηk, nk) = zip(η, 1:length(η))]
# K_η = DFTK.apply_K(basis, η, ψ1, ρ1, occupation)
# K_ηc = DFTK.apply_K(basis, ηc, ψ1, ρ1, occupation)
# norm(sum(dot(η1[ik],K_η1[ik]) - K_η[1]))

# scfres_start = riemannian_conjugate_gradient(basis; 
#         tol = 1e-2,
#         gradient = H1Gradient(basis), 
#         backtracking = StandardBacktracking(ArmijoRule(0.2, 0.5), ApproxHessianStep(2.5), 10));

#naive generation of initial value
filled_occ = DFTK.filled_occupation(model);
n_spin = model.n_spin_components;
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
occupation = [filled_occ * ones(Float64, n_bands) for kpt in basis.kpoints]
# ψ1 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];


mutable struct AdaptiveBacktracking <: AbstractBacktracking 
    const rule::AbstractBacktrackingRule
    const stepsize::AbstractStepSize
    const maxiter::Int64
    const recalculate::Int64
    const only_backtrack::Bool
    τ_old
    iter::Int64
    function AdaptiveBacktracking(rule::AbstractBacktrackingRule, stepsize::AbstractStepSize, maxiter::Int64; recalculate = 0, only_backtrack = true)
        return new(rule, stepsize, maxiter, recalculate, only_backtrack, nothing, -1)
    end
end

function perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
    retraction::AbstractRetraction, transport::AbstractTransport, backtracking::AdaptiveBacktracking)

    next = nothing

    if (backtracking.recalculate != 0 && (backtracking.iter += 1)%backtracking.recalculate != 0 && !isnothing(backtracking.τ_old))
        #try to reuse old step size
        τ = backtracking.τ_old

        next = do_step(basis, occupation, ψ, η, τ, retraction, transport)
        
        if check_rule(energies.total, desc, next, backtracking.rule)
            update_rule!(next, backtracking.rule)
            return next
        end 
    end



    if (backtracking.recalculate != 0 && (backtracking.iter)%backtracking.recalculate == 0 || !backtracking.only_backtrack || isnothing(backtracking.τ_old))
        τ = calculate_τ(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, backtracking.stepsize; next);
        next = nothing;
    else
        τ = copy(backtracking.τ_old);
    end

    next = !isnothing(next) ? next : do_step(basis, occupation, ψ, η, τ, retraction, transport)

    #reset the count if old stepsize couldn't be used
    if (!backtracking.only_backtrack)
        backtracking.iter = 0;
    end



    for k = 1:backtracking.maxiter
        if check_rule(energies.total, desc, next, backtracking.rule)
            update_rule!(next, backtracking.rule)
            break
        end 

        τ = backtrack(τ, backtracking.rule)
        next = do_step(basis, occupation, ψ, η, τ, retraction, transport)
    end

    backtracking.τ_old = τ
    update_rule!(next, backtracking.rule)
    return next
end

mutable struct WolfeSecantRule <: AbstractBacktrackingRule
    const c_1::Float64
    const c_2::Float64
    const δ::Float64
    ω
    function WolfeSecantRule(c_1, c_2, δ)
        return new(c_1, c_2, δ, nothing)
    end
end
function check_rule(E_current, desc_current, next, rule::WolfeSecantRule)
    #small correction if change in energy is small TODO: better workaround.
    E_next = next.energies_next.total
    desc_next = [dot(res_next_k, Tη_next_k) for (res_next_k, Tη_next_k) = zip(next.res_next, next.Tη_next)]
    desc_next_all = real(sum(next.τ .* desc_next))
    desc_curr_all = real(sum(next.τ .* desc_current))

    if (E_next > E_current + (rule.c_1 * desc_curr_all + 32 * eps(Float64) * abs(E_current)))
        #Armijo condition not satisfied 
        rule.ω = rule.δ
        return false
    elseif (abs(desc_next_all) > rule.c_2 * abs(desc_curr_all))
        #Curvature condition not satisfied
        rule.ω = - desc_curr_all / (desc_next_all - desc_curr_all)
        return false
    else
        rule.ω = nothing
        return true
    end
end


function backtrack(τ, rule::WolfeSecantRule)
    if (isnothing(rule.ω))
        return
    end
    return τ * rule.ω
end

function update_rule!(next, ::WolfeSecantRule) end

function calculate_transport(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
    ret_pol::RetractionPolar ; 
    is_prev_dir = true)
Nk = size(ψ)[1]

#println(ξ[1][1])
DSf1 = τ.* [(ξ[ik]'ξ[ik]) for ik = 1:Nk]
T1 = [(ξ[ik] - ψ[ik] * DSf1[ik]) *  ret_pol.Sfinv[ik] for ik = 1:Nk]

Mtx_rhs = [ψ[ik]'ξ[ik]  for ik = 1:Nk]
Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
DSf = [lyap(ret_pol.Sf[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
T2 = [(ξ[ik] - ψ[ik] * DSf[ik]) * ret_pol.Sfinv[ik] for ik = 1:Nk]


if (is_prev_dir)
G = [ψ[ik]'ξ[ik] for ik = 1:Nk]
G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
L2ξ = [ξ[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
println(norm(T1 - T2)/norm(T2))
println(norm(T2 - L2ξ)/norm(T2))
return T1
end
return T2
end

# function calculate_transport(ψ, ξ, τ, ψ_old, ::DifferentiatedRetractionTransport,  
#     ret_pol::RetractionPolar ; 
#     is_prev_dir = true)
# Nk = size(ψ)[1]
# if (is_prev_dir)
# ξPinv = [ξ[ik] * ret_pol.Sfinv[ik] for ik = 1:Nk]
# τψ = τ .* ψ
# return [ξPinv[ik] - τψ[ik] * (ξPinv[ik]'ξPinv[ik]) for ik = 1:Nk]
# else
# Yrf = [ξ[ik] * ret_pol.Sf[ik] for ik = 1:Nk]
# Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
# Mtx_rhs = [Mtx_rhs[ik]' - Mtx_rhs[ik] for ik = 1:Nk]
# X = [lyap(ret_pol.Sfinv[ik], Mtx_rhs[ik]) for ik = 1:Nk]
# return [ ψ[ik] * X[ik] + Yrf[ik] + ψ[ik] * (ψ[ik]'Yrf[ik]) for ik = 1:Nk]
# end

# end


struct NoSolver <: LocalOptimalInnerSolver end
function copy_LOIS(::NoSolver)
    return NoSolver()
end

DFTK.@timing function solve_LOIS(bk, ξ0, Σk, lois::NoSolver)
    return ξ0
end

DFTK.@timing function update_LOIS(Y, HY, Pk, ::NoSolver)
end

DFTK.@timing function init_LOIS(ψk, Hψk, Σk, Pk, ::NoSolver)
end


mutable struct PreconditionedLeastSquaresLOIS <: LocalOptimalInnerSolver 
    A
    B
    P
    Φ
    HΦ
    PΦ
    PHΦ
    Pk
    function PreconditionedLeastSquaresLOIS()
        return new(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
    function PreconditionedLeastSquaresLOIS(A, B, P, Φ, HΦ, PΦ, PHΦ, Pk)
        return new(A, B, P, Φ, HΦ, PΦ, PHΦ, Pk)
    end
end

function copy_LOIS(lois::PreconditionedLeastSquaresLOIS)
    return PreconditionedLeastSquaresLOIS(lois.A, lois.B, lois.P, lois.Φ, lois.HΦ, lois.PΦ, lois.PHΦ, lois.Pk)
end

DFTK.@timing function solve_LOIS(bk, ξ0, Σk, lois::PreconditionedLeastSquaresLOIS)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    x0 = lois.Φ'ξ0
    Pbk = lois.Pk \ bk

    rhs = lois.PHΦ'Pbk + (lois.PΦ'Pbk) * Σk

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    norm_rhs = norm(rhs)

    rhs0 = rhs/ norm_rhs
    rhs0 = pack(rhs0)
    x1 = x0 / norm_rhs
    x1 = pack(x1)

    Σk2 = Σk * Σk'
    function apply_A(x)
        Y = unpack(x)
        return pack(lois.B * Y + lois.A * Y * Σk + lois.P * Y * Σk2)
    end


    J = LinearMap{T}(apply_A, size(rhs0, 1))

    x, stats = Krylov.minres(J, rhs0, x1; itmax = 100, atol = 1e-16, rtol = 1e-16)
    # if (!stats.solved)
    #     print(stats)
    # end
    x *= norm_rhs

    x2 = solve_preconditioned_least_squares(Krylov.minres, lois.HΦ, lois.Φ, bk, lois.Pk, Σk, 100, 1e-16)

    #println(norm(x2- unpack(x))/(norm(x2)))

    return lois.Φ * unpack(x)
end

DFTK.@timing function update_LOIS(Y, HY, Pk, lois::PreconditionedLeastSquaresLOIS)
    add = Y - lois.Φ * (lois.Φ'Y);
    #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
    idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]
    add = add[:, idxs]

    add, R = ortho_qr(add);

    Hadd = HY[:, idxs] - lois.HΦ * (lois.Φ'Y[:, idxs])
    Hadd = Hadd /R
    PHadd = Pk \ Hadd
    Padd = Pk \ add

    A12 = lois.PΦ'PHadd + lois.PHΦ'Padd
    A21 = A12'
    A22 = Padd'PHadd
    A22 = A22 + A22'
    lois.A = vcat(hcat(lois.A  , A12),
            hcat(A21, A22))

    B12 = lois.PHΦ'PHadd
    B21 = B12'
    B22 = PHadd'PHadd
    B22 = 0.5 * (B22 + B22')
    lois.B = vcat(hcat(lois.B  , B12),
            hcat(B21, B22))

    P12 = lois.PΦ'Padd
    P21 = P12'
    P22 = Padd'Padd
    P22 = 0.5 * (P22 + P22')
    lois.P = vcat(hcat(lois.P  , P12),
            hcat(P21, P22))

    lois.Φ = hcat(lois.Φ, add)
    lois.HΦ = hcat(lois.HΦ, Hadd)
    lois.PΦ = hcat(lois.PΦ, Padd)
    lois.PHΦ = hcat(lois.PHΦ, PHadd)

    lois.Pk = Pk
end

DFTK.@timing function init_LOIS(ψk, Hψk, Σk, Pk, lois::PreconditionedLeastSquaresLOIS)
    lois.Φ = ψk
    lois.HΦ = Hψk
    lois.PΦ = Pk \ ψk
    lois.PHΦ = Pk \ Hψk 
    A = lois.PΦ'lois.PHΦ
    lois.A = A' + A
    lois.B = lois.PHΦ'lois.PHΦ
    lois.P = lois.PΦ'lois.PΦ
    lois.Pk = Pk
end


mutable struct LeastSquaresLOIS <: LocalOptimalInnerSolver 
    A
    B
    Prec
    Φ
    HΦ
    PΦ
    function LeastSquaresLOIS()
        return new(nothing, nothing, nothing, nothing, nothing, nothing)
    end
    function LeastSquaresLOIS(A, B, Prec, Φ, HΦ, PΦ)
        return new(A, B, Prec, Φ, HΦ, PΦ)
    end
end

function copy_LOIS(lois::LeastSquaresLOIS)
    return LeastSquaresLOIS(lois.A, lois.B, lois.Prec, lois.Φ, lois.HΦ, lois.PΦ)
end

DFTK.@timing function solve_LOIS(bk, ξ0, Σk, lois::LeastSquaresLOIS)
    #As operator is only given implicitly, we need to solve the normal equation.
    T = Base.Float64

    x0 = lois.Φ'ξ0
    rhs = lois.HΦ'bk + (lois.Φ'bk) * Σk

    n_rows, n_cols = size(rhs)
    unpack(x) = unpack_general(x, n_rows, n_cols)

    norm_rhs = norm(rhs)

    rhs0 = rhs/ norm_rhs
    rhs0 = pack(rhs0)
    x1 = x0 / norm_rhs
    x1 = pack(x1)

    Σk2 = Σk * Σk'
    function apply_A(x)
        Y = unpack(x)
        return pack(lois.B * Y + 2 * lois.A * Y * Σk + Y * Σk2)
    end

    J = LinearMap{T}(apply_A, size(rhs0, 1))

    function apply_Prec(x)
        Y = unpack(x)
        return pack(lois.Prec * Y)
    end
    M = LinearMap{T}(apply_Prec, size(rhs0, 1))

    x, stats = Krylov.minres(J, rhs0, x1; M, itmax = 100, atol = 1e-16, rtol = 1e-16)
    # if (!stats.solved)
    #     print(stats)
    # end
    x *= norm_rhs

    return lois.Φ * unpack(x)
end

DFTK.@timing function update_LOIS(Y, HY, Pk, lois::LeastSquaresLOIS)
    add = Y - lois.Φ * (lois.Φ'Y);
    #only choose vectors sufficiently far away from Φ (if these values are small, condition of R is large)
    idxs = [i for i = 1:size(Y)[2] if norm(add[:,i])/norm(Y[:,i]) > 1e-4]
    add = add[:, idxs]

    add, R = ortho_qr(add);

    Hadd = HY[:, idxs] - lois.HΦ * (lois.Φ'Y[:, idxs])
    Hadd = Hadd /R
    Padd = Pk \ add

    A12 = lois.Φ'Hadd
    A21 = A12'
    A22 = add'Hadd
    A22 = 0.5 * (A22 + A22')
    lois.A = vcat(hcat(lois.A  , A12),
            hcat(A21, A22))

    B12 = lois.HΦ'Hadd
    B21 = B12'
    B22 = Hadd'Hadd
    B22 = 0.5 * (B22 + B22')
    lois.B = vcat(hcat(lois.B  , B12),
            hcat(B21, B22))

    P12 = lois.PΦ'Padd
    P21 = P12'
    P22 = Padd'Padd
    P22 = 0.5 * (P22 + P22')
    lois.Prec = vcat(hcat(lois.Prec, P12),
                hcat(P21 , P22))

    
    lois.Φ = hcat(lois.Φ, add)
    lois.HΦ = hcat(lois.HΦ, Hadd)
    lois.PΦ = hcat(lois.PΦ, Padd)

    # tildeA = lois.Φ'lois.HΦ
    # tildeA = 0.5 * (tildeA' + tildeA)

    # lois.A = tildeA

end

DFTK.@timing function init_LOIS(ψk, Hψk, Σk, Pk, lois::LeastSquaresLOIS)
    lois.Φ = ψk
    lois.HΦ = Hψk 
    lois.PΦ = Pk \ ψk
    A = ψk'Hψk
    lois.A = 0.5 * (A' + A) 
    lois.B = Hψk'Hψk
    Prec = lois.PΦ'lois.PΦ
    lois.Prec = 0.5 * (Prec' + Prec)
end

function calculate_gradient(ψ, Hψ, H, Λ, res, ea_grad::EAGradient)
    Nk = size(ψ)[1];
    Σ = calculate_shift(ψ, Hψ, H, Λ, res, ea_grad.shift);

    for ik = 1:Nk
       DFTK.precondprep!(ea_grad.Pks[ik], ψ[ik]); 
    end

    inner_tols = [min(max(ea_grad.rtol * norm(res[ik]), ea_grad.atol), 1.0) for ik = 1:Nk]


    X = solve_H(ea_grad.krylov_solver, H, res, Σ, ψ, Hψ, ea_grad.itmax, inner_tols, ea_grad.Pks, ea_grad.h_solver)

    return[ψ[ik] - (ψ[ik] -X[ik]) /(I + ψ[ik]'X[ik]) for ik = 1:Nk]

    if ( (abs(ea_grad.shift.μ) < 1e-6))
        #
        return X;
    else
        #needed for the ea projection
        ea_grad.Hinv_ψ = [(ψ[ik] - X[ik]) / (Λ[ik] + Σ[ik] * I)  for ik = 1:Nk]
        #Ga = [ψ[ik]'ea_grad.Hinv_ψ[ik] for ik = 1:Nk]
        #g = [ ψ[ik] - ea_grad.Hinv_ψ[ik] / Ga[ik] for ik = 1:Nk]

        #Alternative without inverting Λ + Σ:
        ψX = [ψ[ik]'X[ik] for ik = 1:Nk]
        g = [-ψ[ik] * ψX[ik] + X[ik] * (I + ψX[ik]) for ik = 1:Nk]
        # L1 = ψ[1]'X[1]


        println(norm(g2[1] - g[1])/norm(g[1]))

        return g
    end
end
