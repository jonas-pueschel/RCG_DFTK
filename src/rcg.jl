using DFTK
using LinearMaps
using IterativeSolvers

include("./rcg_options.jl")
include("./ortho.jl")
include("./lyap_solvers.jl")
include("./solve_H.jl")
include("./rcg_callbacks.jl")
# RCG algorithm to solve SCF equations


function rcg(basis::PlaneWaveBasis{T}, ψ0;
                tol=1e-6, maxiter=400,
                callback=RcgDefaultCallback(),
                options::RcgOptions = default_options(basis, ψ0)) where {T}

    function tangent_space_transport(ξ, options::RcgOptions; previous_dir = true)
        if (previous_dir)
            transport = options.transport_η 
        else
            transport = options.transport_res
        end

        if (transport == "L2-proj")
            G = [ψ[ik]'ξ[ik] for ik = 1:Nk]
            G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
            return [ξ[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
        elseif (transport == "H1-proj")
            #TODO posspible duplicate calculation
            P_ψ = [ prec_H1[ik] \ ψk for (ik, ψk) in enumerate(ψ)]
            Mtx_lhs = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
            Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
            Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
            X = [lyap(Mtx_lhs[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
            return [ξ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
        elseif (transport == "ea-proj")
            if (options.gradient_options.gradient != "ea")
                throw("ea-proj transport is only possible with ea gradient")
            end
            Mtx_lhs = [ψ[ik]'Hinv_ψ[ik] for ik = 1:Nk]
            Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
            Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
            X = [lyap(Mtx_lhs[ik], - Mtx_rhs[ik]) for ik = 1:Nk]
            return [ξ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
        elseif (transport == "tf")
            return ξ
        elseif (transport == "diff-ret" && options.retraction == "qr")
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
        elseif (previous_dir)
            # special transport if the previous direction is transported
            if (transport == "diff-ret")
                #polar retraction
                ξP = [ξ[ik] * R[ik] for ik = 1:Nk]
                return [ξP[ik] - τ[ik] * ψ[ik] * (ξP[ik]'ξP[ik]) for ik = 1:Nk]
            elseif (transport == "inv-ret")
                if (options.retraction == "qr")
                    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
                    R_ir = [tsylv2solve(Mtx_lhs[ik]) for ik = 1:Nk]
                    return [- ψ_old[ik] * R_ir[ik] + ψ[ik] for ik = 1:Nk]
                elseif (options.retraction == "polar")
                    Mtx_lhs = [ψ[ik]'ψ_old[ik] for ik = 1:Nk]
                    S_ir = [lyap(Mtx_lhs[ik], - 2.0 * Matrix{ComplexF64}(I, size(ψ)[1], size(ψ)[1])) for ik = 1:Nk]
                    return [- ψ_old[ik] * S_ir[ik] + ψ[ik] for ik = 1:Nk]
                end
            else
                throw(DomainError(transport, "invalid transport for η"))
            end
        else
            # general transport
            if (transport == "diff-ret")
                #polar retraction
                #TODO this can be improved: the inverese of R can be calculated explicitly during polar decompoistion
                Yrf = [ξ[ik] / R[ik] for ik = 1:Nk]
                Mtx_rhs = [ψ[ik]'ξ[ik] for ik = 1:Nk]
                Mtx_rhs = [Mtx_rhs[ik]' - Mtx_rhs[ik] for ik = 1:Nk]
                X = [lyap(R[ik], Mtx_rhs[ik]) for ik = 1:Nk]
                return [ ψ[ik] * X[ik] + Yrf[ik] + ψ[ik] * (ψ[ik]'Yrf[ik]) for ik = 1:Nk]
            else
                throw(DomainError(transport, "invalid general transport"))
            end
        end

        #DEBUG
        throw("transport not impmented")
    end

    function calculate_gradient(options::GradientOptions)
        if (options.gradient== "prec")
            #use the inner prec instead of H1 
            P_res = [ options.prec[ik] \ resk for (ik, resk) in enumerate(res)]
            G = [ψ[ik]'P_res[ik] for ik = 1:Nk]
            G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
            return [P_res[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
        elseif (options.gradient== "H1-prec")
            P_res = [ prec_H1[ik] \ resk for (ik, resk) in enumerate(res)]
            G = [ψ[ik]'P_res[ik] for ik = 1:Nk]
            G = 0.5 * [G[ik]' + G[ik] for ik = 1:Nk]
            return [P_res[ik] - ψ[ik] * G[ik] for ik = 1:Nk]
        elseif (options.gradient== "ea-prec")
            Hinv_res = solve_H(res, H, res, options)
            Ge = [ψ[ik]'Hinv_res[ik] for ik = 1:Nk]
            Ge = [0.5 * (Ge[ik]' + Ge[ik]) for ik = 1:Nk]
            return [Hinv_res[ik] - ψ[ik] * Ge[ik] for ik = 1:Nk]
        elseif (options.gradient== "H1")
            #TODO posspible duplicate calculation
            P_ψ = [ prec_H1[ik] \ ψk for (ik, ψk) in enumerate(ψ)]
            P_Hψ = [ prec_H1[ik] \ Hψ[ik] for ik = 1:Nk]
            Mtx_lhs = [ψ[ik]'P_ψ[ik] for ik = 1:Nk]
            Mtx_rhs = [ψ[ik]'P_Hψ[ik] for ik = 1:Nk]
            Mtx_rhs = [Mtx_rhs[ik]' + Mtx_rhs[ik] for ik = 1:Nk]
            X = [lyap(Mtx_lhs[ik], -Mtx_rhs[ik]) for ik = 1:Nk]
            return [P_Hψ[ik] - P_ψ[ik] * X[ik] for ik = 1:Nk]
        elseif (options.gradient== "ea")
            ψ0 = [ψ[ik] / Λ[ik] for ik = 1:Nk]
            Hinv_ψ = solve_H(ψ0, H, ψ, options)
            #check if inverse is correct
            #println(norm(ψ .- (H * Hinv_ψ), 2))
            Ga = [ψ[ik]'Hinv_ψ[ik] for ik = 1:Nk]
            return [ ψ[ik] - Hinv_ψ[ik] / Ga[ik] for ik = 1:Nk]
        elseif (options.gradient== "ea2")
            ψ0 = [ψ[ik] / Λ[ik] for ik = 1:Nk]
            Hinv_ψ = solve_H_naive(ψ0, H, ψ, options)
            #check if inverse is correct
            #println(norm(ψ .- (H * Hinv_ψ), 2))
            Ga = [ψ[ik]'Hinv_ψ[ik] for ik = 1:Nk]
            return [ ψ[ik] - Hinv_ψ[ik] / Ga[ik] for ik = 1:Nk]
        elseif (options.gradient == "L2")
            #classic gradient
            return res;
        else 
            throw(DomainError(options.gradient, "invalid gradient"))
        end
    end

    function calculate_τ_0(options::StepSizeOptions)
        if (options.step_size == "eH")
            Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
            η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
            K_η = DFTK.apply_K(basis, η, ψ, ρ, occupation)
            η_K_η = [tr(η[ik]'K_η[ik]) for ik in 1:Nk]
            #return [real(η_Ω_η[ik] + η_K_η[ik]) > 0 ? real(- desc[ik]/(η_Ω_η[ik] + η_K_η[ik])) : options.τ_0_max for ik in 1:Nk]
            return [min(real(- desc[ik]/abs(η_Ω_η[ik] + η_K_η[ik])) , options.τ_0_max) for ik in 1:Nk]
        elseif (options.step_size == "aH")
            Ω_η = [H.blocks[ik] * ηk - ηk * Λ[ik] for (ik, ηk) in enumerate(η)] 
            η_Ω_η = [tr(η[ik]'Ω_η[ik]) for ik in 1:Nk]
            return [min(real(abs(desc[ik]/(η_Ω_η[ik]))), options.τ_0_max) for ik in 1:Nk]
        elseif (options.step_size == "cst")
            return [options.τ_const for ik in 1:Nk]
        else
            throw(DomainError(options.step_size, "invalid step_size"))
        end 

    end



    function calculate_β(options::RcgOptions)
        if (options.β_cg == "FR")
            return [real(gamma[ik]/gamma_old[ik]) for ik = 1:Nk]
        elseif (options.β_cg == "HS")
            return [real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/(tr(T_η_old[ik]'res[ik]) - tr(η_old[ik]'res_old[ik]))) for ik = 1:Nk]
        elseif (options.β_cg == "PRP")
            return [real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/gamma_old[ik]) for ik = 1:Nk]
        elseif (options.β_cg == "DY")
            return [real(gamma[ik]/(tr(T_η_old[ik]'res[ik]) - tr(η_old[ik]'res_old[ik]))) for ik = 1:Nk]
        elseif (options.β_cg == "D")
            return [0 for ik = 1:Nk] #TODO
        elseif (options.β_cg == "FR-PRP")
            return [max(min(gamma[ik], (gamma[ik] - real(tr(T_grad_old[ik]'res[ik]))))/gamma_old[ik], 0) for ik = 1:Nk]
        elseif (options.β_cg == "HS-DY")
            temp = [real(tr(T_η_old[ik]'res[ik]) - tr(η_old[ik]'res_old[ik])) for ik = 1:Nk ]
            return [max(min(real((gamma[ik] - tr(T_grad_old[ik]'res[ik]))/temp[ik] - tr(η_old[ik]'res_old[ik])), real(gamma[ik]/temp[ik])), 0) for ik = 1: Nk]
        elseif (options.β_cg == "HZ")
            temp1 = [real(tr(T_η_old[ik]'res[ik]) - tr(η_old[ik]'res_old[ik])) for ik = 1:Nk]
            # res'res is calculted twice (for res and res_old), this could be optimized
            temp2 = [real(gamma[ik] - 2 * tr(T_grad_old[ik]'res[ik]) + gamma_old[ik]/temp1[ik]) for ik = 1:Nk]
            return [(gamma[ik] - real(tr(T_grad_old[ik]'res[ik]) - tr(T_η_old[ik]'res[ik]) * options.μ * temp2[ik]))/temp1[ik] for ik = 1:Nk]        
        else
            throw(DomainError(options.β_cg, "invalid β_cg"))
        end
    end

    prec_H1 = [DFTK.PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]

    # setting parameters
    model = basis.model
    #@assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed Fermi level

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    @assert n_bands == size(ψ0[1], 2)

    # number of kpoints and occupation
    Nk = length(basis.kpoints)
    σ = options.gradient_options.shift

    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # iterators
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations
    ψ = deepcopy(ψ0)
    ρ = compute_density(basis, ψ, occupation)
    ρ_next = nothing
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)

    # set up backtracking
    q = 1.0
    c = energies.total


    # compute first residual 
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]
    
    #apply preconditioner

    grad = calculate_gradient(options.gradient_options);

    
    gamma = [real(tr(res[ik]'grad[ik])) for ik = 1:Nk]
    #gamma = [real(tr(grad[ik]' * (H.blocks[ik] * grad[ik] + σ * grad[ik]))) for ik = 1:Nk]
    gamma_old = nothing;
    desc = - gamma
    η = - grad

    #define some stuff?
    P_ψ = nothing
    Hinv_ψ = nothing
    ψ_old = nothing
    η_old = nothing 
    T_η_old = nothing
    res_old = nothing
    grad_old = nothing
    T_grad_old = nothing
    ψ_next = nothing
    R = nothing
    τ = nothing
    norm2res = nothing

    # perform iterations
    converged = false
    while n_iter < maxiter
        n_iter += 1
        #save old psi
        ψ_old = ψ

        #compute  Stepsize
        τ = calculate_τ_0(options.step_size_options)

        for k in 0:options.step_size_options.bt_iter
            if (k != 0)
                τ = τ * options.step_size_options.δ
            end

            if (options.retraction == "qr")
                ψR =  [ortho_qr(ψ[ik] + τ[ik] * η[ik]) for ik in 1:Nk]
            elseif (options.retraction == "polar")
                ψR  = [ortho_polar(ψ[ik] + τ[ik] * η[ik]) for ik in 1:Nk]
            else
                throw(DomainError(options.retraction, "invalid retraction"));
            end

            
            ψ_next = [ψR[ik][1] for ik in 1:Nk]
            R = [ψR[ik][2] for ik in 1:Nk]
            ρ_next = DFTK.compute_density(basis, ψ_next, occupation)
            energies, H = DFTK.energy_hamiltonian(basis, ψ_next, occupation; ρ=ρ_next)

            #not sure how to implement the descending with multiple orbitals Nk > 1 ... 
            #maybe TODO: separate backtracking for the different directions
            if (energies.total <= c + options.step_size_options.β * τ[1] * real(desc[1]))
                break;
            end
        end

        #print(real(desc[1]))
        #print(τ)

        #update backtrack parameters
        q = options.step_size_options.α * q + 1;
        c = (1-1/q)*c + 1/q * energies.total;



        # callback and test convergence
        info = (; ham=H, basis, converged, stage=:iterate, ρin=ρ, ρout=ρ_next, n_iter,
                energies, algorithm="RCG")


        #update orbitals and density
        ψ = ψ_next
        ρ = ρ_next



        #compute residual
        res_old = deepcopy(res)
        grad_old = deepcopy(grad)
        η_old = η;

        Hψ = [H.blocks[ik] * ψk + σ * ψk for (ik, ψk) in enumerate(ψ)]
        Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
        Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
        res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]

        gamma_old = deepcopy(gamma)
        grad = calculate_gradient(options.gradient_options);

        gamma = [real(tr(res[ik]'grad[ik])) for ik = 1:Nk]
        #gamma = [real(tr(grad[ik]' * (H.blocks[ik] * grad[ik] + σ * grad[ik]))) for ik = 1:Nk]

        norm2res = 0
        for ik = 1:Nk
            norm2res += real(tr(res[ik]'res[ik]))
        end
        #norm2grad = 0
        #for ik = 1:Nk
        #    norm2grad += real(tr(grad[ik]'grad[ik]))
        #end
        callback(info, sqrt(norm2res))

        if(norm2res < tol * tol)
            break
        end

        if (options.do_cg)

            #tangent space transport (projection)
            T_grad_old = tangent_space_transport(grad_old, options, previous_dir = false)
            
            #check if transported dir is in tangent space
            #GGG = [ψ[ik]'T_grad_old[ik] for ik = 1:Nk]
            #print(sum([norm(GGG[ik]' + GGG[ik]) for ik = 1:Nk]))

            #in theory this may also be saved on η
            T_η_old = tangent_space_transport(η, options, previous_dir = true)

            #cg search dir with FR-PRP param
            β = calculate_β(options);

            η = [-grad[ik] + β[ik] * η[ik] for ik = 1:Nk] 
            
            #check if η is a descent direction. If not, restart
            desc = [tr(η[ik]'res[ik]) for ik in 1:Nk]
            for ik = 1:Nk   
                if (real(desc[ik]) >= 0)
                    η[ik] = - grad[ik]
                    desc[ik] = - gamma[ik]
                end
            end
        else
            η .= -grad
            desc .= - gamma
        end
    end

    # Rayleigh-Ritz
    eigenvalues = []
    for ik = 1:Nk
        Hψk = H.blocks[ik] * ψ[ik]
        F = eigen(Hermitian(ψ[ik]'Hψk))
        push!(eigenvalues, F.values)
        ψ[ik] .= ψ[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
                  # Aufbau property might not even be true

    # return results and call callback one last time with final state for clean
    # up

    #TODO fix converged
    info = (; ham=H, basis, energies, converged = true , ρ, eigenvalues, occupation, εF, n_iter, ψ,
            stage=:finalize, algorithm="RCG")
    callback(info, sqrt(norm2res))
    info
end
