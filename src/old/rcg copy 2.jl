using DFTK
using LinearMaps
using IterativeSolvers

include("./rcg_params.jl")
include("./rcg_callbacks.jl")
# RCG algorithm to solve SCF equations


DFTK.@timing function riemannian_conjugate_gradient(basis::PlaneWaveBasis{T}, ψ0;
                tol=1e-6, maxiter=100,
                callback = RcgDefaultCallback(),
                is_converged = RcgConvergenceResidual(tol),
                gradient = default_EA_gradient(basis),
                retraction = RetractionPolar(),
                cg_param = ParamFR_PRP(),
                transport_η = L2ProjectionTransport(),
                transport_grad = L2ProjectionTransport(),
                backtracking = GreedyBacktracking(ArmijoRule(0.1, 0.5), ExactHessianStep(2.5), 0, 10)
                ) where {T}

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

    occupation = [filled_occ * ones(T, n_bands) for ik = 1:Nk]

    # iterators
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations
    ψ = deepcopy(ψ0)
    ρ = compute_density(basis, ψ, occupation)
    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)

    # compute first residual 
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]

    #innitial callback
    info = (; ham=H, ψ,basis, converged = false, stage=:iterate, norm_res = norm(res), ρin=nothing, ρout=ρ, n_iter,
    energies, algorithm="RCG")
    #callback(info)

    # calculate gradient
    grad = calculate_gradient(ψ, Hψ, H, Λ, res, gradient)
  
    T_η_old = nothing
    gamma = [real(tr(res[ik]'grad[ik])) for ik = 1:Nk]
    desc = - gamma
    η = - grad


    # give β the information from first iteration
    init_β(gamma, res, grad, cg_param)

    # perform iterations
    while n_iter < maxiter
        n_iter += 1

        #perform step
        next = perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
                retraction, transport_grad, backtracking)

        #update orbitals, density, H and energies
        ψ_old = ψ
        ψ = next.ψ_next
        ρ_prev = ρ
        ρ = next.ρ_next
        H = next.H_next
        τ = next.τ
        energies = next.energies_next



        #println([norm(τ[ik] * η[ik]) for ik = 1:Nk])

        if (!isnothing(next.Hψ_next) && !isnothing(next.Λ_next) && !isnothing(next.res_next))
            #compute residual
            Hψ = next.Hψ_next
            Λ = next.Λ_next
            res = next.res_next
        else
            #compute residual
            Hψ = [H.blocks[ik] * ψk for (ik, ψk) in enumerate(ψ)]
            Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
            Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
            res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]
        end


        info = (; ham=H, ψ,basis, converged = is_converged(info), stage=:iterate, norm_res = norm(res), ρin=ρ_prev, ρout=ρ, n_iter,
        energies, algorithm="RCG")

        callback(info)
        # callback and test convergence
        if info.converged
            break
        end

        #calculate_gradient
        grad = calculate_gradient(ψ, Hψ, H, Λ, res, gradient)
        gamma = [real(dot(res[ik],grad[ik])) for ik = 1:Nk]

        # calculate transport of η
        if (!isnothing(next.Tη_next))
            T_η_old = next.Tη_next
        else
            T_η_old = calculate_transport(ψ, η, τ, ψ_old, transport_η, retraction; is_prev_dir = true)
        end
        # give transport rule for grad (only used if necessary for β)
        transport(ξ) = calculate_transport(ψ, ξ, τ, ψ_old, transport_grad, retraction; is_prev_dir = false)

        # calculate the cg param, note that desc is the descent from the *last* iteration
        β = calculate_β(gamma, desc, res, grad, T_η_old, transport , cg_param)

        # calculate new direction
        η = [-grad[ik] + β[ik] * η[ik] for ik = 1:Nk] 

        #check if η is a descent direction. If not, restart
        desc = [tr(η[ik]'res[ik]) for ik in 1:Nk]
        for ik = 1:Nk   
            if (real(desc[ik]) >= 0)
                #println("Warning: this direction is not a desc dir!")
                
                #restarting
                #Note: it may be possible that the ea gradient is not a descent direction
                #this cannot be caught here!
                η[ik] = - grad[ik]
                desc[ik] = - gamma[ik]
            end
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

    # λ_min = [eigmin(real(Λ[ik])) for ik = 1:Nk]
    # println(λ_min)

    info = (; ham=H, ψ, basis, energies, converged = is_converged(info) , norm_res = norm(res), ρ, eigenvalues, occupation, εF, n_iter,
            stage=:finalize, algorithm="RCG")
    callback(info)

    info
end