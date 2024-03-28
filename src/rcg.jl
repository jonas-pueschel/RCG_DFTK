using DFTK
using LinearMaps
using IterativeSolvers

include("./rcg_params.jl")
include("./rcg_callbacks.jl")
# RCG algorithm to solve SCF equations


function rcg(basis::PlaneWaveBasis{T}, ψ0;
                tol=1e-6, maxiter=400,
                callback=RcgDefaultCallback(),
                gradient = get_default_EA_gradient(basis),
                retraction = default_retraction(basis),
                cg_param = ParamFR_PRP(),
                transport_η = DifferentiatedRetractionTransport(),
                transport_grad = DifferentiatedRetractionTransport(),
                stepsize = ExactHessianStep(2.0),
                backtracking = StandardBacktracking(AvgNonmonotoneRule(0.95, 0.0001, 0.5), 10, false)
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
    σ = options.gradient_options.shift

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

    # calculate gradient
    grad = calculate_gradientcalculate_gradient(ψ, Hψ, H, Λ, res, gradient)

    # give β the information from first iteration
    init_β(gamma, res, grad, cg_param)
    
    gamma = [real(tr(res[ik]'grad[ik])) for ik = 1:Nk]
    desc = - gamma
    η = - grad
    T_η_old = nothing

    # perform iterations
    converged = false
    while n_iter < maxiter
        n_iter += 1

        #perform step
        next = perform_backtracking(ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, occupation, basis, energies, 
                retraction, stepsize, backtracking)

        # callback and test convergence
        info = (; ham=H, basis, converged, stage=:iterate, ρin=ρ, ρout=next.ρ_next, n_iter,
                energies, algorithm="RCG")


        #update orbitals, density, H and energies
        ψ_old = ψ
        ψ = next.ψ_next
        ρ = next.ρ_next
        H = next.H_next
        energies = next.energies_next



        #compute residual
        Hψ = [H.blocks[ik] * ψk + σ * ψk for (ik, ψk) in enumerate(ψ)]
        Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
        Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
        res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]

        # check convergence
        norm2res = sum([real(tr(res[ik]'res[ik])) for ik = 1:Nk])
        callback(info, sqrt(norm2res))
        if(norm2res < tol * tol)
            break
        end

        #calculate_gradient
        grad = calculate_gradientcalculate_gradient(ψ, Hψ, H, Λ, res, gradient)
        gamma = [real(tr(res[ik]'grad[ik])) for ik = 1:Nk]

        # calculate transport of η
        T_η_old = calculate_tranpsort(ψ, η, τ, ψ_old, transport_η, retraction, gradient; is_prev_dir = true)

        # give transport rule for grad (only used if necessary for β)
        transport(ξ) = calculate_tranpsort(ψ, ξ, τ, ψ_old, transport_grad, retraction, gradient; is_prev_dir = false)

        # calculate the cg param, note that desc is the descent from the *last* iteration
        β = calculate_β(gamma, desc, res, grad, T_η_old, transport , cg_param)

        # calculate new direction
        η = [-grad[ik] + β[ik] * η[ik] for ik = 1:Nk] 

        #check if η is a descent direction. If not, restart
        desc = [tr(η[ik]'res[ik]) for ik in 1:Nk]
        for ik = 1:Nk   
            if (real(desc[ik]) >= 0)
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

    #TODO fix converged
    info = (; ham=H, basis, energies, converged = true , ρ, eigenvalues, occupation, εF, n_iter, ψ,
            stage=:finalize, algorithm="RCG")
    callback(info, sqrt(norm2res))
    info
end
