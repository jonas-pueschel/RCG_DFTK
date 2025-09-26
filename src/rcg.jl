# RCG algorithm to solve SCF equations
DFTK.@timing function riemannian_conjugate_gradient(
        basis::PlaneWaveBasis{T};
        ρ = guess_density(basis),
        ψ = nothing,
        tol = 1.0e-6, maxiter = 100,
        callback = RcgDefaultCallback(),
        is_converged = RcgConvergenceResidual(tol),
        gradient = EAGradient(basis, CorrectedRelativeΛShift(μ = 0.01)),
        retraction = RetractionPolar(),
        cg_param = ParamFR_PRP(),
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        check_convergence_early = false, #check convergence before expensive gradient calculation
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.05, 0.1, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
    ) where {T}
    start_ns = time_ns()
    # setting parameters
    model = basis.model
    #@assert iszero(model.temperature)  # temperature is not yet supported
    @assert isnothing(model.εF)        # neither are computations with fixed Fermi level

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)

    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
        @assert n_bands == size(ψ[1], 2)
    else
        ψ = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
    end

    if isnothing(ρ)
        ρ = guess_density(basis)
    end


    # number of kpoints and occupation
    Nk = length(basis.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik in 1:Nk]

    # iterators
    n_iter = 0

    # orbitals, densities and energies to be updated along the iterations

    energies, H = energy_hamiltonian(basis, ψ, occupation; ρ)

    # compute first residual
    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik in 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik in 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik in 1:Nk]

    # calculate gradient
    grad = calculate_gradient(ψ, Hψ, H, Λ, res, gradient)

    T_η_old = nothing
    gamma = inner_product_DFTK(basis, res, grad)
    desc = - gamma
    η = - grad

    #initial callback
    info = (;
        ham = H, ψ, grad, res, η, basis, converged = false, stage = :iterate, norm_res = sqrt(abs(inner_product_DFTK(basis, res, res))), norm_grad = sqrt(abs(inner_product_DFTK(basis, res, grad))), ρin = nothing, ρout = ρ, n_iter,
        energies, algorithm = "RCG",
    )
    #callback(info)

    τ = nothing

    # give β the information from first iteration
    init_β(gamma, res, grad, cg_param)

    # perform iterations
    while n_iter < maxiter
        n_iter += 1

        #perform step
        get_next(τ_trial) = get_next_rcg(basis, occupation, ψ, η, τ_trial, retraction, transport_η)
        next = do_step(basis, ψ, η, grad, res, T_η_old, desc, Λ, H, ρ, energies, get_next, iteration_strat)

        #update orbitals, density, H and energies
        ψ_old = ψ
        ψ = next.ψ_next
        ρ_prev = ρ
        ρ = next.ρ_next
        H = next.H_next
        τ = next.τ
        energies = next.energies_next
        Hψ = next.Hψ_next
        Λ = next.Λ_next
        res = next.res_next

        # test convergence before expensive gradient caluclation, note that info contains old grad, η!
        if check_convergence_early
            info = (;
                ham = H, ψ, grad, res, η, τ, basis, converged = false, stage = :iterate, norm_res = sqrt(abs(inner_product_DFTK(basis, res, res))), norm_grad = nothing, ρin = ρ_prev, ρout = ρ, n_iter,
                energies, start_ns, algorithm = "RCG",
            )
            callback(info)
            if is_converged(info)
                break
            end
        end

        #calculate_gradient
        grad = calculate_gradient(ψ, Hψ, H, Λ, res, gradient)
        gamma = inner_product_DFTK(basis, res, grad)

        # calculate transport of η
        if (!isnothing(next.Tη_next))
            T_η_old = next.Tη_next
        else
            T_η_old = calculate_transport(ψ, η, η, τ, ψ_old, transport_η, retraction; is_prev_dir = true)
        end
        # give transport rule for grad (only used if necessary for β)
        transport(ξ) = calculate_transport(ψ, ξ, η, τ, ψ_old, transport_grad, retraction; is_prev_dir = false)

        # calculate the cg param, note that desc is the descent from the *last* iteration
        β = calculate_β(basis, gamma, desc, res, grad, T_η_old, transport, cg_param)

        # calculate new direction
        η = -grad + T_η_old .* β


        #check convergence
        if !check_convergence_early
            # update info and callback
            info = (;
                ham = H, ψ, grad, res, η, τ, basis, converged = false, stage = :iterate, norm_res = sqrt(abs(inner_product_DFTK(basis, res, res))), norm_grad = sqrt(abs(inner_product_DFTK(basis, res, grad))), ρin = ρ_prev, ρout = ρ, n_iter,
                energies, start_ns, algorithm = "RCG",
            )
            callback(info)
            if is_converged(info)
                break
            end
        end

        #check if η is a descent direction. If not, restart
        desc = inner_product_DFTK(basis, η, res)
        if (desc >= 0)
            @warn "the search direction is not a descent direction, try to use a better initial guess"
        end
    end

    # Rayleigh-Ritz
    eigenvalues = []
    for ik in 1:Nk
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

    info = (;
        ham = H, ψ, grad, res, η, τ, basis, energies, converged = is_converged(info), norm_res = sqrt(abs(inner_product_DFTK(basis, res, res))), norm_grad = sqrt(abs(inner_product_DFTK(basis, res, grad))), ρ, eigenvalues, occupation, εF, n_iter,
        stage = :finalize, runtime_ns = time_ns() - start_ns, start_ns, algorithm = "RCG",
    )
    callback(info)

    info
end

function energy_adaptive_riemannian_conjugate_gradient(
        basis::PlaneWaveBasis{T};
        ρ = guess_density(basis),
        ψ = nothing,
        tol = 1.0e-6, maxiter = 100,
        callback = RcgDefaultCallback(),
        is_converged = RcgConvergenceResidual(tol),
        shift = CorrectedRelativeΛShift(μ = 0.01),
        inner_itmax = 100,
        inner_tol = 2.5e-2,
        h_solver = GlobalOptimalHSolver(),
        retraction = RetractionPolar(),
        cg_param = ParamFR_PRP(),
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        check_convergence_early = true, #check convergence before expensive gradient calculation
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.0, 0.1, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
    ) where {T}
    return riemannian_conjugate_gradient(
        basis; ρ, ψ, tol, maxiter, callback, is_converged,
        gradient = EAGradient(basis, shift; itmax = inner_itmax, tol = inner_tol, h_solver),
        retraction, cg_param, transport_η, transport_grad, check_convergence_early, iteration_strat
    )
end

function energy_adaptive_riemannian_gradient(
        basis::PlaneWaveBasis{T};
        ρ = guess_density(basis),
        ψ = nothing,
        tol = 1.0e-6, maxiter = 100,
        callback = RcgDefaultCallback(),
        is_converged = RcgConvergenceResidual(tol),
        shift = CorrectedRelativeΛShift(μ = 0.01),
        inner_itmax = 100,
        inner_tol = 5.0e-2,
        h_solver = GlobalOptimalHSolver(),
        retraction = RetractionPolar(),
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        check_convergence_early = true, #check convergence before expensive gradient calculation
        iteration_strat = StandardBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), BarzilaiBorweinStep(0.1, 2.5, ConstantStep(1.0)), 10)
    ) where {T}
    return riemannian_conjugate_gradient(
        basis; ρ, ψ, tol, maxiter, callback, is_converged,
        gradient = EAGradient(basis, shift; itmax = inner_itmax, tol = inner_tol, h_solver),
        retraction, cg_param = ParamZero(), transport_η, transport_grad, check_convergence_early, iteration_strat
    )
end

function h1_riemannian_conjugate_gradient(
        basis::PlaneWaveBasis{T};
        ρ = guess_density(basis),
        ψ = nothing,
        tol = 1.0e-6, maxiter = 500,
        callback = RcgDefaultCallback(),
        is_converged = RcgConvergenceResidual(tol),
        retraction = RetractionPolar(),
        cg_param = ParamFR_PRP(),
        transport_η = DifferentiatedRetractionTransport(),
        transport_grad = DifferentiatedRetractionTransport(),
        check_convergence_early = true, #check convergence before expensive gradient calculation
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
    ) where {T}
    return riemannian_conjugate_gradient(
        basis; ρ, ψ, tol, maxiter, callback, is_converged,
        gradient = H1Gradient(basis),
        retraction, cg_param, transport_η, transport_grad, check_convergence_early, iteration_strat
    )
end

function h1_riemannian_gradient(
    basis::PlaneWaveBasis{T};
    ρ = guess_density(basis),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 500,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    retraction = RetractionPolar(),
    transport_η = DifferentiatedRetractionTransport(),
    transport_grad = DifferentiatedRetractionTransport(),
    check_convergence_early = true, #check convergence before expensive gradient calculation
    iteration_strat = StandardBacktracking(NonmonotoneRule(0.95, 0.05, 0.5), BarzilaiBorweinStep(0.1, 2.5, ConstantStep(1.0)), 10)
) where {T}
return riemannian_conjugate_gradient(
    basis; ρ, ψ, tol, maxiter, callback, is_converged,
    gradient = H1Gradient(basis),
    retraction, cg_param = ParamZero(), transport_η, transport_grad, check_convergence_early, iteration_strat
)
end