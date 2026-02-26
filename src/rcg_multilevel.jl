"""
multilevel rcg variant
"""

default_coarse_solver(basis_c) = function (ψ_c, ρ_c, Rres, tol)
    callback = (info) -> nothing
    cost_resiudal = CoarseGridCostResidual(ψ_c, Rres)
    is_converged = RcgConvergenceResidualMGH(tol, 0.1, ψ_c)
    result = riemannian_conjugate_gradient(basis_c; ρ = ρ_c, ψ = ψ_c, 
        is_converged,
        callback,
        cost_resiudal, gradient = H1Gradient(basis_c),         
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.05, 0.25, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
        do_rayleigh_ritz = false,
    )
    return result.ψ
end

# RCG algorithm to solve SCF equations
DFTK.@timing function two_level_riemannian_optimization(
    basis_c::PlaneWaveBasis{T},
    basis_f::PlaneWaveBasis{T};
    ρ = guess_density(basis),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 100,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    coarse_solver = default_coarse_solver(basis_c),
    coarse_cond = ToleranceMinStepCoarseCondition(0.45, tol),
    coarse_density = InterpolateDensity(),
    coarse_tol = RelativeResTolerance(1e-3, tol),
    cost_resiudal = StandardCostResiudal(),
    point_restriction = ProjectiveRestriction(),
    multilevel_map = PseudoInverse_1_2_Map(point_restriction),
    gradient =  H1Gradient(basis_f),
    retraction = RetractionPolar(),
    check_convergence_early = true, 
    iteration_strat_fine = StandardBacktracking(
        ArmijoRule(0.1, 0.5),
        ApproxHessianStep(), 10
    ),
    iteration_strat_coarse = AdaptiveBacktracking(
        ModifiedSecantRule(0.05, 0.1, 1.0e-12, 0.5),
        ConstantStep(1.0), 10
    ),
) where {T}
    start_ns = time_ns()
    # setting parameters
    model_f = basis_f.model
    
    #@assert iszero(model_f.temperature)  # temperature is not yet supported
    @assert isnothing(model_f.εF)        # neither are computations with fixed Fermi level

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model_f)
    n_spin = model_f.n_spin_components
    n_bands = div(model_f.n_electrons, n_spin * filled_occ, RoundUp)

    ψ_f = ψ
    ρ_f = ρ
    if !isnothing(ψ_f)
        @assert length(ψ_f) == length(basis_f.kpoints)
        @assert n_bands == size(ψ_f[1], 2)
    else
        ψ_f = [DFTK.random_orbitals(basis_f, kpt, n_bands) for kpt in basis_f.kpoints]
    end

    if isnothing(ρ_f)
        ρ_f = guess_density(basis_f)
    end


    # number of kpoints and occupation
    Nk = length(basis_f.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik in 1:Nk]

    # iterators
    n_iter = 0
    coarse_corrections = []

    # orbitals, densities and energies to be updated along the iterations

    energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)

    # compute first residual
    Hψ_f, Λ, res, cost = calculate_cost_residual(H, ψ_f, energies.total, basis_f, cost_resiudal)

    # restrict point and residual
    ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
    Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)

    # calculate descent direction

    if check_coarse_condition(basis_c, basis_f, ψ_f, res, Rres, coarse_cond)
        ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
        tol_c = get_coarse_tol(basis_c, basis_f, Rres, res, coarse_tol)
        ϕ_c = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
        η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
        push!(coarse_corrections, true)
    else
        η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
        push!(coarse_corrections, false)
    end


    desc = inner_product_DFTK(basis_f, res, η)

    #initial callback
    info = (;
        ham = H, ψ_f, res, η, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = nothing, ρout = ρ_f, coarse_corrections, n_iter,
        energies, cost, algorithm = "RCG",
    )
    #callback(info)

    τ = nothing

    # give β the information from first iteration


    # perform iterations
    while n_iter < maxiter
        n_iter += 1

        #perform step
        get_next(τ_trial) = get_next_rcg(basis_f, occupation, ψ_f, η, τ_trial, retraction, NoTransport(), cost_resiudal)
        iteration_strat = coarse_corrections[end] ? iteration_strat_coarse : iteration_strat_fine
        next = do_step(basis_f, ψ_f, η, nothing, res, nothing, desc, Λ, H, ρ_f, cost, get_next, iteration_strat)

        #update orbitals, density, H and energies
        ψ_f = next.ψ_next
        ρ_prev = ρ_f
        ρ_f = next.ρ_next
        H = next.H_next
        τ = next.τ
        energies = next.energies_next
        cost = next.cost_next
        Hψ_f = next.Hψ_next
        Λ = next.Λ_next
        res = next.res_next

        # test convergence before expensive direction calculation
        if check_convergence_early
            #info contains new res, but old η!
            info = (;
                ham = H, ψ_f, res, η, τ, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = ρ_prev, ρout = ρ, coarse_corrections, n_iter,
                energies, cost, start_ns, algorithm = "RCG",
            )
            callback(info)
            if is_converged(info)
                break
            end
        end


        # calculate new direction

        # restrict point and residual
        ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
        Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)

        # calculate descent direction
        if check_coarse_condition(basis_c, basis_f, ψ_f, res, Rres, coarse_cond)
            ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
            tol_c = get_coarse_tol(basis_c, basis_f, Rres, res, coarse_tol)
            ϕ_c = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
            η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
            push!(coarse_corrections, true)
        else
            η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
            push!(coarse_corrections, false)
        end


        #check convergence
        if !check_convergence_early
            info = (;
                ham = H, ψ_f, res, η, τ, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = ρ_prev, ρout = ρ, coarse_corrections, n_iter,
                energies, cost, start_ns, algorithm = "RCG",
            )
            callback(info)
            if is_converged(info)
                break
            end
        end

        #check if η is a descent direction. If not, restart
        desc = inner_product_DFTK(basis_f, η, res)
        if (desc >= 0)
            @warn "the search direction is not a descent direction, try to use a better initial guess"
        end
    end

    # Rayleigh-Ritz
    eigenvalues = []
    for ik in 1:Nk
        Hψ_fk = H.blocks[ik] * ψ_f[ik]
        F = eigen(Hermitian(ψ_f[ik]'Hψ_fk))
        push!(eigenvalues, F.values)
        ψ_f[ik] .= ψ_f[ik] * F.vectors
    end

    εF = nothing  # does not necessarily make sense here, as the
    # Aufbau property might not even be true

    # return results and call callback one last time with final state for clean
    # up

    # λ_min = [eigmin(real(Λ[ik])) for ik = 1:Nk]
    # println(λ_min)

    info = (;
        ham = H, ψ_f, res, η, τ, basis = basis_f, energies, cost, converged = is_converged(info), norm_res = norm_DFTK(basis_f, res), ρ, eigenvalues, occupation, εF, coarse_corrections, n_iter,
        stage = :finalize, runtime_ns = time_ns() - start_ns, start_ns, algorithm = "RCG",
    )
    callback(info)

    info
end
