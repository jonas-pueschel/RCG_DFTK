"""
multilevel rcg variant
"""

rcg_coarse_solver(basis, gradient, maxiter; callback = (info) -> nothing, 
            coarse_cost_residual = CoarseGridCostResidual, 
            levels = Float64[],
            is_converged = nothing
            ) = function (ψ_c, ρ_c, Rres, tol)
                
    cr = coarse_cost_residual(ψ_c, Rres)
    is_converged_inner = isnothing(is_converged) ? RcgConvergenceResidualMGH(tol, 0.1, ψ_c) : 
                is_converged(tol);
    result = riemannian_conjugate_gradient(basis; ρ = ρ_c, ψ = ψ_c, 
        is_converged = is_converged_inner,
        maxiter,
        callback,
        cost_residual = cr, gradient,         
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.05, 0.2, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
        do_rayleigh_ritz = false,
        levels,
    )
    return result.ψ, result.ρ
end

# 2 level riemannian optimization algorithm
DFTK.@timing function two_level_riemannian_optimization(
    basis_c::PlaneWaveBasis{T},
    basis_f::PlaneWaveBasis{T};
    ρ = guess_density(basis),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 10,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    gradient_c = EAGradient(basis_c),
    coarse_cost_residual = CoarseGridCostResidual,
    levels = Float64[],  # Ecut of the basis of every gradient calculation (shared across levels)
    coarse_solver = rcg_coarse_solver(basis_c, gradient_c, maxiter_inner; coarse_cost_residual, levels),
    coarse_cond_tol = 0.45,
    coarse_steps_dist = 1,
    force_post_smoothing = false,
    coarse_cond = ToleranceMinStepCoarseCondition(coarse_cond_tol, tol; dist = coarse_steps_dist),
    coarse_density = RecalculateDensity(),
    coarse_model_tol = 1e-2,
    coarse_tol = RelativeResTolerance(coarse_model_tol),
    cost_residual = StandardCostResiudal(),
    point_restriction = ProjectiveRestriction(),
    multilevel_map = AlgebraicMap(point_restriction),
    gradient = EAGradient(basis_f),
    retraction = RetractionPolar(),
    check_convergence_early = true, 
    iteration_strat_fine = StandardBacktracking(
        ModifiedSecantRule(0.1, 0.45, 1e-13, 0.5),
        ConstantStep(1.0), 10
    ),
    iteration_strat_coarse = StandardBacktracking(
        ModifiedSecantRule(0.1, 0.45, 1e-13, 0.5),
        ConstantStep(1.0), 10
    ),
    do_rayleigh_ritz = true,
    n_iter = 0,
    terminate_on_non_descent = false,
) where {T}
    start_ns = time_ns()
    # the following objects carry state (of the current level and run): work on own copies,
    # the multilevel map may refer to the point restriction, hence they are copied together
    coarse_cond = init_coarse_condition!(deepcopy(coarse_cond), tol)
    point_restriction, multilevel_map, retraction = deepcopy((point_restriction, multilevel_map, retraction))
    # maps using the factorization of the point restriction (AlgebraicMap, GeometricMap) need the
    # very restriction used here, also if they were constructed with another one
    hasfield(typeof(multilevel_map), :pr) && (multilevel_map.pr = point_restriction)
    # setting parameters
    model_f = basis_f.model
    
    #@assert iszero(model_f.temperature)  # temperature is not yet supported
    @assert isnothing(model_f.εF)        # neither are computations with fixed Fermi level

    @assert basis_c.Ecut < basis_f.Ecut


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
    # n_iter = 0 #moved to fct head 
    coarse_corrections = []

    # orbitals, densities and energies to be updated along the iterations

    if isa(cost_residual, GreedyCostResidual)
        energies = (;total = Inf)
        H = nothing
        Hψ_f, Λ, res, cost = initialize_cost_residual(H, ψ_f, energies.total, basis_f, cost_residual)
    else
        energies, H = energy_hamiltonian(basis_f, ψ_f, occupation; ρ = ρ_f)

        # compute first residual
        Hψ_f, Λ, res, cost = initialize_cost_residual(H, ψ_f, energies.total, basis_f, cost_residual)
    end


    η = nothing
    
    cc = false
    # restrict point and residual
    ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
    Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)

    # initial info, the direction η is added below
    info = (;
        ham = H, ψ = ψ_f, res, Rres, η, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), norm_Rres = norm_DFTK(basis_c, Rres), ρin = nothing, ρout = ρ_f, coarse_corrections, n_iter,
        energies, cost, tol, algorithm = "2gRG", levels,
    )

    # calculate descent direction
    if check_coarse_condition(basis_c, basis_f, ψ_c, ψ_f, res, Rres, n_iter, coarse_cond)
        ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
        tol_c = get_coarse_tol(info, coarse_tol)
        ϕ_c, _ = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
        η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
        cc = true
    else 
        η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
        push!(levels, basis_f.Ecut)
        #println("Ecut: $(basis_f.Ecut); gradient step")
    end
    push!(coarse_corrections, cc)

    #check if η is a descent direction. If not, restart
    desc = inner_product_DFTK(basis_f, η, res)
    if (desc >= 0)
        @warn "the search direction at Ecut = $(basis_f.Ecut) is not a descent direction, try to use a better initial guess"
        terminate_on_non_descent && (return merge(info, (;ρ = ρ_f, stage = :finalize, runtime_ns = time_ns() - start_ns)))
    end

    #initial callback
    info = merge(info, (; η))
    check_convergence_early || callback(info)

    τ = nothing

    # give β the information from first iteration


    # perform iterations
    while n_iter < maxiter
        n_iter += 1

        #perform step
        get_next(τ_trial) = get_next_rcg(basis_f, occupation, ψ_f, η, τ_trial, retraction, NoTransport(), cost_residual)
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
                ham = H, ψ = ψ_f, res, Rres, η, τ, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = ρ_prev, ρout = ρ_f, coarse_corrections, n_iter,
                energies, cost, tol, start_ns, algorithm = "2gRG", levels,
            )
            callback(info)
            if is_converged(info) && (!force_post_smoothing || !coarse_corrections[end])
                break
            end
            # do not compute a direction (possibly a coarse correction) that is never used
            n_iter >= maxiter && break
        end


        # calculate new direction
        cc = false
    
        # TODO: iter cond --> norm cond --> angle cond
        # restrict point and residual
        ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
        Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)
        # calculate descent direction
        if check_coarse_condition(basis_c, basis_f, ψ_c, ψ_f, res, Rres, n_iter, coarse_cond)
            ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
            # info does not contain the current Rres yet (and without check_convergence_early
            # not even the current iterate)
            info_c = merge(info, (; ψ = ψ_f, res, Rres, norm_res = norm_DFTK(basis_f, res),
                                    norm_Rres = norm_DFTK(basis_c, Rres), n_iter))
            tol_c = get_coarse_tol(info_c, coarse_tol)
            ϕ_c, _ = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
            η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
            cc = true
        end

        if !cc
            η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
            push!(levels, basis_f.Ecut)
            #println("Ecut: $(basis_f.Ecut); gradient step")
        end
        push!(coarse_corrections, cc)

        #check convergence
        if !check_convergence_early
            info = (;
                ham = H, ψ = ψ_f, res, Rres, η, τ, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = ρ_prev, ρout = ρ_f, coarse_corrections = coarse_corrections[1:end-1], n_iter,
                energies, cost, tol, start_ns, algorithm = "2gRG", levels,
            )
            callback(info)
            if is_converged(info) && (!force_post_smoothing || !coarse_corrections[end-1])
                break
            end
        end

        #check if η is a descent direction. If not, restart
        desc = inner_product_DFTK(basis_f, η, res)
        if (desc >= 0)
            @warn "the search direction at Ecut = $(basis_f.Ecut) is not a descent direction, try to use a better initial guess"
            terminate_on_non_descent && (return merge(info, (;ρ = ρ_f, stage = :finalize, runtime_ns = time_ns() - start_ns)))
        end
    end

    # Rayleigh-Ritz
    eigenvalues = []
    for ik in 1:Nk
        Hψk = H.blocks[ik] * ψ_f[ik]
        F = eigen(Hermitian(ψ_f[ik]'Hψk))
        push!(eigenvalues, F.values)
        if (do_rayleigh_ritz)
            ψ_f[ik] .= ψ_f[ik] * F.vectors
        end
    end

    εF = nothing  # does not necessarily make sense here, as the
    # Aufbau property might not even be true

    # return results and call callback one last time with final state for clean
    # up

    # λ_min = [eigmin(real(Λ[ik])) for ik = 1:Nk]
    # println(λ_min)

    info = (;
        ham = H, ψ = ψ_f, res, Rres, η, τ, basis = basis_f, energies, cost, converged = is_converged(info), norm_res = norm_DFTK(basis_f, res), 
        ρ = ρ_f, eigenvalues, occupation, εF, coarse_corrections, n_iter,
        stage = :finalize, runtime_ns = time_ns() - start_ns, start_ns, tol, algorithm = "2gRG", levels,
    )
    callback(info)

    info
end

#two_level_riemannian_optimization(basis_c::PlaneWaveBasis, basis_f::PlaneWaveBasis; kwargs...) = two_level_riemannian_optimization([basis_c], basis_f; kwargs...) 
two_level_riemannian_optimization(basis_arr::Array; kwargs...) = two_level_riemannian_optimization(basis_arr[1:end-1], basis_arr[end]; kwargs...)


# RCG algorithm to solve SCF equations
DFTK.@timing function multilevel_riemannian_optimization(
    basis_arr::Vector;
    ρ = guess_density(basis_arr[end]),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 100,
    callback = RcgDefaultCallback(),
    callback_mid = (info) -> nothing,
    callback_coarse = (info) -> nothing,
    is_converged = RcgConvergenceResidual(tol),
    coarse_cond_tol = 0.45,
    coarse_steps_dist = 1,
    force_post_smoothing = false,
    # the tolerance of the condition is the one of the respective level, see init_coarse_condition!
    coarse_cond = ToleranceMinStepCoarseCondition(coarse_cond_tol; dist = coarse_steps_dist),
    coarse_density = RecalculateDensity(),
    coarse_cost_residual = CoarseGridCostResidual,
    coarse_model_tol = 1e-2,
    coarse_tol = RelativeResTolerance(coarse_model_tol),
    cost_residual = StandardCostResiudal(),
    gradients = [EAGradient(basis) for basis = basis_arr],
    # copied on every level by two_level_riemannian_optimization
    point_restriction = ProjectiveRestriction(),
    multilevel_map = AlgebraicMap(point_restriction),
    retraction = RetractionPolar(),
    check_convergence_early = true, 
    iteration_strats_fine = [StandardBacktracking(
        ModifiedSecantRule(0.1, 0.45, 1e-13, 0.5),
        ConstantStep(1.0), 10
    ) for basis = basis_arr[2:end]],
    iteration_strats_coarse = [StandardBacktracking(
        ModifiedSecantRule(0.1, 0.45, 1e-13, 0.5),
        ConstantStep(1.0), 10
    ) for basis =  basis_arr[2:end]],
    do_rayleigh_ritz = true,
    n_iter = 0,
    levels = Float64[],
    cycle = nothing,
    cycle_finest = cycle,
    terminate_on_non_descent = false,
)
    for k = 1:(length(basis_arr) - 1)
        @assert basis_arr[k].Ecut < basis_arr[k + 1].Ecut
    end

    if length(basis_arr) <= 1
        throw("Basis array needs to contain at least two elements")
    elseif length(basis_arr) == 2
        ml_coarse_solver = rcg_coarse_solver(basis_arr[1], gradients[1], maxiter_inner; callback = callback_coarse, coarse_cost_residual, levels)
    else
        ml_coarse_solver = function (ψ_c, ρ_c, Rres, tol)
            cost_residual_inner = coarse_cost_residual(ψ_c, Rres)
            result = multilevel_riemannian_optimization(basis_arr[1:end-1]; ρ = ρ_c, ψ = ψ_c, 
                tol,
                maxiter = isnothing(cycle) ? maxiter_inner : cycle.γ * cycle_length(cycle),
                maxiter_inner,
                callback = callback_mid,
                callback_mid,
                callback_coarse,
                is_converged = isnothing(cycle) ? RcgConvergenceResidual(tol) : (info -> false), #TODO what to use here?
                cycle,
                coarse_cond,
                force_post_smoothing,
                coarse_density,
                coarse_cost_residual,
                coarse_tol,
                gradients = gradients[1:end-1],
                cost_residual = cost_residual_inner,
                point_restriction,
                multilevel_map,
                retraction,
                check_convergence_early,
                iteration_strats_fine = iteration_strats_fine[1:end-1],
                iteration_strats_coarse = iteration_strats_coarse[1:end-1],
                do_rayleigh_ritz = false,
                levels,
                terminate_on_non_descent
            )
            return result.ψ, result.ρ
        end
    end
    gradient = gradients[end]
    # not reassigning coarse_cond, as it is captured by ml_coarse_solver for the inner levels
    coarse_cond_finest = isnothing(cycle) ? coarse_cond : cycle_finest
    return two_level_riemannian_optimization(basis_arr[end-1], basis_arr[end]; 
        ρ, ψ, tol, maxiter, maxiter_inner, callback, is_converged, 
        coarse_solver = ml_coarse_solver,
        force_post_smoothing,
        coarse_cond = coarse_cond_finest, coarse_density, coarse_tol, cost_residual, 
        point_restriction, multilevel_map, gradient,retraction, check_convergence_early, 
        iteration_strat_fine = iteration_strats_fine[end], 
        iteration_strat_coarse = iteration_strats_coarse[end], 
        do_rayleigh_ritz,
        n_iter,
        terminate_on_non_descent,
        levels)
end
