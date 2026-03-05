"""
multilevel rcg variant
"""

rcg_coarse_solver(basis, gradient, maxiter; callback = (info) -> nothing) = function (ψ_c, ρ_c, Rres, tol)
    cost_residual = CoarseGridCostResidual(ψ_c, Rres)
    is_converged = RcgConvergenceResidualMGH(tol, 0.1, ψ_c)
    result = riemannian_conjugate_gradient(basis; ρ = ρ_c, ψ = ψ_c, 
        is_converged,
        maxiter,
        callback,
        cost_residual, gradient,         
        iteration_strat = AdaptiveBacktracking(
            ModifiedSecantRule(0.05, 0.2, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
        do_rayleigh_ritz = false,
    )
    return result.ψ
end

# RCG algorithm to solve SCF equations
DFTK.@timing function two_level_riemannian_optimization(
    basis_c_arr::Array,
    basis_f::PlaneWaveBasis{T};
    ρ = guess_density(basis),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 10,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    gradients_c = [EAGradient(basis_c) for basis_c = basis_c_arr],
    coarse_solvers = [rcg_coarse_solver(basis_c, gradient_c, maxiter_inner) for (basis_c, gradient_c) = zip(basis_c_arr, gradients_c)],
    coarse_cond_tol = 0.45,
    coarse_cond = ToleranceMinStepCoarseCondition(coarse_cond_tol, tol),
    coarse_density = RecalculateDensity(),
    coarse_model_tol = 1e-2,
    coarse_tol = RelativeResTolerance(coarse_model_tol, tol),
    cost_residual = StandardCostResiudal(),
    point_restriction = ProjectiveRestriction(),
    multilevel_map = PseudoInverse_1_2_Map(point_restriction),
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
) where {T}
    start_ns = time_ns()
    # setting parameters
    model_f = basis_f.model
    
    #@assert iszero(model_f.temperature)  # temperature is not yet supported
    @assert isnothing(model_f.εF)        # neither are computations with fixed Fermi level

    for k = 1:(length(basis_c_arr) - 1)
        @assert basis_c_arr[k].Ecut < basis_c_arr[k + 1].Ecut
    end

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
    Hψ_f, Λ, res, cost = initialize_cost_residual(H, ψ_f, energies.total, basis_f, cost_residual)

    η = nothing
    Rres = nothing
    
    cc = false

    for (basis_c, coarse_solver) = zip(basis_c_arr, coarse_solvers)
        # restrict point and residual
        ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
        Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)
        # calculate descent direction
        if check_coarse_condition(basis_c, basis_f, ψ_f, res, Rres, n_iter, coarse_cond)
            ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
            tol_c = get_coarse_tol(basis_c, basis_f, Rres, res, coarse_tol)
            ϕ_c = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
            η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
            cc = true
            break
        end
    end
    if !cc
        η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
    end
    push!(coarse_corrections, cc)

    #check if η is a descent direction. If not, restart
    desc = inner_product_DFTK(basis_f, η, res)
    if (desc >= 0)
        @warn "the search direction is not a descent direction, try to use a better initial guess"
    end

    #initial callback
    info = (;
        ham = H, ψ = ψ_f, res, Rres, η, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = nothing, ρout = ρ_f, coarse_corrections, n_iter,
        energies, cost, algorithm = "2gRG",
    )
    #callback(info)

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
                energies, cost, start_ns, algorithm = "2gRG",
            )
            callback(info)
            if is_converged(info)
                break
            end
        end


        # calculate new direction
        cc = false
    
        # TODO: iter cond --> norm cond --> angle cond
        for (basis_c, coarse_solver) = zip(basis_c_arr, coarse_solvers)
            # restrict point and residual
            ψ_c = restrict_point(basis_c, basis_f, ψ_f, point_restriction)
            Rres = restrict_vector(basis_c, basis_f, ψ_c, ψ_f, res, multilevel_map)
            # calculate descent direction
            if check_coarse_condition(basis_c, basis_f, ψ_f, res, Rres, n_iter, coarse_cond)
                ρ_c = calculate_coarse_density(basis_c, basis_f, ρ_f, ψ_c, ψ_f, coarse_density)
                tol_c = get_coarse_tol(basis_c, basis_f, Rres, res, coarse_tol)
                ϕ_c = coarse_solver(ψ_c, ρ_c, Rres, tol_c)
                η = prolongate_vector(basis_c, basis_f, ψ_c, ψ_f, invRet(ψ_c, ϕ_c), multilevel_map)
                cc = true
                break
            end
        end
        if !cc
            η = - calculate_gradient(ψ_f, Hψ_f, H, Λ, res, gradient)
        end
        push!(coarse_corrections, cc)

        #check convergence
        if !check_convergence_early
            info = (;
                ham = H, ψ = ψ_f, res, Rres, η, τ, basis = basis_f, converged = false, stage = :iterate, norm_res = norm_DFTK(basis_f, res), ρin = ρ_prev, ρout = ρ_f, coarse_corrections = coarse_corrections[1:end-1], n_iter,
                energies, cost, start_ns, algorithm = "2gRG",
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
        Hψk = H.blocks[ik] * ψ[ik]
        F = eigen(Hermitian(ψ[ik]'Hψk))
        push!(eigenvalues, F.values)
        if (do_rayleigh_ritz)
            ψ[ik] .= ψ[ik] * F.vectors
        end
    end

    εF = nothing  # does not necessarily make sense here, as the
    # Aufbau property might not even be true

    # return results and call callback one last time with final state for clean
    # up

    # λ_min = [eigmin(real(Λ[ik])) for ik = 1:Nk]
    # println(λ_min)

    info = (;
        ham = H, ψ = ψ_f, res, Rres, η, τ, basis = basis_f, energies, cost, converged = is_converged(info), norm_res = norm_DFTK(basis_f, res), ρ, eigenvalues, occupation, εF, coarse_corrections, n_iter,
        stage = :finalize, runtime_ns = time_ns() - start_ns, start_ns, algorithm = "2gRG",
    )
    callback(info)

    info
end

two_level_riemannian_optimization(basis_c::PlaneWaveBasis, basis_f::PlaneWaveBasis; kwargs...) = two_level_riemannian_optimization([basis_c], basis_f; kwargs...) 
two_level_riemannian_optimization(basis_arr::Array; kwargs...) = two_level_riemannian_optimization(basis_arr[1:end-1], basis_arr[end]; kwargs...)


# RCG algorithm to solve SCF equations
DFTK.@timing function multilevel_riemannian_optimization(
    basis_arr::Vector;
    ρ = guess_density(basis_arr[end]),
    ψ = nothing,
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 10,
    callback = RcgDefaultCallback(),
    callback_mid = (info) -> nothing,
    callback_coarse = (info) -> nothing,
    is_converged = RcgConvergenceResidual(tol),
    coarse_cond_tol = 0.45,
    coarse_steps_dist = 1,
    coarse_cond_functor = (tol) -> ToleranceMinStepCoarseCondition(coarse_cond_tol, tol; dist = coarse_steps_dist),
    coarse_density = RecalculateDensity(),
    coarse_model_tol = 1e-2,
    coarse_tol_functor = (tol) -> RelativeResTolerance(coarse_model_tol, tol),
    cost_residual = StandardCostResiudal(),
    gradients = [EAGradient(basis) for basis = basis_arr],
    point_restriction_functor = () -> ProjectiveRestriction(),
    multilevel_map_functor = (point_restriction) -> PseudoInverse_1_2_Map(point_restriction),
    retraction_functor = () -> RetractionPolar(),
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
)
    for k = 1:(length(basis_arr) - 1)
        @assert basis_arr[k].Ecut < basis_arr[k + 1].Ecut
    end

    if length(basis_arr) <= 1
        throw("Basis array needs to contain at least two elements")
    elseif length(basis_arr) == 2
        ml_coarse_solver = rcg_coarse_solver(basis_arr[1], gradients[1], maxiter_inner; callback = callback_coarse)
    else
        ml_coarse_solver = function (ψ_c, ρ_c, Rres, tol)
            cost_residual_inner = CoarseGridCostResidual(ψ_c, Rres)
            result = multilevel_riemannian_optimization(basis_arr[1:end-1]; ρ = ρ_c, ψ = ψ_c, 
                tol,
                maxiter = maxiter_inner,
                callback = callback_mid,
                is_converged = RcgConvergenceResidual(tol), #TODO what to use here?
                coarse_cond_functor,
                coarse_density,
                coarse_tol_functor,
                gradients = gradients[1:end-1],
                cost_residual = cost_residual_inner,
                point_restriction_functor,
                multilevel_map_functor,
                check_convergence_early,
                iteration_strats_fine = iteration_strats_fine[1:end-1],
                iteration_strats_coarse = iteration_strats_coarse[1:end-1],
                do_rayleigh_ritz = false
            )
            return result.ψ
        end
    end
    retraction = retraction_functor()
    point_restriction = point_restriction_functor()
    multilevel_map = multilevel_map_functor(point_restriction)
    gradient = gradients[end]
    coarse_cond = coarse_cond_functor(tol)
    coarse_tol = coarse_tol_functor(tol)
    return two_level_riemannian_optimization(basis_arr[end-1], basis_arr[end]; 
        ρ, ψ, tol, maxiter, maxiter_inner, callback, is_converged, 
        coarse_solvers = [ml_coarse_solver],
        coarse_cond, coarse_density, coarse_tol, cost_residual, 
        point_restriction, multilevel_map, gradient,retraction, check_convergence_early, 
        iteration_strat_fine = iteration_strats_fine[end], 
        iteration_strat_coarse = iteration_strats_coarse[end], 
        do_rayleigh_ritz)
end
