function coarsest_solve(basis, gradient, ψ_c, ρ_c; maxiter = 100, tol = 1e-2, callback = (info) -> nothing)
    is_converged = RcgConvergenceErrEstimate(tol)
    result = riemannian_conjugate_gradient(basis; ρ = ρ_c, ψ = ψ_c, 
        is_converged,
        maxiter,
        callback,
        gradient,
        do_rayleigh_ritz = false
    )
    return result.ψ
end


DFTK.@timing function prolongation_based_two_level()

end

DFTK.@timing function riemannian_full_multigrid(
    basis_arr::Vector;
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 100,
    callback = RcgDefaultCallback(),
    is_converged = RcgConvergenceResidual(tol),
    coarse_cond_tol = 0.45,
    coarse_steps_dist = 1,
    force_post_smoothing = false,
    coarse_cond_functor = (tol) -> ToleranceMinStepCoarseCondition(coarse_cond_tol, tol; dist = coarse_steps_dist),
    coarse_density = RecalculateDensity(),
    coarse_cost_residual = CoarseGridCostResidual,
    coarse_model_tol = 1e-2,
    coarse_tol_functor = (tol) -> RelativeResTolerance(coarse_model_tol, tol),
    cost_residual = StandardCostResiudal(),
    gradients = [EAGradient(basis) for basis = basis_arr],
    point_restriction_functor = () -> ProjectiveRestriction(),
    multilevel_map_functor = (point_restriction) -> AlgebraicMap(point_restriction),
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
    end

    basis_arr_curr = [basis_arr[1]]
    basis_next = basis_arr[2]
    n_iter = 0

    ψ_c = coarsest_solve(basis_arr[1], gradients[1], ψ, ρ; tol = coarse_model_tol)



    while n_iter < maxiter  
        n_iter += 1

        # solve 2-level

        # get residual from basis_next

        # error estimation; if possible break, if necessary refine



        if length(basis_arr_curr) == 1
            ml_coarse_solver = rcg_coarse_solver(basis_arr_curr[1], gradients[1], maxiter_inner; callback = callback_coarse, coarse_cost_residual)
        else
            ml_coarse_solver = function (ψ_c, ρ_c, Rres, tol)
                cost_residual_inner = coarse_cost_residual(ψ_c, Rres)
                result = multilevel_riemannian_optimization(basis_arr_curr; ρ = ρ_c, ψ = ψ_c, 
                    tol,
                    maxiter = maxiter_inner,
                    callback = callback_mid,
                    is_converged = RcgConvergenceResidual(tol), #TODO what to use here?
                    coarse_cond_functor,
                    force_post_smoothing,
                    coarse_density,
                    coarse_cost_residual,
                    coarse_tol_functor,
                    gradients = gradients[1:length(basis_arr_curr)],
                    cost_residual = cost_residual_inner,
                    point_restriction_functor,
                    multilevel_map_functor,
                    check_convergence_early,
                    iteration_strats_fine = iteration_strats_fine[1:length(basis_arr_curr)],
                    iteration_strats_coarse = iteration_strats_coarse[1:length(basis_arr_curr)],
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
    end

    return 
end