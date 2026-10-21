function coarsest_solve(basis, ψ_c, ρ_c; maxiter = 100, tol = 1e-2, callback = (info) -> nothing, levels = Float64[])
    is_converged = RcgConvergenceErrEstimate(tol)
    result = riemannian_conjugate_gradient(basis; ρ = ρ_c, ψ = ψ_c, 
        is_converged,
        maxiter,
        callback,
        gradient = H1Gradient(basis),
        do_rayleigh_ritz = false,
        levels,
    )
    return result.ψ, result.ρ
end


DFTK.@timing function riemannian_full_multigrid(
    basis_arr::Vector, basis_arr_ref::Vector; ψ = nothing, ρ = nothing,
    tol = 1.0e-6, maxiter = 100, maxiter_inner = 100,
    callback = RfmgDefaultCallback(),
    callback_mid = (info) -> nothing,
    callback_coarse = (info) -> nothing,
    #TODO: Error estimator
    refinement_tol = 0.45,
    # norm of the error estimate and refinement decision, see AbstractRefinementCondition
    refinement_cond = EstimateL2Norm(refinement_tol),
    coarse_cond_tol = 0.45,
    coarse_steps_dist = 1,
    force_post_smoothing = false,
    coarse_cond_functor = (tol) -> ToleranceMinStepCoarseCondition(coarse_cond_tol, tol; dist = coarse_steps_dist),
    coarse_density = RecalculateDensity(),
    coarse_model_tol = 1e-2, 
    #TODO this does not have to be a functor, does it?
    coarse_tol_functor = (tol) -> RelativeResTolerance(coarse_model_tol, tol),
    coarse_model_tol_outer = 1e-2, 
    coarse_tol_functor_outer = (tol) -> RelativeResTolerance(coarse_model_tol_outer, tol),
    #cost_residual = StandardCostResiudal(), #TODO what happens with this?
    gradients = [EAGradient(basis) for basis = basis_arr],
    point_restriction_functor = () -> ProjectiveRestriction(),
    multilevel_map_functor = (point_restriction) -> AlgebraicMap(point_restriction),
    retraction_functor = () -> RetractionPolar(),
    check_convergence_early = true, 
    iteration_strats_fine = [RestartedGreedySecantStrategy(ConstantStep(1.0), 0.1) for basis = basis_arr[1:end]],
    iteration_strats_coarse = [RestartedGreedySecantStrategy(ConstantStep(1.0), 0.1) for basis =  basis_arr[1:end]],
    do_rayleigh_ritz = true,
    # line search for the fine energy along the prolongated coarse correction
    iteration_strat_outer = RestartedGreedySecantStrategy(ConstantStep(1.0), 0.1),
    # if true, the coarse problems of the outer iteration are solved without the coarse grid
    # correction from the fine residual, i.e. with StandardCostResiudal() instead of
    # CoarseGridCostResidual(ψ_c, Rres)
    # TODO: Ecut growth and Ecut dist? i.e. dynamically generate basis?
    basis_distance = 1,
    greedy_cost_residual = false,
    only_rcg = false,
    levels = Float64[],  # Ecut of the basis of every gradient calculation (shared across levels)
    # classical cycle (e.g. VCycle(1, 1), WCycle(1, 1), SawtoothCycle(1)) for the multilevel coarse solves,
    # see multilevel_riemannian_optimization. The cycles are repeated until the coarse tolerance is met.
    cycle = nothing,
    cycle_finest = cycle,
)
    start_ns = time_ns()
    @assert length(basis_arr) == length(basis_arr_ref)
    for k = 1:(length(basis_arr) - 1)
        @assert basis_arr[k].Ecut < basis_arr[k + 1].Ecut
        @assert basis_arr_ref[k].Ecut < basis_arr_ref[k + 1].Ecut
    end
    if length(basis_arr) <= 1
        throw("Basis array needs to contain at least two elements")
    end

    basis_arr_curr = [basis_arr[1]]
    basis_ref = basis_arr_ref[1]
    model_f = basis_ref.model
    basis_curr = basis_arr[1]

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model_f)
    n_spin = model_f.n_spin_components
    n_bands = div(model_f.n_electrons, n_spin * filled_occ, RoundUp)
    # number of kpoints and occupation
    Nk = length(basis_ref.kpoints)
    occupation = [filled_occ * ones(eltype(basis_ref), n_bands) for ik in 1:Nk]

    if (isnothing(ψ) || isnothing(ρ))
        ψ_c, ρ_c = coarsest_solve(basis_arr[1], ψ, ρ; tol = coarse_model_tol, levels)
    else
        ψ_c, ρ_c = ψ, ρ
    end
    retraction = retraction_functor()
    transport = DifferentiatedRetractionTransport()

    # fine grid quantities at the prolongated current point
    fine = evaluate_prolongated(basis_curr, basis_ref, occupation, ψ_c)
    ρ_prev = nothing
    local Rres, est_err, est_err_coarse, est_err_norm, est_err_perc
    converged = false
    τ = nothing  # step size of the line search leading to the current iterate

    depth = 1
    n_iter = 0
    while n_iter < maxiter  
        n_iter += 1
        (; ψ_f, ρ_f, energies, H, res) = fine

        # all three vector maps are identcal here!
        Rres = restrict_vector(basis_curr, basis_ref, ψ_c, ψ_f, res, ProjectionMap())

        #TODO debug: relative size of the fine grid correction ‖Rres − res_c(ψ_c)‖ / ‖Rres‖
        # ρ_c_dbg = DFTK.compute_density(basis_curr, ψ_c, occupation)
        # energies_c_dbg, H_c_dbg = energy_hamiltonian(basis_curr, ψ_c, occupation; ρ = ρ_c_dbg)
        # _, _, res_c_dbg, _ = calculate_cost_residual(H_c_dbg, ψ_c, energies_c_dbg.total, basis_curr, StandardCostResiudal())
        # quot = norm_DFTK(basis_curr, Rres - res_c_dbg) / norm_DFTK(basis_curr, Rres)
        # println(quot)

        # error estimation; if possible break, if necessary refine
        # TODO different error estimators

        est_err = invert_refinement_metric_improved(basis_ref, ψ_f, res)
        est_err_coarse = restrict_vector(basis_curr, basis_ref, ψ_c, ψ_f, est_err, ProjectionMap())

        #TODO this is different than inverting the metric on Rres! not clear what is the best choice!
        est_info = (; basis = basis_curr, basis_ref, est_err, est_err_coarse)
        (; err, err_coarse, perc) = estimate_error_norms(refinement_cond, est_info)
        est_err_norm, est_err_perc = err, perc
        converged = est_err_norm < tol
        # fallback on the finest level to avoid getting stuck: the remaining error is (almost)
        # only discretization error, which cannot be reduced further; not reported as converged
        stagnated = !converged && depth == length(basis_arr) && err_coarse < tol * 1e-2
        info = (;
            ham = H, ψ = ψ_c, res, Rres, basis = basis_curr, basis_ref, converged, stage = :iterate, norm_res = norm_DFTK(basis_ref, res), norm_Rres = norm_DFTK(basis_curr, Rres), ρin = ρ_prev, ρout = ρ_f, n_iter,
            energies, cost = energies.total, est_err, est_err_coarse, est_err_norm, est_err_perc, τ, start_ns, algorithm = "RFMG", levels,
        )
        callback(info)
        if converged || stagnated
            break;
        end


        if refinement_cond(info)
            if depth == length(basis_arr)
                @warn "cannot refine anymore"
                # the error estimate of the finest level is computed once per iterate
                # and shared between the convergence check and the callback
                # is_converged = RcgConvergenceErrEstimate(tol)
                # cb_ml = function (info)
                #     if info.stage == :iterate && !haskey(info, :est_err)
                #         info = merge(info, (; est_err = estimate_error!(is_converged, info)))
                #     end
                #     callback(info)
                # end
                # info = multilevel_riemannian_optimization(
                #     basis_arr;
                #     ψ = ψ_f,
                #     ρ = ρ_f,
                #     tol,
                #     is_converged,
                #     maxiter, maxiter_inner,
                #     callback = cb_ml,
                #     callback_mid,
                #     callback_coarse,
                #     coarse_cond_tol,
                #     coarse_steps_dist,
                #     force_post_smoothing,
                #     coarse_cond_functor,
                #     coarse_density,
                #     coarse_model_tol,
                #     coarse_tol_functor,
                #     gradients,
                #     point_restriction_functor,
                #     multilevel_map_functor,
                #     retraction_functor,
                #     check_convergence_early,
                #     iteration_strats_fine,
                #     iteration_strats_coarse,
                #     do_rayleigh_ritz,
                #     n_iter,
                # )
                # # the runtime of the multilevel method does not include the RFMG iterations
                # return merge(info, (; runtime_ns = time_ns() - start_ns, start_ns))
                break;
            else
                #need of refinement
                #TODO extrapolate prev density?
                depth += 1
                basis_arr_curr = basis_arr[1:depth]
                basis_ref = basis_arr_ref[depth]
                basis_prev = basis_curr
                basis_curr = basis_arr[depth]
                # fine grid quantities on the new fine grid (also needed for the line search)
                ψ_c = prolongate_point(basis_prev, basis_curr, ψ_c)
                ρ_c = DFTK.compute_density(basis_curr, ψ_c , occupation)

                fine = evaluate_prolongated(basis_curr, basis_ref, occupation, ψ_c)
                (; ψ_f, ρ_f, energies, H, res) = fine
                Rres = restrict_vector(basis_curr, basis_ref, ψ_c, ψ_f, res, ProjectionMap())
            end
        end

        # solve 2-level
        basis_idx = [depth - basis_distance*i for i = Int(floor((depth - 1)/basis_distance)):-1:0]
        if length(basis_idx) == 1 || only_rcg
            coarse_cost_residual = greedy_cost_residual ? ((ψ_c, Rres) -> StandardCostResiudal()) :
                                                          CoarseGridCostResidual
            ml_coarse_solver = rcg_coarse_solver(basis_arr_curr[depth], gradients[depth], maxiter_inner; 
                callback = callback_coarse, coarse_cost_residual, levels)
        else
            ml_coarse_solver = function (ψ_c, ρ_c, Rres, tol)
                cost_residual_inner = greedy_cost_residual ? StandardCostResiudal() :
                                                             CoarseGridCostResidual(ψ_c, Rres)
                result = multilevel_riemannian_optimization(basis_arr_curr[basis_idx]; ρ = ρ_c, ψ = ψ_c, 
                    tol,
                    maxiter = maxiter_inner,
                    maxiter_inner,
                    callback = callback_mid,
                    callback_mid,
                    callback_coarse,
                    is_converged = RcgConvergenceResidual(tol), # TODO what to use here?
                    coarse_cond_functor,
                    force_post_smoothing,
                    coarse_density,
                    coarse_tol_functor,
                    gradients = gradients[basis_idx],
                    cost_residual = cost_residual_inner,
                    point_restriction_functor,
                    multilevel_map_functor,
                    check_convergence_early,
                    iteration_strats_fine = iteration_strats_fine[1:length(basis_arr_curr)],
                    iteration_strats_coarse = iteration_strats_coarse[1:length(basis_arr_curr)],
                    do_rayleigh_ritz = false,
                    levels,
                    cycle,
                    cycle_finest,
                )
                return result.ψ, result.ρ
            end
        end
        coarse_tol = coarse_tol_functor_outer(tol)
        tol_c = get_coarse_tol(basis_curr, basis_ref, Rres, res, coarse_tol)
        ψ_c_sol, ρ_c_sol = ml_coarse_solver(ψ_c, ρ_c, Rres, tol_c)

        # line search for the fine energy along the prolongated coarse correction
        η_c = invRet(ψ_c, ψ_c_sol)
        η_f = prolongate_vector(basis_curr, basis_ref, ψ_c, ψ_f, η_c, ProjectionMap())
        # should be the same as fine desc
        desc = inner_product_DFTK(basis_ref, η_c, Rres)
        get_next = τ -> get_next_prolongated(basis_curr, basis_ref, occupation, ψ_c, η_c, τ, retraction, transport)
        if desc < 0
            next = do_step(basis_ref, ψ_f, η_f, nothing, res, nothing, desc, fine.Λ, H, ρ_f, energies.total,
                           get_next, iteration_strat_outer)
        else
            @warn "the coarse correction is not a descent direction on the fine grid, taking the full step"
            next = get_next(1.0)
        end

        # the coarse density is only used as initial guess of the coarse solver
        ψ_c = next.ψ_c_next
        τ = next.τ
        ρ_c = next.τ == 1 ? ρ_c_sol : DFTK.compute_density(basis_curr, ψ_c, occupation)
        ρ_prev = ρ_f
        # the fine residual at the next point is already known from the line search
        fine = (; ψ_f = next.ψ_next, ρ_f = next.ρ_next, energies = next.energies_next, cost = next.cost_next,
                  H = next.H_next, Hψ = next.Hψ_next, Λ = next.Λ_next, res = next.res_next)
    end

    (; ψ_f, ρ_f, energies, H, res) = fine

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
        ham = H, ψ = ψ_c, res, Rres, basis = basis_curr, basis_ref, energies, cost= energies.total, converged, norm_res = norm_DFTK(basis_ref, res), norm_Rres = norm_DFTK(basis_curr, Rres), 
        ρ = ρ_f, eigenvalues, occupation, εF, n_iter,
        stage = :finalize, runtime_ns = time_ns() - start_ns, start_ns, est_err, est_err_coarse, est_err_norm, est_err_perc, algorithm = "RFMG", levels,
    )


    callback(info)
    info
end


"""
    RfmgDefaultCallback(; show_time = true, lpadding = "", basis_arr = nothing, ψ_ref = nothing)

Prints one line per iteration and records the history of the current run in `times`
(runtime in seconds since the start of the method, without the time spent in this
callback), `dists` (Grassmann distance to `ψ_ref`, `NaN` if not available), `n_iters` and
`Ecuts`, and `dists_h1` (H1 distance of the L2-closest representatives, see `h1_dist_DFTK`).
The history is reset whenever a new run starts (`n_iter == 1`).
`ψ_ref` is an optional reference solution on the finest grid `basis_arr[end]`; it can also
be set afterwards via `callback.ψ_ref = ...`.
"""
mutable struct RfmgDefaultCallback
    show_time
    lpadding
    basis_ref
    ψ_ref
    prev_time
    prev_cost
    start_ns
    overhead_ns
    times
    dists
    dists_h1
    n_iters
    Ecuts
end
function RfmgDefaultCallback(; show_time = true, lpadding = "", basis_ref = nothing, ψ_ref = nothing)
    RfmgDefaultCallback(show_time, lpadding, basis_ref, ψ_ref, nothing, NaN, nothing, 0,
                        Float64[], Float64[], Float64[], Int[], Float64[])
end

function (cb::RfmgDefaultCallback)(info)
    entry_ns = time_ns()
    (; show_time, lpadding, basis_ref, ψ_ref) = cb
    show_dist = !isnothing(basis_ref) && !isnothing(ψ_ref)
    #!mpi_master() && return info  # Rest is printing => only do on master
    if info.stage == :finalize
        if (haskey(info, :converged))
            info.converged || @warn "$(info.algorithm) not converged."
        end
        return info
    end
    if info.n_iter == 1
        cb.prev_time = nothing
        cb.prev_cost = NaN
        # start of the method: take it from the info if possible
        cb.start_ns = haskey(info, :start_ns) ? info.start_ns :
                      haskey(info, :runtime_ns) ? entry_ns - info.runtime_ns : entry_ns
        cb.overhead_ns = 0
        empty!(cb.times); empty!(cb.dists); empty!(cb.dists_h1); empty!(cb.n_iters); empty!(cb.Ecuts)
        println("$(lpadding)n     Cost              log10|R|   log10|RR|   log10(ΔC)    log10(Δρ)   log10(err)$(show_dist ? "   log10(dist)   log10(dH1)" : "")   err_c %   τ         Ecut     Δtime  ")
        println("$(lpadding)---   ---------------   ---------   ---------  ----------   ---------   ----------$(show_dist ? "   -----------   ----------" : "")   -------   -------   ------   -------")
    end
    cost = haskey(info, :cost) ? info.cost : info.energies.total
    cost = isnothing(cost) ? Inf : cost
    Δρ = !haskey(info, :ρin) || !haskey(info, :ρout) || isnothing(info.ρin) || (size(info.ρout) != size(info.ρin)) ?
        nothing : norm(info.ρout - info.ρin) * sqrt(abs(info.basis.dvol))

    tstr = " "^7
    if show_time && !isnothing(cb.prev_time)
        tstr = @sprintf " %6s" TimerOutputs.prettytime(entry_ns - cb.prev_time)
    end

    format_log8(e) = @sprintf "%8.2f" log10(abs(e))

    Estr = (@sprintf "%+15.12f" round(cost, sigdigits = 13))[1:15]
    if isnan(cb.prev_cost)
        ΔC = " "^10
    else
        sign = cost < cb.prev_cost ? "  " : "+ "
        ΔC = sign * format_log8(cost - cb.prev_cost)
    end

    Δρstr = isnothing(Δρ) ? " "^9 : " " * format_log8(Δρ)

    # methods without a residual (e.g. SCF) leave all residual related fields empty
    resstr = haskey(info, :norm_res) && !isnothing(info.norm_res) ?
        " " * format_log8(info.norm_res) : " "^9
    # the restricted residual and the coarse error percentage only exist in RFMG itself;
    # other methods (2gRG called by RFMG on the finest level, single-grid methods
    # in a nested iteration) have no coarser grid to compare to
    is_rfmg = info.algorithm == "RFMG"
    Rresstr = is_rfmg && haskey(info, :norm_Rres) && !isnothing(info.norm_Rres) ?
        " " * format_log8(info.norm_Rres) : " "^9

    est_err = haskey(info, :est_err) ? info.est_err : nothing
    if isnothing(est_err) && !is_rfmg && haskey(info, :res) && !isnothing(info.res)
        est_err = invert_refinement_metric_improved(info.basis, info.ψ, info.res)
    end
    # methods with a reference error estimate (RFMG, nested iteration) provide its norm
    est_err_norm = haskey(info, :est_err_norm) ? info.est_err_norm :
                   isnothing(est_err) ? nothing : norm_DFTK(info.basis, est_err)
    errstr = !isnothing(est_err_norm) ? "  " * format_log8(est_err_norm) : " "^10
    diststr = ""
    dist = NaN
    dist_h1 = NaN
    if show_dist
        # methods like SCF also carry virtual orbitals: only compare the first p bands,
        # p being the number of bands of ψ_ref
        ψ = [ψk[:, 1:size(ψ_ref[ik], 2)] for (ik, ψk) in enumerate(info.ψ)]
        ψ_fine = info.basis === basis_ref ? ψ : prolongate_point(info.basis, basis_ref, ψ)
        dist = grassmann_dist_DFTK(basis_ref, ψ_fine, ψ_ref)
        dist_h1 = h1_dist_DFTK(basis_ref, ψ_fine, ψ_ref)
        diststr = "   " * (@sprintf "%11.2f" log10(dist)) * "   " * (@sprintf "%10.2f" log10(dist_h1))
    end
    percstr = haskey(info, :est_err_perc) && !isnothing(info.est_err_perc) ?
        (@sprintf "%7.2f" 100 * info.est_err_perc) : " "^7
    # the τ of SCF is the kinetic energy density, not a step size
    τstr = info.algorithm != "SCF" && haskey(info, :τ) && info.τ isa Real ?
        (@sprintf "%7.3f" info.τ) : " "^7
    Ecutstr = @sprintf "%6.1f" info.basis.Ecut

    print(lpadding)
    @printf "% 3d   %s   %s   %s   %s   %s   %s%s   %s   %s   %s   %s" info.n_iter Estr resstr Rresstr ΔC Δρstr errstr diststr percstr τstr Ecutstr tstr
    println()
    push!(cb.times, (entry_ns - cb.start_ns - cb.overhead_ns) / 1e9)
    push!(cb.dists, dist)
    push!(cb.dists_h1, dist_h1)
    push!(cb.n_iters, info.n_iter)
    push!(cb.Ecuts, info.basis.Ecut)

    cb.prev_cost = cost
    flush(stdout)
    cb.prev_time = time_ns()
    cb.overhead_ns += cb.prev_time - entry_ns
    return info
end