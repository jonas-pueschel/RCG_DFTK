using DFTK
using RCG_DFTK

function run_method(basis, ψ1, ρ1, tol, method_name; get_gap = false)

    println(method_name)
    defaultCallback = RcgDefaultCallback();


    eval = method_name == "SCF" ? EvalSCF() : EvalRCG()
    callback = ResidualEvalCallback(; defaultCallback, method = eval)
    
    DFTK.reset_timer!(DFTK.timer)
    
    # L2RCG
    if (method_name == "L2RCG")
        scfres_rcg1 = l2_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 1000,
            callback = callback,
            iteration_strat = StandardBacktracking(
                ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
                ApproxHessianStep(), 10
            )
        );
    # H1RCG
    elseif (method_name == "H1RCG")
        scfres_rcg2 = h1_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 1000,
            callback = callback,
            iteration_strat = StandardBacktracking(
                ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
                ApproxHessianStep(), 10
            )
        );
    # EARCG
    elseif (method_name == "EARCG-St")
        scfres_rcg3 = energy_adaptive_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            callback = callback
        );
    # EARG
    elseif (method_name == "EARG-St")
        scfres_rcg4 = energy_adaptive_riemannian_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            callback = callback
        );
    # EARCG
    elseif (method_name == "EARCG-Gr")
        scfres_rcg5 = energy_adaptive_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            shift = CorrectedRelativeΛShift(μ = 0.0),
            callback = callback
        );
    # EARG
    elseif (method_name == "EARG-Gr")
        scfres_rcg6 = energy_adaptive_riemannian_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            shift = CorrectedRelativeΛShift(μ = 0.0),
            callback = callback_earg0
        );
    # SCF
    elseif (method_name == "SCF")
        is_converged = ResidualEvalConverged(tol, callback)
        scfres = self_consistent_field(
            basis; tol,
            callback = callback,
            is_converged = is_converged,
            ψ = ψ1, ρ = ρ1,
            maxiter = 100
        );

        if get_gap
            percentage = RCG_DFTK.get_ham_time(eval)
            println(DFTK.timer)
            gap = min([evs[scfres.n_bands_converge + 1] for evs = scfres.eigenvalues]...) - max([evs[scfres.n_bands_converge] for evs = scfres.eigenvalues]...)
            gap_eff = min([evs[scfres.n_bands_converge + 1] - evs[scfres.n_bands_converge] for evs = scfres.eigenvalues]...)
            return callback, percentage, gap, gap_eff
        end
    else
        throw("method name'$method_name' not known!")
    end
    percentage = RCG_DFTK.get_ham_time(eval)
    println(DFTK.timer)

    return callback, percentage
end
