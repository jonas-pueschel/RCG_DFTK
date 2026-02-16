using PseudoPotentialData
using RCG_DFTK
using DFTK
using LinearAlgebra

include("../setups/silicon_setup.jl")

function test_gaps(; as = 10:0.1:11.4, n_examples = 10)
    callbacks_l2rcg = []
    callbacks_h1rcg = []
    callbacks_earcg = []
    callbacks_earg = []
    callbacks_earcg0 = []
    callbacks_earg0 = []
    callbacks_scf = []
    gaps = [0.0 for a = as]
    gaps_eff = [0.0 for a = as]
    i = 0
    for a = as
        i += 1
        println("######################################################")
        println(" a = $a")
        println("######################################################")

        callbacks_l2rcg = [callbacks_l2rcg..., []]
        callbacks_h1rcg = [callbacks_h1rcg..., []]
        callbacks_earcg = [callbacks_earcg..., []]
        callbacks_earg = [callbacks_earg..., []]
        callbacks_earcg0 = [callbacks_earcg0..., []]
        callbacks_earg0 = [callbacks_earg0..., []]
        callbacks_scf = [callbacks_scf..., []]

        for n_example = 1:n_examples

        model, basis = silicon_setup(; Ecut = 30, kgrid = [4, 4, 4], supercell_size = [1, 1, 1], a);



        # Convergence we desire in the residual
        tol = 1.0e-8;

        #Initial value
        #scfres_start = h1_riemannian_conjugate_gradient(basis; tol = 0.5e-1)
        scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
        ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
        ρ1 = scfres_start.ρ;

        defaultCallback = RcgDefaultCallback();

        # L2RCG
        println("L2RCG")
        callback_l2rcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg1 = l2_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 1000,
            callback = callback_l2rcg,
            iteration_strat = StandardBacktracking(
                ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
                ApproxHessianStep(), 10
            )
        );
        println(DFTK.timer)
        callbacks_l2rcg[end] = [callbacks_l2rcg[end]..., callback_l2rcg]

        # H1RCG
        println("H1RCG")
        callback_h1rcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg2 = h1_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 1000,
            callback = callback_h1rcg,
            iteration_strat = StandardBacktracking(
                ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
                ApproxHessianStep(), 10
            )
        );
        println(DFTK.timer)
        callbacks_h1rcg[end] = [callbacks_h1rcg[end]..., callback_h1rcg]

        # EARCG
        println("EARCG")
        callback_earcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg3 = energy_adaptive_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            callback = callback_earcg
        );
        println(DFTK.timer)
        callbacks_earcg[end] = [callbacks_earcg[end]..., callback_earcg]

        # EARG
        println("EARG")
        callback_earg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg4 = energy_adaptive_riemannian_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            callback = callback_earg
        );
        println(DFTK.timer)
        callbacks_earg[end] = [callbacks_earg[end]..., callback_earg]

        # EARCG
        println("EARCG0")
        callback_earcg0 = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg3 = energy_adaptive_riemannian_conjugate_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            shift = CorrectedRelativeΛShift(μ = 0.0),
            callback = callback_earcg0
        );
        println(DFTK.timer)
        callbacks_earcg0[end] = [callbacks_earcg0[end]..., callback_earcg0]

        # EARG
        println("EARG0")
        callback_earg0 = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

        DFTK.reset_timer!(DFTK.timer)
        scfres_rcg4 = energy_adaptive_riemannian_gradient(
            basis;
            ψ = ψ1, ρ = ρ1,
            tol, maxiter = 200,
            shift = CorrectedRelativeΛShift(μ = 0.0),
            callback = callback_earg0
        );
        println(DFTK.timer)
        callbacks_earg0[end] = [callbacks_earg0[end]..., callback_earg0]

        try
        # SCF
        println("SCF")
        callback_scf = ResidualEvalCallback(; defaultCallback, method = EvalSCF())
        is_converged = ResidualEvalConverged(tol, callback_scf)

        DFTK.reset_timer!(DFTK.timer)
        scfres = self_consistent_field(
            basis; tol,
            callback = callback_scf,
            is_converged = is_converged,
            ψ = ψ1, ρ = ρ1,
            maxiter = 100
        );
        println(DFTK.timer)
        if n_example == 1 || gaps[i] == 0.0
            gaps[i] = min([evs[scfres.n_bands_converge + 1] for evs = scfres.eigenvalues]...) - max([evs[scfres.n_bands_converge] for evs = scfres.eigenvalues]...)
            gaps_eff[i] = min([evs[scfres.n_bands_converge + 1] - evs[scfres.n_bands_converge] for evs = scfres.eigenvalues]...)
        end

        callbacks_scf[end] = [callbacks_scf[end]..., callback_scf]
        catch e
           println(e) 
        end
    end
    end
    return (callbacks_l2rcg, callbacks_h1rcg, callbacks_earcg, callbacks_earg, callbacks_earcg0, callbacks_earg0, callbacks_scf, as, gaps, gaps_eff)
end