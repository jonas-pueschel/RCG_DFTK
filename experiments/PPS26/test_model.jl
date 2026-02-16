using PseudoPotentialData
using RCG_DFTK
using DFTK
using LinearAlgebra

# In order to make this result independent from precompile time,
# run the precompile_methods function before (otherwise times will be off)
include("../precompile_methods.jl")
precompile_methods()

include("../setups/silicon_setup.jl")


function test_model(; model_name = "silicon", initial_guess = "scf")


    if model_name == "silicon"
        model, basis = silicon_setup(; Ecut = 30, kgrid = [4, 4, 4], supercell_size = [1, 1, 1]);
    elseif model_name == "GaAs"
        model, basis = GaAs_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);
    elseif model_name == "TiO2"
        model, basis = TiO2_setup(;Ecut = 60, kgrid = [2,2,2], supercell_size = [1,1,1]);
    else
        throw("model name'$model_name' not known!")
    end

    percentages = []

    # Convergence we desire in the residual
    tol = 1.0e-8;

    #Initial value
    if initial_guess == "scf"
        scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
    else
        scfres_start = h1_riemannian_conjugate_gradient(basis; tol = 0.5e-1)
    end
    ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
    ρ1 = scfres_start.ρ;

    defaultCallback = RcgDefaultCallback();



    # H1RCG
    println("H1RCG")
    callback_h1rcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg2 = h1_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = callback_h1rcg,
        iteration_strat = StandardBacktracking(
            ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
            ApproxHessianStep(), 10
        )
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # L2RCG
    println("L2RCG")
    callback_l2rcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg2 = l2_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 120,
        callback = callback_l2rcg,
        iteration_strat = StandardBacktracking(
            ModifiedSecantRule(0.0, 0.25, 1.0e-12, 0.5),
            ApproxHessianStep(), 10
        )
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # EARCG
    println("EARCG")
    callback_earcg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = callback_earcg
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # EARG
    println("EARG")
    callback_earg = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg1 = energy_adaptive_riemannian_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = callback_earg
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # EARCG
    println("EARCG0")
    callback_earcg0 = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        shift = CorrectedRelativeΛShift(μ = 0.0),
        callback = callback_earcg0
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # EARG
    println("EARG0")
    callback_earg0 = ResidualEvalCallback(; defaultCallback, method = EvalRCG())

    DFTK.reset_timer!(DFTK.timer)
    scfres_rcg1 = energy_adaptive_riemannian_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        shift = CorrectedRelativeΛShift(μ = 0.0),
        callback = callback_earg0
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalRCG())]

    # SCF
    println("SCF")
    callback_scf = ResidualEvalCallback(; defaultCallback, method = EvalSCF())
    is_converged = ResidualEvalConverged(tol, callback_scf)

    DFTK.reset_timer!(DFTK.timer)
    scfres_scf = self_consistent_field(
        basis; tol,
        callback = callback_scf,
        is_converged = is_converged,
        ψ = ψ1, ρ = ρ1,
        maxiter = 100
    );
    println(DFTK.timer)
    percentages = [percentages..., RCG_DFTK.get_ham_time(EvalSCF())]


    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]

    norm_res_0 = norm(DFTK.compute_projected_gradient(basis, ψ1, occupation))

    return callback_h1rcg, callback_l2rcg, callback_earcg, callback_earg, callback_earg0, callback_earcg0, callback_scf, norm_res_0, percentages
end