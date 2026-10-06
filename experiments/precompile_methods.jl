using PseudoPotentialData
include("setups/silicon_setup.jl")

function precompile_methods()
    println("run each method once to ensure they are precompiled")
    # Silicon lattice constant in Bohr
    model_f, basis_f = silicon_setup(; Ecut = 70, kgrid = [4,4,4]);
    model_c, basis_c = silicon_setup(; Ecut = 15, kgrid = [4,4,4]);


    # Convergence tolerance
    tol = 1.0e-6;

    # Initial value
    scfres_start = self_consistent_field(basis_c; tol = 0.5e-1, 
        callback = (info) -> nothing, nbandsalg = DFTK.FixedBands(basis_c.model));
    ψ1_c = DFTK.select_occupied_orbitals(basis_c, scfres_start.ψ, scfres_start.occupation).ψ;
    ρ1_c = scfres_start.ρ
    ψ1 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c)
    ρ1 = DFTK.interpolate_density(ρ1_c, basis_c, basis_f)

    # scfres_ref = self_consistent_field(basis_f; tol = 1e-10);
    # e_ref = scfres_ref.energies.total 

    es0 ,res0 = RCG_DFTK.init_E_res(ψ1, ρ1, basis_f)

    init_norm_res = RCG_DFTK.norm_DFTK(basis_f, res0)
    init_e = es0.total


    # multilevel
    default_callback = (info) -> nothing


    ea_coarse_solver(basis_c) = function (ψ_c, ρ_c, Rres, tol)
        callback = (info) -> nothing
        cost_resiudal = CoarseGridCostResidual(ψ_c, Rres)
        is_converged = RcgConvergenceResidualMGH(tol, 0.1, ψ_c)
        result = riemannian_conjugate_gradient(basis_c; ρ = ρ_c, ψ = ψ_c, 
            cost_resiudal,
            is_converged,
            callback,
            do_rayleigh_ritz = false,
        )
        return result.ψ
    end

    cb0 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_rcg0 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
        ψ = ψ1, ρ = ρ1, tol, 
        callback = cb0,
        #multilevel_map = ProjectionMap(),
        gradient = EAGradient(basis_f, CorrectedRelativeΛShift(μ = 0.0)),
        check_convergence_early = true,
        maxiter = 3,
        coarse_tol = RCG_DFTK.RelativeResTolerance(1e-1, tol),
        coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.3, tol),
        iteration_strat_coarse = AdaptiveBacktracking(
            ModifiedSecantRule(0.05, 0.1, 1.0e-12, 0.5),
            ConstantStep(1.0), 10
        ),
        );
    #println(cb0.times_tot[end] / 1e9)

    cb1 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_rcg1 = RCG_DFTK.two_level_riemannian_optimization(basis_c, basis_f;
        ψ = ψ1, ρ = ρ1, tol, 
        callback = cb1,
        check_convergence_early = true,
        coarse_density = RCG_DFTK.RecalculateDensity(),
        maxiter = 3,
        coarse_tol = RCG_DFTK.RelativeResTolerance(1e-1, tol),
        coarse_cond = RCG_DFTK.ToleranceMinStepCoarseCondition(0.3, tol),
        );
    #println(cb1.times_tot[end] / 1e9)


    cb2 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_rcg2 = RCG_DFTK.h1_riemannian_conjugate_gradient(basis_f;
        callback = cb2, ψ = ψ1, ρ = ρ1, tol,
        maxiter = 3);
    #println(cb2.times_tot[end] / 1e9)

    cb3 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_rcg3 = RCG_DFTK.h1_riemannian_gradient(basis_f;
        callback = cb3, ψ = ψ1, ρ = ρ1, tol,
        maxiter = 3);
    #println(cb3.times_tot[end] / 1e9)

    cb4 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_rcg4 = RCG_DFTK.energy_adaptive_riemannian_conjugate_gradient(basis_f;
        callback = cb4, ψ = ψ1, ρ = ρ1, tol,
        maxiter = 3);
    #println(cb4.times_tot[end] / 1e9)

    cb5 = TrackResTimeCallback(default_callback, init_norm_res, init_e)
    scfres_scf = self_consistent_field(basis_f;
        callback = cb5, ψ = ψ1, ρ = ρ1, tol = 1e-7,
        maxiter = 3);
    #println(cb5.times_tot[end] / 1e9)

    println("done.")
end