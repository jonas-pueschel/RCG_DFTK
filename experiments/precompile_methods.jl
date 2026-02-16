using PseudoPotentialData

include("setups/silicon_setup.jl")

function precompile_methods()
    println("Running each method once to ensure they are precompiled...")
    
    model, basis = silicon_setup(; Ecut = 10, kgrid = [1, 1, 1], supercell_size = [1, 1, 1]);


    # Convergence we desire in the residual
    tol = 1.0e-2;

    #Initial value
    scfres_start = DFTK.self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model), callback = (info) -> ());
    ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
    ρ1 = scfres_start.ρ;

    scfres_rcg2 = h1_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = (info) -> ()
    );

    scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = (info) -> ()
    );

    scfres_rcg1 = energy_adaptive_riemannian_gradient(
        basis;
        ψ = ψ1, ρ = ρ1,
        tol, maxiter = 100,
        callback = (info) -> ()
    );


    scfres_scf = self_consistent_field(
        basis; tol,
        callback = (info) -> (),
        ψ = ψ1, ρ = ρ1,
        maxiter = 100
    );

    println("finished.")
end