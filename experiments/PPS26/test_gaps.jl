using PseudoPotentialData
using RCG_DFTK
using DFTK
using LinearAlgebra

include("../setups/silicon_setup.jl")
include("./run_method.jl")


function test_gaps(; as = 10:0.1:11.4, n_examples = 10, method_names = ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"])

    callbacks = [[] for m = method_names]

    gaps = [0.0 for a = as]
    gaps_eff = [0.0 for a = as]
    i = 0
    for a = as
        i += 1
        println("######################################################")
        println(" a = $a")
        println("######################################################")


        for im = 1:length(method_names)
            callbacks[im] = [callbacks[im]..., []]
        end



        for n_example = 1:n_examples

            model, basis = silicon_setup(; Ecut = 30, kgrid = [4, 4, 4], supercell_size = [1, 1, 1], a);

            # Convergence we desire in the residual
            tol = 1.0e-8;

            #Initial value
            #scfres_start = h1_riemannian_conjugate_gradient(basis; tol = 0.5e-1)
            scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
            ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
            ρ1 = scfres_start.ρ;

            for im = 1:length(method_names)
                method_name = method_names[im]
                if method_name == "SCF" && n_example == 1
                    callback, ~, gap, gap_eff = run_method(basis, ψ1, ρ1, tol, method_name; get_gap = true)
                    gaps[i] = gap
                    gaps_eff[i] = gap_eff
                else
                    callback, ~ = run_method(basis, ψ1, ρ1, tol, method_name)
                end
                callbacks[im][end] = [callbacks[im][end]..., callback] 

            end
        end
    end
    return (callbacks, gaps, gaps_eff)
end