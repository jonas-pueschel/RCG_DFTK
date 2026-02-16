using PseudoPotentialData
using RCG_DFTK
using DFTK
using LinearAlgebra

# In order to make this result independent from precompile time,
# run the precompile_methods function before (otherwise times will be off)
include("../precompile_methods.jl")
precompile_methods()


include("./run_method.jl")
include("../setups/silicon_setup.jl")
include("../setups/GaAs_setup.jl")
include("../setups/TiO2_setup.jl")


function test_model(; model_name = "silicon", method_names = ["EARCG-St", "EARCG-Gr", "H1RCG", "L2RCG", "SCF"])


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
    callbacks = []

    # Convergence we desire in the residual
    tol = 1.0e-8;

    #Initial value
    scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));

    ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
    ρ1 = scfres_start.ρ;

    defaultCallback = RcgDefaultCallback();

    for method_name = method_names
        callback, percentage = run_method(basis, ψ1, ρ1, tol, method_name)
        callbacks = [callbacks..., callback]
        
        percentages = [percentages..., percentage]

    end

    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    Nk = length(basis.kpoints)
    occupation = [filled_occ * ones(Float64, n_bands) for _ in 1:Nk]

    norm_res_0 = norm(DFTK.compute_projected_gradient(basis, ψ1, occupation))

    return callbacks, norm_res_0, percentages
end