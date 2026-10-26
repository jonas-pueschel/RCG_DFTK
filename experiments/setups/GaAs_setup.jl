using DFTK
using PseudoPotentialData
include(joinpath(@__DIR__, "setup_utils.jl"))

function GaAs_setup(; Ecut = 25, a = 10.68290949909, kgrid = [2, 2, 2], supercell_size = [1, 1, 1],
                     supersampling = 2, model_terms = :lda)
    # GaAs lattice constant in Bohr
    lattice = a / 2 * [
        [0 1 1.0];
        [1 0 1.0];
        [1 1 0.0]
    ]

    # same GTH pseudopotentials as the former hgh/lda/ga-q3 and hgh/lda/as-q5 (PBE versions for :pbe)
    family = psp_family("largecore", model_terms)
    Ga = ElementPsp(:Ga, family)
    As = ElementPsp(:As, family)
    atoms = [Ga, As]
    positions = [
        ones(3) / 8 #+ [0.24, -0.33, 0.12] / 15
        , -ones(3) / 8,
    ]

    model_prim = setup_model(lattice, atoms, positions; model_terms)
    #model_prim = model_PBE(lattice, atoms, positions)
    model = model_prim
    if (prod(supercell_size) != 1)
        lattice, atoms, positions = create_supercell(lattice, atoms, positions, supercell_size)
        model = setup_model(lattice, atoms, positions; model_terms)
    end

    return setup_bases(model_prim, model, Ecut, kgrid, supercell_size; supersampling)
end
