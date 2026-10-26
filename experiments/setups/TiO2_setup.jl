using DFTK
using PseudoPotentialData
include(joinpath(@__DIR__, "setup_utils.jl"))

function TiO2_setup(; Ecut = 50, kgrid = [2, 2, 2], supercell_size = [1, 1, 1],
                     supersampling = 2, model_terms = :lda)
    # same GTH pseudopotentials as the former hgh/lda/ti-q4 and hgh/lda/o-q6 (PBE versions for :pbe)
    family = psp_family("largecore", model_terms)
    Ti = ElementPsp(:Ti, family)
    O = ElementPsp(:O, family)
    atoms = [Ti, Ti, O, O, O, O]
    positions = [
        [0.5, 0.5, 0.5],  # Ti
        [0.0, 0.0, 0.0],  # Ti
        [0.19542, 0.80458, 0.5],  # O
        [0.80458, 0.19542, 0.5],  # O
        [0.30458, 0.30458, 0.0],  # O
        [0.69542, 0.69542, 0.0],
    ]  # O
    #positions[1] .+= [0.22, -0.28, 0.35] / 5
    lattice = [
        [8.79341  0.0      0.0];
        [0.0      8.79341  0.0];
        [0.0      0.0      5.61098]
    ]

    #model = model_PBE(lattice, atoms, positions)

    model_prim = setup_model(lattice, atoms, positions; model_terms)
    model = model_prim
    if (prod(supercell_size) != 1)
        lattice, atoms, positions = create_supercell(lattice, atoms, positions, supercell_size)
        model = setup_model(lattice, atoms, positions; model_terms)
    end

    return setup_bases(model_prim, model, Ecut, kgrid, supercell_size; supersampling)
end
