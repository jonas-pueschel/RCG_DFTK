using DFTK
using PseudoPotentialData
include(joinpath(@__DIR__, "setup_utils.jl"))

function silicon_setup(; Ecut = 20, a = 10.26, kgrid = [2, 2, 2], supercell_size = [1, 1, 1],
                       supersampling = 2)
    # Silicon lattice constant in Bohr
    lattice = a / 2 * [
        [0 1 1.0];
        [1 0 1.0];
        [1 1 0.0]
    ]
    Si = ElementPsp(:Si, PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"))
    atoms = [Si, Si]
    positions = [ones(3) / 8, -ones(3) / 8]

    model_prim = model_LDA(lattice, atoms, positions)
    #model_prim = model_PBE(lattice, atoms, positions)
    model = model_prim
    if (prod(supercell_size) != 1)
        lattice, atoms, positions = create_supercell(lattice, atoms, positions, supercell_size)
        model = model_LDA(lattice, atoms, positions)
    end

    return setup_bases(model_prim, model, Ecut, kgrid, supercell_size; supersampling)
end
