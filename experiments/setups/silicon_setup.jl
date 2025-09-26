
function silicon_setup(; Ecut = 20, a = 10.26, kgrid=[2, 2, 2], supercell_size = [1,1,1])
    # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Si))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    if (prod(supercell_size) != 1)
        lattice, atoms, positions = create_supercell(lattice, atoms, positions, supercell_size)
    end

    model = model_LDA(lattice, atoms, positions)
    #model = model_PBE(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid)
    return [model, basis]
end
