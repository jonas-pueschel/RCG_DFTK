function GaAs_setup(; Ecut = 25, a = 10.68290949909, kgrid = [2, 2, 2], supercell_size = [1, 1, 1])
    # GaAs lattice constant in Bohr
    lattice = a / 2 * [
        [0 1 1.0];
        [1 0 1.0];
        [1 1 0.0]
    ]

    Ga = ElementPsp(:Ga, psp = load_psp("hgh/lda/ga-q3"))
    As = ElementPsp(:As, psp = load_psp("hgh/lda/as-q5"))
    atoms = [Ga, As]
    positions = [
        ones(3) / 8 #+ [0.24, -0.33, 0.12] / 15
        , -ones(3) / 8,
    ]

    if (prod(supercell_size) != 1)
        lattice, atoms, positions = create_supercell(lattice, atoms, positions, supercell_size)
    end
    model = model_LDA(lattice, atoms, positions)
    #model = model_PBE(lattice, atoms, positions)
    # k-point grid (Regular Monkhorst-Pack grid)
    ss = 4
    fft_size = compute_fft_size(model, Ecut; supersampling = ss)
    basis = PlaneWaveBasis(model; Ecut = Ecut, kgrid, fft_size = fft_size)
    return [model, basis]
end
