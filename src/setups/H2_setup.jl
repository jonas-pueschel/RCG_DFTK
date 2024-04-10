function H2_setup(;Ecut = 15)
    kgrid = [1, 1, 1]       # k-point grid
    #Ecut = 5                # kinetic energy cutoff in Hartree
    
    #TODO this isnt right!
    a = 15.23                  # lattice constant in Bohr
    lattice = a * I(3)
    H = ElementPsp(:H; psp=load_psp("hgh/lda/h-q1"));
    atoms = [H, H];
    positions = [ones(3)/(10 * sqrt(3)), -ones(3)/(10 * sqrt(3))]

    model = model_LDA(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    return [model, basis]
end