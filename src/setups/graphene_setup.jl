function  graphene_setup(;Ecut = 15)
    L = 20  # height of the simulation box
    kgrid = [6, 6, 1]


    # Define the geometry and pseudopotential
    a = 4.66  # lattice constant
    a1 = a*[1/2,-sqrt(3)/2, 0]
    a2 = a*[1/2, sqrt(3)/2, 0]
    a3 = L*[0  , 0        , 1]
    lattice = [a1 a2 a3]
    C1 = [1/3,-1/3,0.0]  # in reduced coordinates
    C2 = -C1
    positions = [C1, C2]
    C = ElementPsp(:C; psp=load_psp("hgh/pbe/c-q4"))
    atoms = [C, C]

    # Run SCF
    model = model_PBE(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut, kgrid)

    return [model, basis]
end