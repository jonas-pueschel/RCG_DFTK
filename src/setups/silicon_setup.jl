
function silicon_setup(; Ecut = 20, a = 10.26, kgrid=[6, 6, 6])
    # Silicon lattice constant in Bohr
    lattice = a / 2 * [[0 1 1.];
                    [1 0 1.];
                    [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]

    model = model_LDA(lattice, atoms, positions)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid)
    return [model, basis]
end