function  iron_setup(;Ecut = 25, a = 5.42352, kgrid = [3, 3, 3]  )
    lattice = a / 2 * [[-1  1  1];
                    [ 1 -1  1];
                    [ 1  1 -1]]
    atoms     = [ElementPsp(:Fe; psp=load_psp("hgh/lda/Fe-q8.hgh"))]
    positions = [zeros(3)];

    model_nospin = model_LDA(lattice, atoms, positions, temperature=0.01)
    basis_nospin = PlaneWaveBasis(model_nospin; kgrid, Ecut)
    return [model_nospin, basis_nospin]
end