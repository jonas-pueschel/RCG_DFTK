function TiO2_setup(; Ecut = 25)
    Ti = ElementPsp(:Ti, psp=load_psp("hgh/lda/ti-q4.hgh"))
    O  = ElementPsp(:O, psp=load_psp("hgh/lda/o-q6.hgh"))
    atoms     = [Ti, Ti, O, O, O, O]
    positions = [[0.5,     0.5,     0.5],  # Ti
                 [0.0,     0.0,     0.0],  # Ti
                 [0.19542, 0.80458, 0.5],  # O
                 [0.80458, 0.19542, 0.5],  # O
                 [0.30458, 0.30458, 0.0],  # O
                 [0.69542, 0.69542, 0.0]]  # O
    positions[1] .+= [0.22, -0.28, 0.35] / 5
    lattice   = [[8.79341  0.0      0.0];
                 [0.0      8.79341  0.0];
                 [0.0      0.0      5.61098]];
    
    model = model_LDA(lattice, atoms, positions)
    kgrid = [2, 2, 2]  # k-point grid (Regular Monkhorst-Pack grid)
    ss = 4
    #fft_size = compute_fft_size(model, 25; supersampling=ss)
    basis = PlaneWaveBasis(model; Ecut, kgrid)#, fft_size=fft_size)
    return [model, basis]
end