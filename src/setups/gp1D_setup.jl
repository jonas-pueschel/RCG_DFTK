function gp1D_setup(;Ecut = 500)
    a = 10
    lattice = a .* [[1 0 0.]; [0 0 0]; [0 0 0]];
    pot(x) = (x - a/2)^2;
    C = 1.0
    α = 2;

    n_electrons = 1  # Increase this for fun
    terms = [Kinetic(),
            ExternalFromReal(r -> pot(r[1])),
            LocalNonlinearity(ρ -> C * ρ^α),
    ]
    model = Model(lattice; n_electrons, terms, spin_polarization=:spinless); 
    basis = PlaneWaveBasis(model, Ecut=Ecut, kgrid=(1, 1, 1))
    return  [model, basis]
end