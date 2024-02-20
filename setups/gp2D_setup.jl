using StaticArrays

function gp2D_setup(;Ecut = 20)
    a = 15
    lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]];

    # Confining scalar potential, and magnetic vector potential
    pot(x, y, z) = ((x - a/2)^2 + (y - a/2)^2)/2
    ω = .0
    Apot(x, y, z) = ω * @SVector [y - a/2, -(x - a/2), 0]
    Apot(X) = Apot(X...);


    # Parameters
    η = 200
    C = η/2
    α = 2
    n_electrons = 1;  # Increase this for fun

    # Collect all the terms, build and run the model
    terms = [Kinetic(),
            ExternalFromReal(X -> pot(X...)),
            LocalNonlinearity(ρ -> C * ρ^α),
            Magnetic(Apot),
    ]
    model = Model(lattice; n_electrons, terms, spin_polarization=:spinless)  # spinless electrons
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    return[model, basis]
end