using DFTK
using LinearAlgebra

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

model, basis = graphene_setup(Ecut = 15);

# Convergence we desire in the density
tol = 1e-6

filled_occ = DFTK.filled_occupation(model)
n_spin = model.n_spin_components
n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)


ψ1 = [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];

#scf messes up dimensions
scfres_start = self_consistent_field(basis; ψ = ψ1 , maxiter = 1);
ψ2 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;

# different dims in the last kpoint!
println(length(ψ1[length(ψ1)])) #4248
println(length(ψ2[length(ψ2)])) #sometimes 4248, sometimes 5310