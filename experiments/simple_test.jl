using DFTK
using RCG_DFTK
using PseudoPotentialData

# Silicon lattice constant in Bohr
a = 10.26
lattice = a / 2 * [
    [0 1 1.0];
    [1 0 1.0];
    [1 1 0.0]
]
Si = ElementPsp(:Si; psp = load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Si))
atoms = [Si, Si]
positions = [ones(3) / 8, -ones(3) / 8]
model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut = 20, kgrid = [2, 2, 2])

# Convergence tolerance
tol = 1.0e-8;

# Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1, nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

# EARCG
scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol);

# H1RCG
scfres_rcg2 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol);

# SCF
scfres_scf = self_consistent_field(basis; ψ = ψ1, ρ = ρ1, tol);
