using DFTK
using RCG_DFTK
using PseudoPotentialData


# this script forces precompliation for all methods, ensuring comparability in runtime
# include("precompile_methods.jl");
# precompile_methods();

include("setups/silicon_setup.jl")
# Silicon lattice constant in Bohr

function rand_orb(basis::PlaneWaveBasis{T}) where {T}
    model = basis.model

    # check that there are no virtual orbitals
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    return [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints]
end

function calculate_density(basis::PlaneWaveBasis{T}, ψ) where {T}
    # check that there are no virtual orbitals
    model = basis.model
    filled_occ = DFTK.filled_occupation(model)
    n_spin = model.n_spin_components
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp)
    # number of kpoints and occupation
    Nk = length(basis.kpoints)

    occupation = [filled_occ * ones(T, n_bands) for ik in 1:Nk]
   
    return DFTK.compute_density(basis, ψ , occupation)
end


model_f, basis_f = silicon_setup(; Ecut = 20, kgrid = [4,4,4]);
model_c, basis_c = silicon_setup(; Ecut = 15, kgrid = [4,4,4]);

# Convergence tolerance
tol = 1.0e-8;

# Initial value
ψ1_c = rand_orb(basis_c)
ρ1_c = calculate_density(basis_c, ψ1_c)


ψ1_f = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c);
ρ1_f = DFTK.interpolate_density(ρ1_c, basis_c, basis_f);

scfres_rcg1 = h1_riemannian_conjugate_gradient(basis_c; ψ = ψ1_c, ρ = ρ1_c, 
    tol = 1e-4, callback = (info) -> nothing);

ψ1_c2 = scfres_rcg1.ψ
ρ1_c2 = scfres_rcg1.ρ;

ψ1_f2 = RCG_DFTK.interpolate_c2f(basis_c, basis_f, ψ1_c2);
ρ1_f2 = DFTK.interpolate_density(ρ1_c2, basis_c, basis_f);


scfres_rcg2 = h1_riemannian_conjugate_gradient(basis_f; ψ = ψ1_f, ρ = ρ1_f, tol, 
    callback = (info) -> nothing);

println("Steps without multilevel: $(scfres_rcg2.n_iter)")

scfres_rcg3 = h1_riemannian_conjugate_gradient(basis_f; ψ = ψ1_f2, ρ = ρ1_f2, tol, 
callback = (info) -> nothing);
println("Steps with multilevel: $(scfres_rcg3.n_iter)")