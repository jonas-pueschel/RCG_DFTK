The Riemannian conjugate gradient method is a method to calculate ground states of the Kohn-Sham minimization problem, implemented using [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 
This is the implementation from this [preprint](https://arxiv.org/abs/2503.16225). The experiments from the paper can be found in the [`paper` branch](https://github.com/jonas-pueschel/RCG_DFTK/tree/paper).

# Getting started
In julia you can get add the `RCG_DFTK` package by running
```julia
using Pkg; Pkg.add(path = "https://github.com/jonas-pueschel/RCG_DFTK.git")
```
It then can be used in Julia via
```julia
include RCG_DFTK
```
Note that `RCG_DFTK` can only be used in conjuction with [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 

# Usage
The package exports the functions `riemannian_conjugate_gradient`, `h1_riemannian_conjugate_gradient`, `energy_adaptive_riemannian_conjugate_gradient`, `h1_riemannian_gradient` and `energy_adaptive_riemannian_gradient`, where the latter four are just 
variants of the first with a respective selection of the parameters. They can be used similar to `self_consistent_field` from [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl), i.e. they take `basis::PlaneWaveBasis{T}` as argument and share many `kwargs...`. Note that in-place variants are not yet implemented. 

## Example

A brief example on the usage of methods from this package, a smiliar example can also be found in `experiments/simple_test.jl`. We emphasise that energy-adaptive methods usually need an initial guess that is close to the ground state in order to converge, while H1RCG will usually show global convergence. As a trade-off, energy-adaptive methods generally converge faster locally with respect to runtime.

```julia
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

```
