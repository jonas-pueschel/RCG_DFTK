The Riemannian conjugate gradient method is a method to calculate ground states of the Kohn-Sham minimization problem, implemented using [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 
This is the implementation from this [preprint](https://arxiv.org/abs/2503.16225).

# Disclaimer

[DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl) is in the process to adpot Riemannian solvers from [Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl/), see this [PR](https://github.com/JuliaMolSim/DFTK.jl/pull/1105). We hope that this also allows to integrate energy-adaptive methods, thus there are no plans to directly integrate code from this repository into [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl) and it will probably not be maintained in the future. Up to that point, it will however allow the use of energy-adaptive Riemannian methods with [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl).

# Getting started
In julia you can get started by just typing
```julia
using Pkg; Pkg.add(path = "https://github.com/jonas-pueschel/RCG_DFTK.git")
```

# Usage
The package exports the functions `riemannian_conjugate_gradient`, `h1_riemannian_conjugate_gradient`, `energy_adaptive_riemannian_conjugate_gradient`, `h1_riemannian_gradient` and `energy_adaptive_riemannian_gradient`, where the latter four are just 
variants of the first with a respective selection of the parameters. They can be used similar to `self_consistent_field` from [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl), i.e. they take `basis::PlaneWaveBasis{T}` as argument and share many `kwargs...`. Note that in-place variants are not yet implemented. 

## Example

A brief example on the usage of methods from this package, it can also be found in `experiments/simple_test.jl`. We emphasise that energy-adaptive methods usually need an initial guess that is close to the ground state in order to converge, while H1-RCG will usually show global convergence. As a trade-off, energy-adaptive methods generally converge faster locally with respect to runtime.

```julia
using DFTK
using RCG_DFTK
using PseudoPotentialData

# Silicon lattice constant in Bohr
a = 10.26
lattice = a / 2 * [[0 1 1.];
                [1 0 1.];
                [1 1 0.]]
Si = ElementPsp(:Si; psp=load_psp(PseudoFamily("cp2k.nc.sr.lda.v0_1.semicore.gth"), :Si))
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]
model = model_LDA(lattice, atoms, positions)
basis = PlaneWaveBasis(model; Ecut = 20, kgrid = [2, 2, 2])

# Convergence tolerance
tol = 1e-8;

# Initial value
scfres_start = self_consistent_field(basis; tol = 0.5e-1,  nbandsalg = DFTK.FixedBands(model));
ψ1 = DFTK.select_occupied_orbitals(basis, scfres_start.ψ, scfres_start.occupation).ψ;
ρ1 = scfres_start.ρ;

# EARCG
scfres_rcg1 = energy_adaptive_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol);

# H1RCG
scfres_rcg2 = h1_riemannian_conjugate_gradient(basis; ψ = ψ1, ρ = ρ1, tol);

# SCF
scfres_scf = self_consistent_field(basis; ψ = ψ1, ρ = ρ1, tol);
```
