The Riemannian conjugate gradient method is a method to calculate ground states of the Kohn-Sham minimization problem, implemented using [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 
This is the implementation from this [preprint](https://arxiv.org/abs/2503.16225). This branch is dedicated to preserve the experiments from that paper, for the current version, refer to the `main` branch.

# Getting started
In julia you can get started by installing version 0.7.21 of `DFTK` and this branch of the repository
```julia
using Pkg;
Pkg.add(Pkg.PackageSpec(;name="DFTK", version="0.7.21"));
Pkg.add(url = "https://github.com/jonas-pueschel/RCG_DFTK.git", rev = "paper");
```

# Running the Experiments

Main purpose of this branch is to run the experiemtns from the [preprint](https://arxiv.org/abs/2503.16225). One first needs to locally save the folder `./experiments` and its content locally, since the experiments are not contained in the `src` folder of the package but stand-alone.

## Comparisons of Methods
In order to compare the methods, running
```
julia PATH/TO/experiments/PPS26/generate_plots_models.jl 
```
generates the Iterations, Hamiltonian and Times `.tex` files of the plot for all three models and also generates the table with the percentage of runtimes caused by Hamiltonian multiplications for all methods and models. In the lines 9-11 of the script, the user can manually set which methods and models should be run. We note that the `method_ids` object points to the ordering of the results from the internally called `test_model` function, and these should fit the respective name (no manual checks if the user input is valid, the user must give consistent values). 

The method `test_model` from `test_model.jl` runs the methods (EARCG-St, EARCG-Gr, EARG-St, EARG-Gr, H1RCG, L2RCG, SCF) for the model, given as key word argument `model_name`. Implemented are `"silicon", "GaAs", "TiO2"`. It returns `ResidualEvalCallback` objects for each method (these have collected all relevant data), the norm of the initial residual `norm_res_0` and the percentages of runtime caused by Hamiltonian application. 

## Testing for Small Gaps

In order to run the small gaps test, running
```
julia PATH/TO/experiments/PPS26/generate_plots_gaps.jl 
```
generates the gaps plot and the two performance plots of the methods for the gaps as `.tex` files. In lines 11, 12, the user can manually set the range of `as` and `n_examples`. 

Internally, it calls the method `test_gaps` from `test_gaps.jl`. Given key word argument `as` and `n_examples`, it runs for each lattice constant `a` in collection `as` the methods (EARCG-St, EARCG-Gr, EARG-St, EARG-Gr, H1RCG, L2RCG, SCF)  `n_example` times and returns a 2-dim array of `ResidualEvalCallback` objects for each method (array indices are index of `a` and number of run) as well as the HOMO-LUMO gaps and the effective gaps (read from the SCF result) for the different values of `a`. It also calls `get_virtual_gaps` from `calc_gaps.jl`, which calculates the SCF virtual gap for each value of `a`.

# General Usage
The package exports the functions `riemannian_conjugate_gradient`, `h1_riemannian_conjugate_gradient`, `energy_adaptive_riemannian_conjugate_gradient`, `h1_riemannian_gradient` and `energy_adaptive_riemannian_gradient`, where the latter four are just 
variants of the first with a respective selection of the parameters. They can be used similar to `self_consistent_field` from [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl), i.e. they take `basis::PlaneWaveBasis{T}` as argument and share many `kwargs...`. Note that in-place variants are not yet implemented. 

## Example

A brief example on the usage of methods from this package, a similar can also be found in `experiments/simple_test.jl`. We emphasise that energy-adaptive methods usually need an initial guess that is close to the ground state in order to converge, while H1-RCG will usually show global convergence. As a trade-off, energy-adaptive methods generally converge faster locally with respect to runtime.

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
