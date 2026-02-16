The Riemannian conjugate gradient method is a method to calculate ground states of the Kohn-Sham minimization problem, implemented using [DFTK.jl](https://github.com/JuliaMolSim/DFTK.jl). 
This is the implementation from this [preprint](https://arxiv.org/abs/2503.16225). This branch is dedicated to preserve the experiments from that paper, for the current version, refer to the `main` branch.

# Getting started

Main purpose of this branch is to run the experiemtns from the [preprint](https://arxiv.org/abs/2503.16225). One first needs to locally save the directory `./experiments` and its contents, since the experiments are not contained in the `src` folder of the package but stand-alone.

After navigating the directory `experiment`, one needs to create a Julia-enviroment by in the Julia REPL 
```julia
]activate .
]instantiate
]add DFTK@0.7.21
]add PseudoPotentialData@0.2.4
]add https://github.com/jonas-pueschel/RCG_DFTK.git#paper
```
Then, the experiment scripts can be run in that enviroment.


## Comparisons of Methods
In order to compare the methods, running (from the `experiments` directory)
```
julia --project=. PPS26/generate_plots_models.jl 
```
generates the Iterations, Hamiltonian and Times `.tex` files of the plot for all three models and also generates the table with the percentage of runtimes caused by Hamiltonian multiplications for all methods and models. In the lines 9-11 of the script, the user can manually set which methods and models should be run. We note that the `method_ids` object points to the ordering of the results from the internally called `test_model` function, and these should fit the respective name (no manual checks if the user input is valid, the user must give consistent values). 

The method `test_model` from `test_model.jl` runs the methods (EARCG-St, EARCG-Gr, EARG-St, EARG-Gr, H1RCG, L2RCG, SCF) for the model, given as key word argument `model_name`. Implemented are `"silicon", "GaAs", "TiO2"`. It returns `ResidualEvalCallback` objects for each method (these have collected all relevant data), the norm of the initial residual `norm_res_0` and the percentages of runtime caused by Hamiltonian application. 

## Testing for Small Gaps

In order to run the small gaps test, running (from the `experiments` directory)
```
julia --project=. PPS26/generate_plots_gaps.jl 
```
generates the gaps plot and the two performance plots of the methods for the gaps as `.tex` files. In lines 11, 12, the user can manually set the range of `as` and `n_examples`. 

Internally, it calls the method `test_gaps` from `test_gaps.jl`. Given key word argument `as` and `n_examples`, it runs for each lattice constant `a` in collection `as` the methods (EARCG-St, EARCG-Gr, EARG-St, EARG-Gr, H1RCG, L2RCG, SCF)  `n_example` times and returns a 2-dim array of `ResidualEvalCallback` objects for each method (array indices are index of `a` and number of run) as well as the HOMO-LUMO gaps and the effective gaps (read from the SCF result) for the different values of `a`. It also calls `get_virtual_gaps` from `calc_gaps.jl`, which calculates the SCF virtual gap for each value of `a`.
